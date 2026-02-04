"""Council agent wrapper."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_consil.api.schemas import AgentConfig, VotePosition
from ai_consil.council.roles import (
    ANALYSIS_PROMPT_TEMPLATE,
    ANSWER_PROMPT_TEMPLATE,
    QUESTION_PROMPT_TEMPLATE,
    VOTE_PROMPT_TEMPLATE,
    get_role_prompt,
)
from ai_consil.providers import get_provider
from ai_consil.providers.base import ProviderAdapter, ProviderMessage

logger = logging.getLogger(__name__)


@dataclass
class ParsedVote:
    """Parsed vote from agent response."""

    position: VotePosition
    confidence: float
    reasoning: str


@dataclass
class ParsedQuestion:
    """Parsed question from agent response."""

    to_agent: str
    question: str


@dataclass
class AgentState:
    """Mutable state for an agent during deliberation."""

    questions_asked: dict[int, int] = field(default_factory=dict)  # round -> count
    analyses: dict[int, str] = field(default_factory=dict)  # round -> analysis


class CouncilAgent:
    """A council agent that participates in deliberation.

    Wraps a provider adapter and manages the agent's participation
    in analysis, Q&A, and voting phases.
    """

    def __init__(
        self,
        config: AgentConfig,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize a council agent.

        Args:
            config: The agent configuration.
            temperature: Sampling temperature for completions.
            max_tokens: Maximum tokens for completions.
        """
        self.config = config
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.state = AgentState()

        # Get system prompt (custom or role-based)
        self.system_prompt = config.system_prompt or get_role_prompt(config.role) or ""

        # Initialize provider adapter
        self._adapter: ProviderAdapter | None = None

    @property
    def id(self) -> str:
        """The agent's ID."""
        return self.config.id

    @property
    def role(self) -> str:
        """The agent's role."""
        return self.config.role

    def _get_adapter(self) -> ProviderAdapter:
        """Lazily initialize the provider adapter."""
        if self._adapter is None:
            self._adapter = get_provider(
                self.config.provider,
                self.config.model,
            )
        return self._adapter

    async def _complete(self, messages: list[ProviderMessage]) -> str:
        """Generate a completion."""
        adapter = self._get_adapter()
        response = await adapter.complete(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.content

    def _build_messages(
        self,
        user_prompt: str,
        context: str | None = None,
    ) -> list[ProviderMessage]:
        """Build message list for completion."""
        messages = []

        if self.system_prompt:
            messages.append(ProviderMessage(role="system", content=self.system_prompt))

        content = user_prompt
        if context:
            content = f"{context}\n\n{user_prompt}"

        messages.append(ProviderMessage(role="user", content=content))

        return messages

    async def analyze(
        self,
        topic: str,
        context: str | None = None,
        round_num: int = 1,
    ) -> str:
        """Generate analysis for the topic.

        Args:
            topic: The topic to analyze.
            context: Optional context from previous rounds.
            round_num: The current round number.

        Returns:
            The agent's analysis.
        """
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            role=self.role,
            topic=topic,
            context=context or "No additional context.",
        )

        messages = self._build_messages(prompt)
        analysis = await self._complete(messages)

        self.state.analyses[round_num] = analysis
        return analysis

    async def ask_question(
        self,
        discussion: str,
        round_num: int,
        max_questions: int,
    ) -> ParsedQuestion | None:
        """Decide whether to ask a question and to whom.

        Args:
            discussion: Summary of discussion so far.
            round_num: The current round number.
            max_questions: Maximum questions allowed per round.

        Returns:
            A parsed question if the agent wants to ask one, None otherwise.
        """
        questions_asked = self.state.questions_asked.get(round_num, 0)

        if questions_asked >= max_questions:
            return None

        prompt = QUESTION_PROMPT_TEMPLATE.format(
            discussion=discussion,
            questions_asked=questions_asked,
            max_questions=max_questions,
        )

        messages = self._build_messages(prompt)
        response = await self._complete(messages)

        # Parse response
        if "NO_QUESTION" in response.upper():
            return None

        # Try to parse question format
        to_match = re.search(r"QUESTION_TO:\s*(\S+)", response, re.IGNORECASE)
        q_match = re.search(r"QUESTION:\s*(.+)", response, re.IGNORECASE | re.DOTALL)

        if to_match and q_match:
            self.state.questions_asked[round_num] = questions_asked + 1
            return ParsedQuestion(
                to_agent=to_match.group(1).strip(),
                question=q_match.group(1).strip(),
            )

        return None

    async def answer_question(
        self,
        from_agent: str,
        question: str,
        context: str,
    ) -> str:
        """Answer a question from another agent.

        Args:
            from_agent: The agent asking the question.
            question: The question being asked.
            context: Context from the discussion.

        Returns:
            The agent's answer.
        """
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            from_agent=from_agent,
            question=question,
            context=context,
        )

        messages = self._build_messages(prompt)
        return await self._complete(messages)

    async def vote(
        self,
        topic: str,
        discussion: str,
    ) -> ParsedVote:
        """Cast a vote on the topic.

        Args:
            topic: The topic being voted on.
            discussion: Summary of the discussion.

        Returns:
            The agent's parsed vote.
        """
        prompt = VOTE_PROMPT_TEMPLATE.format(
            topic=topic,
            discussion=discussion,
            role=self.role,
        )

        messages = self._build_messages(prompt)
        response = await self._complete(messages)

        return self._parse_vote(response)

    def _parse_vote(self, response: str) -> ParsedVote:
        """Parse a vote response from the agent.

        Expected format:
        VOTE: support|oppose|abstain
        CONFIDENCE: 0.0-1.0
        REASONING: text

        Also handles mock provider format:
        VOTE:support|CONFIDENCE:0.75|REASONING:text
        """
        # Try pipe-delimited format (mock provider)
        if "|" in response and "VOTE:" in response:
            parts = response.split("|")
            vote_data: dict[str, Any] = {}
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    vote_data[key.strip().upper()] = value.strip()

            position_str = vote_data.get("VOTE", "abstain").lower()
            confidence_str = vote_data.get("CONFIDENCE", "0.5")
            reasoning = vote_data.get("REASONING", "No reasoning provided.")

            try:
                position = VotePosition(position_str)
            except ValueError:
                position = VotePosition.ABSTAIN

            try:
                confidence = float(confidence_str)
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

            return ParsedVote(
                position=position,
                confidence=confidence,
                reasoning=reasoning,
            )

        # Try line-by-line format
        position = VotePosition.ABSTAIN
        confidence = 0.5
        reasoning = "No reasoning provided."

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith("VOTE:"):
                vote_str = line.split(":", 1)[1].strip().lower()
                try:
                    position = VotePosition(vote_str)
                except ValueError:
                    pass
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return ParsedVote(
            position=position,
            confidence=confidence,
            reasoning=reasoning,
        )

    def get_questions_asked(self, round_num: int) -> int:
        """Get number of questions asked this round."""
        return self.state.questions_asked.get(round_num, 0)

    def reset_round_state(self, round_num: int) -> None:
        """Reset state for a new round."""
        self.state.questions_asked[round_num] = 0
