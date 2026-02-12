"""Council orchestrator - main coordination loop."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from ai_consil.api.schemas import (
    DEFAULT_VOTE_OPTIONS,
    AgentAnalysis,
    AgentConfig,
    CouncilConfig,
    CouncilEvent,
    CouncilEventType,
    CouncilTrace,
    QuestionAnswer,
    RoundTrace,
    RoundVoteResult,
    Vote,
    VoteTally,
    VotingSchedule,
)
from ai_consil.council.agent import CouncilAgent
from ai_consil.council.roles import ORCHESTRATOR_PROMPT, SYNTHESIS_PROMPT_TEMPLATE
from ai_consil.council.sanitizer import AgentContextSanitizer, BlindVotingSanitizer
from ai_consil.council.vote_vault import VoteVault
from ai_consil.providers import get_provider
from ai_consil.providers.base import ProviderMessage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _now() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RoundState:
    """State for a single round."""

    round_num: int
    analyses: list[AgentAnalysis] = field(default_factory=list)
    questions: list[QuestionAnswer] = field(default_factory=list)
    vote_result: RoundVoteResult | None = None


@dataclass
class SessionState:
    """State for the entire deliberation session."""

    session_id: str
    topic: str
    config: CouncilConfig
    rounds: list[RoundState] = field(default_factory=list)
    final_answer: str = ""
    consensus_reached: bool = False


class CouncilOrchestrator:
    """Orchestrates the council deliberation process.

    The orchestrator is NON-VOTING and only coordinates the agents.
    It manages rounds, Q&A, and voting according to the configuration.
    """

    def __init__(
        self,
        config: CouncilConfig,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: The council configuration.
            temperature: Default temperature for agent completions.
            max_tokens: Default max tokens for completions.
        """
        self.config = config
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Session state
        self.session_id = f"council-{uuid.uuid4().hex[:12]}"
        self.state: SessionState | None = None

        # Blind voting infrastructure
        self.vote_vault = VoteVault()
        self.sanitizer = BlindVotingSanitizer()
        self.context_sanitizer = AgentContextSanitizer()

        # Initialize agents
        self.agents: dict[str, CouncilAgent] = {}
        for agent_config in config.agents:
            self.agents[agent_config.id] = CouncilAgent(
                config=agent_config,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Orchestrator's own provider (for synthesis)
        self._synth_provider = get_provider("mock", "synthesis-v1")

    def _create_event(
        self,
        event_type: CouncilEventType,
        round_num: int | None = None,
        **kwargs: Any,
    ) -> CouncilEvent:
        """Create a council event."""
        return CouncilEvent(
            id=self.session_id,
            event_type=event_type,
            timestamp=_now(),
            round=round_num,
            **kwargs,
        )

    def _should_vote_this_round(self, round_num: int, total_rounds: int) -> bool:
        """Determine if voting should occur this round based on schedule."""
        schedule = self.config.voting_schedule

        if schedule == VotingSchedule.EACH_ROUND:
            return True
        elif schedule == VotingSchedule.START_END:
            return round_num == 1 or round_num == total_rounds
        elif schedule == VotingSchedule.END_ONLY:
            return round_num == total_rounds

        return False

    def _build_discussion_summary(self, state: SessionState) -> str:
        """Build a summary of the discussion so far."""
        parts = [f"Topic: {state.topic}\n"]

        for round_state in state.rounds:
            parts.append(f"\n--- Round {round_state.round_num} ---\n")

            for analysis in round_state.analyses:
                parts.append(f"\n{analysis.agent_id} ({analysis.timestamp}):\n{analysis.content}\n")

            for qa in round_state.questions:
                parts.append(f"\nQ from {qa.from_agent} to {qa.to_agent}: {qa.question}")
                if qa.answer:
                    parts.append(f"\nA: {qa.answer}")

        return "\n".join(parts)

    async def run(self, topic: str) -> tuple[str, CouncilTrace, list[CouncilEvent]]:
        """Run the full council deliberation.

        Args:
            topic: The topic/question to deliberate on.

        Returns:
            Tuple of (final answer, council trace, events list).
        """
        events: list[CouncilEvent] = []
        async for event in self.run_stream(topic):
            events.append(event)

        # Get final answer and trace from state
        if self.state is None:
            raise RuntimeError("Session state not initialized")

        trace = self._build_trace()
        return self.state.final_answer, trace, events

    async def run_stream(self, topic: str) -> AsyncIterator[CouncilEvent]:
        """Run the council deliberation with streaming events.

        Args:
            topic: The topic/question to deliberate on.

        Yields:
            Council events as they occur.
        """
        # Initialize session state
        self.state = SessionState(
            session_id=self.session_id,
            topic=topic,
            config=self.config,
        )

        # Emit session start
        yield self._create_event(
            CouncilEventType.SESSION_START,
            content=topic,
            metadata={"agent_count": len(self.agents)},
        )

        # Run rounds
        for round_num in range(1, self.config.rounds + 1):
            async for event in self._run_round(round_num):
                # Sanitize event before yielding
                sanitized = self.sanitizer.sanitize_event(event)
                if sanitized:
                    yield sanitized

            # Check for early consensus
            if self.state.consensus_reached:
                break

        # Synthesize final answer
        async for event in self._synthesize():
            sanitized = self.sanitizer.sanitize_event(event)
            if sanitized:
                yield sanitized

        # Emit session end
        yield self._create_event(
            CouncilEventType.SESSION_END,
            content=self.state.final_answer,
            metadata={"consensus": self.state.consensus_reached},
        )

    async def _run_round(self, round_num: int) -> AsyncIterator[CouncilEvent]:
        """Run a single deliberation round."""
        if self.state is None:
            raise RuntimeError("Session state not initialized")

        round_state = RoundState(round_num=round_num)
        self.state.rounds.append(round_state)

        # Emit round start
        yield self._create_event(
            CouncilEventType.ROUND_START,
            round_num=round_num,
            metadata={"agents": list(self.agents.keys())},
        )

        # Phase 1: Analysis from all agents
        context = self._build_discussion_summary(self.state) if round_num > 1 else None

        for agent_id, agent in self.agents.items():
            try:
                analysis = await agent.analyze(
                    topic=self.state.topic,
                    context=context,
                    round_num=round_num,
                )

                agent_analysis = AgentAnalysis(
                    agent_id=agent_id,
                    content=analysis,
                    timestamp=_now(),
                )
                round_state.analyses.append(agent_analysis)

                yield self._create_event(
                    CouncilEventType.AGENT_ANALYSIS,
                    round_num=round_num,
                    agent_id=agent_id,
                    content=analysis,
                )

            except Exception as e:
                logger.error(f"Agent {agent_id} analysis failed: {e}")
                yield self._create_event(
                    CouncilEventType.ERROR,
                    round_num=round_num,
                    agent_id=agent_id,
                    error=str(e),
                )

        # Phase 2: Q&A
        async for event in self._run_qa_phase(round_num, round_state):
            yield event

        # Phase 3: Voting (if scheduled)
        if self._should_vote_this_round(round_num, self.config.rounds):
            async for event in self._run_voting_phase(round_num, round_state):
                yield event

        # Emit round end
        yield self._create_event(
            CouncilEventType.ROUND_END,
            round_num=round_num,
        )

    async def _run_qa_phase(
        self,
        round_num: int,
        round_state: RoundState,
    ) -> AsyncIterator[CouncilEvent]:
        """Run the Q&A phase of a round."""
        if self.state is None:
            raise RuntimeError("Session state not initialized")

        max_questions = self.config.max_questions_per_agent
        if max_questions == 0:
            return

        discussion = self._build_discussion_summary(self.state)

        # Each agent can ask questions
        for agent_id, agent in self.agents.items():
            while agent.get_questions_asked(round_num) < max_questions:
                try:
                    parsed_q = await agent.ask_question(
                        discussion=discussion,
                        round_num=round_num,
                        max_questions=max_questions,
                    )

                    if parsed_q is None:
                        break  # Agent has no more questions

                    # Check if target agent exists
                    if parsed_q.to_agent not in self.agents:
                        logger.warning(
                            f"Agent {agent_id} asked question to unknown agent {parsed_q.to_agent}"
                        )
                        continue

                    qa = QuestionAnswer(
                        question_id=f"q-{uuid.uuid4().hex[:8]}",
                        from_agent=agent_id,
                        to_agent=parsed_q.to_agent,
                        question=parsed_q.question,
                        timestamp=_now(),
                    )

                    # Emit question event
                    yield self._create_event(
                        CouncilEventType.QUESTION,
                        round_num=round_num,
                        from_agent=agent_id,
                        to_agent=parsed_q.to_agent,
                        content=parsed_q.question,
                    )

                    # Get answer from target agent
                    target_agent = self.agents[parsed_q.to_agent]
                    answer = await target_agent.answer_question(
                        from_agent=agent_id,
                        question=parsed_q.question,
                        context=discussion,
                    )

                    qa.answer = answer
                    round_state.questions.append(qa)

                    # Emit answer event
                    yield self._create_event(
                        CouncilEventType.ANSWER,
                        round_num=round_num,
                        from_agent=parsed_q.to_agent,
                        to_agent=agent_id,
                        content=answer,
                    )

                    # Update discussion with new Q&A
                    discussion = self._build_discussion_summary(self.state)

                except Exception as e:
                    logger.error(f"Q&A error for agent {agent_id}: {e}")
                    break

    async def _run_voting_phase(
        self,
        round_num: int,
        round_state: RoundState,
    ) -> AsyncIterator[CouncilEvent]:
        """Run the voting phase of a round."""
        if self.state is None:
            raise RuntimeError("Session state not initialized")

        # Emit voting open
        yield self._create_event(
            CouncilEventType.VOTING_OPEN,
            round_num=round_num,
        )

        discussion = self._build_discussion_summary(self.state)

        # Collect votes from all agents (in parallel)
        vote_options = self.config.vote_options or DEFAULT_VOTE_OPTIONS
        vote_tasks = []
        for agent_id, agent in self.agents.items():
            vote_tasks.append(self._collect_vote(agent, round_num, discussion, vote_options))

        await asyncio.gather(*vote_tasks)

        # Close voting
        closed_at = self.vote_vault.close_voting(round_num)
        self.sanitizer.close_voting(round_num)
        self.context_sanitizer.mark_round_closed(round_num)

        # Emit voting closed
        yield self._create_event(
            CouncilEventType.VOTING_CLOSED,
            round_num=round_num,
        )

        # Reveal votes (now safe)
        votes, tally = self.vote_vault.reveal_votes(round_num)

        # Check for consensus (plurality: top option vs threshold)
        total_votes = sum(tally.counts.values())
        if total_votes > 0:
            max_count = max(tally.counts.values())
            top_ratio = max_count / total_votes
            self.state.consensus_reached = top_ratio >= self.config.consensus_threshold

        round_state.vote_result = RoundVoteResult(
            round=round_num,
            voting_closed_at=closed_at,
            votes=votes,
            tally=tally,
            consensus=self.state.consensus_reached,
        )

        # Emit vote reveal
        yield self._create_event(
            CouncilEventType.VOTE_REVEAL,
            round_num=round_num,
            votes=votes,
            tally=tally,
            metadata={"consensus": self.state.consensus_reached},
        )

    async def _collect_vote(
        self,
        agent: CouncilAgent,
        round_num: int,
        discussion: str,
        vote_options: list[str] | None = None,
    ) -> None:
        """Collect a vote from an agent."""
        if self.state is None:
            return

        try:
            parsed_vote = await agent.vote(
                topic=self.state.topic,
                discussion=discussion,
                vote_options=vote_options,
            )

            self.vote_vault.submit_vote(
                agent_id=agent.id,
                round_num=round_num,
                position=parsed_vote.position,
                confidence=parsed_vote.confidence,
                reasoning=parsed_vote.reasoning,
            )

        except Exception as e:
            logger.error(f"Vote collection failed for {agent.id}: {e}")

    async def _synthesize(self) -> AsyncIterator[CouncilEvent]:
        """Synthesize the final answer from the deliberation."""
        if self.state is None:
            raise RuntimeError("Session state not initialized")

        # Build round summaries
        round_summaries = []
        positions = []

        for round_state in self.state.rounds:
            summary = f"Round {round_state.round_num}:\n"
            for analysis in round_state.analyses:
                summary += f"  {analysis.agent_id}: {analysis.content[:200]}...\n"
            round_summaries.append(summary)

            if round_state.vote_result:
                for vote in round_state.vote_result.votes:
                    positions.append(
                        f"{vote.agent_id}: {vote.position} "
                        f"(confidence: {vote.confidence}) - {vote.reasoning}"
                    )

        # Get final vote tally
        final_tally = "No votes recorded"
        if self.state.rounds and self.state.rounds[-1].vote_result:
            tally = self.state.rounds[-1].vote_result.tally
            final_tally = ", ".join(f"{k}: {v}" for k, v in tally.counts.items())

        # Build synthesis prompt
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            topic=self.state.topic,
            round_summaries="\n".join(round_summaries),
            vote_tally=final_tally,
            positions="\n".join(positions) if positions else "No positions recorded",
        )

        # Generate synthesis (using mock provider or configured orchestrator provider)
        messages = [
            ProviderMessage(role="system", content=ORCHESTRATOR_PROMPT),
            ProviderMessage(role="user", content=prompt),
        ]

        response = await self._synth_provider.complete(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        self.state.final_answer = response.content

        # Emit synthesis event
        yield self._create_event(
            CouncilEventType.SYNTHESIS,
            content=response.content,
        )

    def _build_trace(self) -> CouncilTrace:
        """Build the council trace from session state."""
        if self.state is None:
            raise RuntimeError("Session state not initialized")

        round_traces = []
        for round_state in self.state.rounds:
            round_traces.append(
                RoundTrace(
                    round=round_state.round_num,
                    analyses=round_state.analyses,
                    questions=round_state.questions,
                    vote_result=round_state.vote_result,
                )
            )

        final_vote = None
        if self.state.rounds and self.state.rounds[-1].vote_result:
            final_vote = self.state.rounds[-1].vote_result

        return CouncilTrace(
            session_id=self.session_id,
            rounds=round_traces,
            final_vote=final_vote,
            consensus_reached=self.state.consensus_reached,
            artifacts_path=f"./out/{self.session_id}/",
        )
