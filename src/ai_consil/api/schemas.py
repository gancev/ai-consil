"""Pydantic schemas for OpenAI-compatible API with council extensions."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class VotingSchedule(str, Enum):
    """When voting occurs during deliberation."""

    EACH_ROUND = "each_round"
    START_END = "start_end"
    END_ONLY = "end_only"


class VotePosition(str, Enum):
    """Possible vote positions."""

    SUPPORT = "support"
    OPPOSE = "oppose"
    ABSTAIN = "abstain"


class AgentConfig(BaseModel):
    """Configuration for a single council agent."""

    id: str = Field(..., description="Unique agent identifier")
    role: str = Field(..., description="Agent role (e.g., skeptic, advocate, analyst)")
    system_prompt: str | None = Field(None, description="Custom system prompt override")
    provider: str = Field("openai", description="LLM provider name")
    model: str = Field("gpt-4o", description="Model identifier")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent id cannot be empty")
        return v.strip()


class CouncilConfig(BaseModel):
    """Configuration for the council deliberation."""

    agents: list[AgentConfig] = Field(..., min_length=1, description="Council member agents")
    rounds: int = Field(2, ge=1, le=10, description="Number of deliberation rounds")
    max_questions_per_agent: int = Field(2, ge=0, le=10, description="Questions per agent per round")
    voting_schedule: VotingSchedule = Field(
        VotingSchedule.EACH_ROUND, description="When voting occurs"
    )
    consensus_threshold: float = Field(
        0.67, ge=0.0, le=1.0, description="Threshold for consensus"
    )
    trace: bool = Field(False, description="Include detailed trace in response")


# --- OpenAI-compatible message types ---


class MessageRole(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A chat message."""

    role: MessageRole
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with council extensions."""

    model: str = Field("ai-consil", description="Model identifier (informational for council)")
    messages: list[Message] = Field(..., min_length=1, description="Conversation messages")
    stream: bool = Field(False, description="Whether to stream responses")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1)

    # Council extension - can be inline config, path to file, or None
    council: CouncilConfig | str | None = Field(
        None, description="Council config (inline or path to JSON file)"
    )


# --- Response types ---


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    """Message in a completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    """A completion choice."""

    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class DeltaMessage(BaseModel):
    """Delta message for streaming."""

    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """A streaming completion choice."""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


# --- Council trace types ---


class Vote(BaseModel):
    """A single agent's vote."""

    agent_id: str
    position: VotePosition
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class VoteTally(BaseModel):
    """Vote tally for a round."""

    support: int = 0
    oppose: int = 0
    abstain: int = 0


class RoundVoteResult(BaseModel):
    """Vote results for a single round."""

    round: int
    voting_closed_at: str
    votes: list[Vote]
    tally: VoteTally
    consensus: bool


class QuestionAnswer(BaseModel):
    """A Q&A exchange between agents."""

    question_id: str
    from_agent: str
    to_agent: str
    question: str
    answer: str | None = None
    timestamp: str


class AgentAnalysis(BaseModel):
    """An agent's analysis contribution."""

    agent_id: str
    content: str
    timestamp: str


class RoundTrace(BaseModel):
    """Trace data for a single round."""

    round: int
    analyses: list[AgentAnalysis]
    questions: list[QuestionAnswer]
    vote_result: RoundVoteResult | None = None


class CouncilTrace(BaseModel):
    """Full council trace for a session."""

    session_id: str
    rounds: list[RoundTrace]
    final_vote: RoundVoteResult | None = None
    consensus_reached: bool
    artifacts_path: str


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response with council extensions."""

    id: str = Field(default_factory=lambda: f"council-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "ai-consil"
    choices: list[Choice]
    usage: Usage | None = None

    # Council extension
    council_trace: CouncilTrace | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str = "ai-consil"
    choices: list[StreamChoice]


# --- Council event types for streaming ---


class CouncilEventType(str, Enum):
    """Types of council events emitted during streaming."""

    SESSION_START = "session_start"
    ROUND_START = "round_start"
    AGENT_ANALYSIS = "agent_analysis"
    QUESTION = "question"
    ANSWER = "answer"
    VOTING_OPEN = "voting_open"
    VOTING_CLOSED = "voting_closed"
    VOTE_REVEAL = "vote_reveal"
    ROUND_END = "round_end"
    SYNTHESIS = "synthesis"
    SESSION_END = "session_end"
    ERROR = "error"


class CouncilEvent(BaseModel):
    """A council event for streaming."""

    id: str
    object: str = "council.event"
    event_type: CouncilEventType
    timestamp: str
    round: int | None = None
    agent_id: str | None = None
    from_agent: str | None = None
    to_agent: str | None = None
    content: str | None = None
    votes: list[Vote] | None = None
    tally: VoteTally | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
