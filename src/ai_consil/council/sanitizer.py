"""Blind voting sanitizer for streaming events."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ai_consil.api.schemas import CouncilEvent, CouncilEventType

logger = logging.getLogger(__name__)


@dataclass
class SanitizerViolation:
    """Record of a blocked event due to blind voting violation."""

    event_type: str
    round: int | None
    reason: str
    timestamp: str


class BlindVotingSanitizer:
    """Sanitizer that enforces blind voting during streaming.

    This sanitizer MUST be used for all events before they are emitted
    to clients. It ensures that vote information is never leaked before
    voting is officially closed for a round.
    """

    # Patterns that might indicate vote information in content
    VOTE_PATTERNS = [
        re.compile(r"\bvotes?\s*:\s*\d+", re.IGNORECASE),
        re.compile(r"\btally\b.*\d+", re.IGNORECASE),
        re.compile(r"\b\d+\s*votes?\b", re.IGNORECASE),
        re.compile(r"(current|vote)\s+(vote|tally|count)", re.IGNORECASE),
        re.compile(r"(voted|voting)\s+(for|against)", re.IGNORECASE),
    ]

    def __init__(self) -> None:
        self._voting_closed: dict[int, bool] = {}
        self._violations: list[SanitizerViolation] = []

    def close_voting(self, round_num: int) -> None:
        """Mark voting as closed for a round."""
        self._voting_closed[round_num] = True
        logger.debug(f"Voting closed for round {round_num}")

    def is_voting_closed(self, round_num: int) -> bool:
        """Check if voting is closed for a round."""
        return self._voting_closed.get(round_num, False)

    def _is_vote_event(self, event: CouncilEvent) -> bool:
        """Check if an event is a vote-related event."""
        return event.event_type in (
            CouncilEventType.VOTE_REVEAL,
        )

    def _contains_vote_info(self, content: str | None) -> bool:
        """Check if content contains vote information."""
        if not content:
            return False

        for pattern in self.VOTE_PATTERNS:
            if pattern.search(content):
                return True

        return False

    def _record_violation(
        self,
        event: CouncilEvent,
        reason: str,
    ) -> None:
        """Record a sanitizer violation for debugging."""
        violation = SanitizerViolation(
            event_type=event.event_type.value,
            round=event.round,
            reason=reason,
            timestamp=event.timestamp,
        )
        self._violations.append(violation)
        logger.warning(
            f"BLIND VOTING VIOLATION BLOCKED: {reason} "
            f"(event_type={event.event_type}, round={event.round})"
        )

    def sanitize_event(self, event: CouncilEvent) -> CouncilEvent | None:
        """Sanitize an event before streaming.

        Args:
            event: The event to sanitize.

        Returns:
            The event if safe to emit, or None if it should be blocked.
        """
        round_num = event.round

        # If round is specified and voting isn't closed, check for violations
        if round_num is not None and not self.is_voting_closed(round_num):
            # Block vote reveal events
            if self._is_vote_event(event):
                self._record_violation(
                    event,
                    f"Vote event emitted before voting closed for round {round_num}",
                )
                return None

            # Block events with vote data
            if event.votes is not None or event.tally is not None:
                self._record_violation(
                    event,
                    f"Event contains vote data before voting closed for round {round_num}",
                )
                # Return event without vote data
                return CouncilEvent(
                    id=event.id,
                    object=event.object,
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    round=event.round,
                    agent_id=event.agent_id,
                    from_agent=event.from_agent,
                    to_agent=event.to_agent,
                    content=event.content,
                    votes=None,  # Removed
                    tally=None,  # Removed
                    error=event.error,
                    metadata=event.metadata,
                )

            # Check content for vote information
            if self._contains_vote_info(event.content):
                self._record_violation(
                    event,
                    f"Event content may contain vote info before voting closed",
                )
                # Sanitize content by removing vote references
                sanitized_content = self._sanitize_content(event.content)
                return CouncilEvent(
                    id=event.id,
                    object=event.object,
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    round=event.round,
                    agent_id=event.agent_id,
                    from_agent=event.from_agent,
                    to_agent=event.to_agent,
                    content=sanitized_content,
                    votes=event.votes,
                    tally=event.tally,
                    error=event.error,
                    metadata=event.metadata,
                )

        return event

    def _sanitize_content(self, content: str | None) -> str | None:
        """Remove vote information from content."""
        if not content:
            return content

        sanitized = content
        for pattern in self.VOTE_PATTERNS:
            sanitized = pattern.sub("[VOTE INFO REDACTED]", sanitized)

        return sanitized

    def get_violations(self) -> list[SanitizerViolation]:
        """Get all recorded violations."""
        return self._violations.copy()

    def clear_violations(self) -> None:
        """Clear recorded violations."""
        self._violations.clear()


@dataclass
class AgentContextSanitizer:
    """Sanitizer for agent context to prevent vote information leakage.

    When building prompts for agents, this sanitizer ensures that
    no vote information from the current round is included.
    """

    closed_rounds: set[int] = field(default_factory=set)

    def mark_round_closed(self, round_num: int) -> None:
        """Mark a round as closed (votes can now be shared)."""
        self.closed_rounds.add(round_num)

    def sanitize_for_agent(
        self,
        content: str,
        current_round: int,
    ) -> str:
        """Sanitize content before including in agent context.

        Args:
            content: The content to sanitize.
            current_round: The current round number.

        Returns:
            Sanitized content safe to include in agent prompts.
        """
        # For the current round, always remove vote info
        if current_round not in self.closed_rounds:
            for pattern in BlindVotingSanitizer.VOTE_PATTERNS:
                content = pattern.sub("[VOTE INFO PENDING]", content)

        return content
