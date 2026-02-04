"""Sealed vote storage with blind voting guarantee."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_consil.api.schemas import Vote, VotePosition, VoteTally


@dataclass
class VoteReceipt:
    """Receipt returned when a vote is recorded."""

    vote_id: str
    agent_id: str
    round: int
    recorded_at: str
    status: str = "recorded"


@dataclass
class SealedVote:
    """Internal representation of a sealed vote."""

    vote_id: str
    agent_id: str
    round: int
    position: VotePosition
    confidence: float
    reasoning: str
    recorded_at: str


@dataclass
class RoundVotes:
    """Votes for a single round."""

    round: int
    votes: dict[str, SealedVote] = field(default_factory=dict)
    voting_open: bool = True
    closed_at: str | None = None


class VoteVaultError(Exception):
    """Base exception for VoteVault errors."""

    pass


class VotingNotClosedError(VoteVaultError):
    """Raised when attempting to reveal votes before voting is closed."""

    pass


class VotingClosedError(VoteVaultError):
    """Raised when attempting to vote after voting is closed."""

    pass


class DuplicateVoteError(VoteVaultError):
    """Raised when an agent tries to vote twice in the same round."""

    pass


class VoteVault:
    """Sealed storage for votes with blind voting guarantee.

    This class ensures that votes cannot be accessed until voting is
    explicitly closed for a round. All operations are thread-safe.

    The VoteVault is the ONLY place votes are stored, and it enforces
    strict access control to prevent any premature vote disclosure.
    """

    def __init__(self) -> None:
        self._rounds: dict[int, RoundVotes] = {}
        self._lock = threading.Lock()

    def _get_or_create_round(self, round_num: int) -> RoundVotes:
        """Get or create a round's vote storage."""
        if round_num not in self._rounds:
            self._rounds[round_num] = RoundVotes(round=round_num)
        return self._rounds[round_num]

    def submit_vote(
        self,
        agent_id: str,
        round_num: int,
        position: VotePosition,
        confidence: float,
        reasoning: str,
    ) -> VoteReceipt:
        """Submit a vote for a round.

        Args:
            agent_id: The voting agent's ID.
            round_num: The round number.
            position: The vote position (support/oppose/abstain).
            confidence: Confidence level (0.0-1.0).
            reasoning: The agent's reasoning for their vote.

        Returns:
            A receipt confirming the vote was recorded.

        Raises:
            VotingClosedError: If voting is already closed for this round.
            DuplicateVoteError: If the agent has already voted this round.
        """
        with self._lock:
            round_votes = self._get_or_create_round(round_num)

            if not round_votes.voting_open:
                raise VotingClosedError(
                    f"Voting is closed for round {round_num}"
                )

            if agent_id in round_votes.votes:
                raise DuplicateVoteError(
                    f"Agent '{agent_id}' has already voted in round {round_num}"
                )

            vote_id = f"vote-{uuid.uuid4().hex[:8]}"
            recorded_at = datetime.now(timezone.utc).isoformat()

            sealed_vote = SealedVote(
                vote_id=vote_id,
                agent_id=agent_id,
                round=round_num,
                position=position,
                confidence=confidence,
                reasoning=reasoning,
                recorded_at=recorded_at,
            )

            round_votes.votes[agent_id] = sealed_vote

            return VoteReceipt(
                vote_id=vote_id,
                agent_id=agent_id,
                round=round_num,
                recorded_at=recorded_at,
                status="recorded",
            )

    def close_voting(self, round_num: int) -> str:
        """Close voting for a round.

        Args:
            round_num: The round number to close.

        Returns:
            The timestamp when voting was closed.
        """
        with self._lock:
            round_votes = self._get_or_create_round(round_num)
            round_votes.voting_open = False
            round_votes.closed_at = datetime.now(timezone.utc).isoformat()
            return round_votes.closed_at

    def is_voting_open(self, round_num: int) -> bool:
        """Check if voting is still open for a round."""
        with self._lock:
            if round_num not in self._rounds:
                return True  # Not started yet
            return self._rounds[round_num].voting_open

    def reveal_votes(self, round_num: int) -> tuple[list[Vote], VoteTally]:
        """Reveal votes for a round after voting is closed.

        Args:
            round_num: The round number.

        Returns:
            Tuple of (list of votes, vote tally).

        Raises:
            VotingNotClosedError: If voting is still open.
        """
        with self._lock:
            if round_num not in self._rounds:
                raise VotingNotClosedError(
                    f"No votes recorded for round {round_num}"
                )

            round_votes = self._rounds[round_num]

            if round_votes.voting_open:
                raise VotingNotClosedError(
                    f"Voting is still open for round {round_num}. "
                    "Close voting before revealing votes."
                )

            # Convert sealed votes to public Vote objects
            votes = [
                Vote(
                    agent_id=sv.agent_id,
                    position=sv.position,
                    confidence=sv.confidence,
                    reasoning=sv.reasoning,
                )
                for sv in round_votes.votes.values()
            ]

            # Calculate tally
            tally = VoteTally()
            for vote in votes:
                if vote.position == VotePosition.SUPPORT:
                    tally.support += 1
                elif vote.position == VotePosition.OPPOSE:
                    tally.oppose += 1
                else:
                    tally.abstain += 1

            return votes, tally

    def get_vote_count(self, round_num: int) -> int:
        """Get the number of votes recorded for a round (without revealing votes)."""
        with self._lock:
            if round_num not in self._rounds:
                return 0
            return len(self._rounds[round_num].votes)

    def has_voted(self, agent_id: str, round_num: int) -> bool:
        """Check if an agent has voted in a round (without revealing the vote)."""
        with self._lock:
            if round_num not in self._rounds:
                return False
            return agent_id in self._rounds[round_num].votes

    def get_voting_closed_at(self, round_num: int) -> str | None:
        """Get when voting was closed for a round."""
        with self._lock:
            if round_num not in self._rounds:
                return None
            return self._rounds[round_num].closed_at

    def to_dict(self) -> dict[str, Any]:
        """Export vault state for persistence (only closed rounds).

        Returns:
            Dictionary representation of closed rounds' votes.

        Note:
            This will NOT include votes from rounds where voting is still open.
        """
        with self._lock:
            result: dict[str, Any] = {"rounds": []}

            for round_num in sorted(self._rounds.keys()):
                round_votes = self._rounds[round_num]

                if not round_votes.voting_open:
                    votes, tally = self.reveal_votes(round_num)
                    result["rounds"].append({
                        "round": round_num,
                        "voting_closed_at": round_votes.closed_at,
                        "votes": [v.model_dump() for v in votes],
                        "tally": tally.model_dump(),
                    })

            return result
