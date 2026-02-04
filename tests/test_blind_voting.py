"""Blind voting tests - CRITICAL for ensuring vote secrecy."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ai_consil.api.schemas import CouncilEvent, CouncilEventType, VotePosition, VoteTally
from ai_consil.council.sanitizer import BlindVotingSanitizer
from ai_consil.council.vote_vault import (
    DuplicateVoteError,
    VoteVault,
    VotingClosedError,
    VotingNotClosedError,
)


class TestVoteVault:
    """Tests for the VoteVault sealed storage."""

    def test_submit_vote_returns_receipt(self) -> None:
        """Test that submitting a vote returns a receipt without vote data."""
        vault = VoteVault()
        receipt = vault.submit_vote(
            agent_id="agent-1",
            round_num=1,
            position=VotePosition.SUPPORT,
            confidence=0.8,
            reasoning="Test reasoning",
        )

        assert receipt.status == "recorded"
        assert receipt.agent_id == "agent-1"
        assert receipt.round == 1
        assert receipt.vote_id.startswith("vote-")
        # Receipt should NOT contain vote position or reasoning
        assert not hasattr(receipt, "position")
        assert not hasattr(receipt, "confidence")

    def test_vault_reject_early_reveal(self) -> None:
        """Test that reveal() raises if voting is not closed."""
        vault = VoteVault()
        vault.submit_vote(
            agent_id="agent-1",
            round_num=1,
            position=VotePosition.SUPPORT,
            confidence=0.8,
            reasoning="Test",
        )

        with pytest.raises(VotingNotClosedError):
            vault.reveal_votes(1)

    def test_reveal_after_close(self) -> None:
        """Test that reveal works after voting is closed."""
        vault = VoteVault()
        vault.submit_vote(
            agent_id="agent-1",
            round_num=1,
            position=VotePosition.SUPPORT,
            confidence=0.8,
            reasoning="Support reasoning",
        )
        vault.submit_vote(
            agent_id="agent-2",
            round_num=1,
            position=VotePosition.OPPOSE,
            confidence=0.6,
            reasoning="Oppose reasoning",
        )

        vault.close_voting(1)
        votes, tally = vault.reveal_votes(1)

        assert len(votes) == 2
        assert tally.support == 1
        assert tally.oppose == 1
        assert tally.abstain == 0

    def test_no_voting_after_close(self) -> None:
        """Test that voting after close raises error."""
        vault = VoteVault()
        vault.close_voting(1)

        with pytest.raises(VotingClosedError):
            vault.submit_vote(
                agent_id="agent-1",
                round_num=1,
                position=VotePosition.SUPPORT,
                confidence=0.8,
                reasoning="Test",
            )

    def test_no_duplicate_votes(self) -> None:
        """Test that an agent cannot vote twice in same round."""
        vault = VoteVault()
        vault.submit_vote(
            agent_id="agent-1",
            round_num=1,
            position=VotePosition.SUPPORT,
            confidence=0.8,
            reasoning="First vote",
        )

        with pytest.raises(DuplicateVoteError):
            vault.submit_vote(
                agent_id="agent-1",
                round_num=1,
                position=VotePosition.OPPOSE,
                confidence=0.7,
                reasoning="Second vote",
            )

    def test_has_voted_without_revealing(self) -> None:
        """Test that has_voted works without revealing vote details."""
        vault = VoteVault()
        assert not vault.has_voted("agent-1", 1)

        vault.submit_vote(
            agent_id="agent-1",
            round_num=1,
            position=VotePosition.SUPPORT,
            confidence=0.8,
            reasoning="Test",
        )

        assert vault.has_voted("agent-1", 1)
        assert not vault.has_voted("agent-2", 1)

    def test_vote_count_without_revealing(self) -> None:
        """Test that vote count works without revealing vote details."""
        vault = VoteVault()
        assert vault.get_vote_count(1) == 0

        vault.submit_vote(
            agent_id="agent-1",
            round_num=1,
            position=VotePosition.SUPPORT,
            confidence=0.8,
            reasoning="Test",
        )
        assert vault.get_vote_count(1) == 1

        vault.submit_vote(
            agent_id="agent-2",
            round_num=1,
            position=VotePosition.OPPOSE,
            confidence=0.7,
            reasoning="Test",
        )
        assert vault.get_vote_count(1) == 2


class TestBlindVotingSanitizer:
    """Tests for the streaming event sanitizer."""

    def test_sanitizer_blocks_vote_reveal_before_close(self) -> None:
        """Test that vote reveal events are blocked before voting closed."""
        sanitizer = BlindVotingSanitizer()

        event = CouncilEvent(
            id="test",
            event_type=CouncilEventType.VOTE_REVEAL,
            timestamp="2025-01-01T00:00:00Z",
            round=1,
            votes=[],
            tally=VoteTally(support=1, oppose=0, abstain=0),
        )

        # Before closing, event should be blocked
        result = sanitizer.sanitize_event(event)
        assert result is None

        # After closing, event should pass
        sanitizer.close_voting(1)
        result = sanitizer.sanitize_event(event)
        assert result is not None

    def test_sanitizer_strips_vote_data(self) -> None:
        """Test that sanitizer strips vote data from events before close."""
        sanitizer = BlindVotingSanitizer()

        event = CouncilEvent(
            id="test",
            event_type=CouncilEventType.AGENT_ANALYSIS,
            timestamp="2025-01-01T00:00:00Z",
            round=1,
            agent_id="agent-1",
            content="Analysis content",
            votes=[],  # Should be stripped
            tally=VoteTally(support=1, oppose=0, abstain=0),  # Should be stripped
        )

        result = sanitizer.sanitize_event(event)
        assert result is not None
        assert result.votes is None
        assert result.tally is None
        assert result.content == "Analysis content"

    def test_sanitizer_detects_vote_info_in_content(self) -> None:
        """Test that sanitizer detects vote info patterns in content."""
        sanitizer = BlindVotingSanitizer()

        # Test various patterns that might leak vote info
        leak_patterns = [
            "Current votes: 3 support, 1 oppose",
            "The tally shows 5 in favor",
            "Agent-1 voted for the proposal",
            "Vote count is now 2-1",
        ]

        for content in leak_patterns:
            event = CouncilEvent(
                id="test",
                event_type=CouncilEventType.AGENT_ANALYSIS,
                timestamp="2025-01-01T00:00:00Z",
                round=1,
                content=content,
            )

            result = sanitizer.sanitize_event(event)
            if result is not None and result.content:
                # Content should be sanitized
                assert "[VOTE INFO REDACTED]" in result.content or content != result.content

    def test_sanitizer_allows_after_close(self) -> None:
        """Test that sanitizer allows all events after voting closed."""
        sanitizer = BlindVotingSanitizer()
        sanitizer.close_voting(1)

        event = CouncilEvent(
            id="test",
            event_type=CouncilEventType.VOTE_REVEAL,
            timestamp="2025-01-01T00:00:00Z",
            round=1,
            votes=[],
            tally=VoteTally(support=2, oppose=1, abstain=0),
        )

        result = sanitizer.sanitize_event(event)
        assert result is not None
        assert result.tally is not None
        assert result.tally.support == 2

    def test_sanitizer_records_violations(self) -> None:
        """Test that sanitizer records blocked events."""
        sanitizer = BlindVotingSanitizer()

        event = CouncilEvent(
            id="test",
            event_type=CouncilEventType.VOTE_REVEAL,
            timestamp="2025-01-01T00:00:00Z",
            round=1,
        )

        sanitizer.sanitize_event(event)
        violations = sanitizer.get_violations()

        assert len(violations) == 1
        assert violations[0].event_type == "vote_reveal"


class TestBlindVotingInStreaming:
    """Tests for blind voting in streaming responses."""

    def test_no_vote_leak_during_voting(
        self, test_client: TestClient, sample_stream_request: dict
    ) -> None:
        """CRITICAL: Verify no vote/tally data appears before voting_closed."""
        sample_stream_request["council"]["trace"] = True

        with test_client.stream(
            "POST", "/v1/chat/completions", json=sample_stream_request
        ) as response:
            content = response.read().decode("utf-8")

            # Parse all events
            events = []
            for line in content.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        if data.get("object") == "council.event":
                            events.append(data)
                    except json.JSONDecodeError:
                        pass

            # Track voting state
            voting_closed_rounds = set()

            for event in events:
                event_type = event.get("event_type")
                round_num = event.get("round")

                if event_type == "voting_closed":
                    voting_closed_rounds.add(round_num)

                elif event_type == "vote_reveal":
                    # Vote reveal must only appear AFTER voting_closed
                    assert round_num in voting_closed_rounds, (
                        f"BLIND VOTING VIOLATION: vote_reveal for round {round_num} "
                        f"appeared before voting_closed"
                    )

                elif round_num is not None and round_num not in voting_closed_rounds:
                    # Before voting closed, no vote data should appear
                    assert event.get("votes") is None, (
                        f"BLIND VOTING VIOLATION: votes in {event_type} "
                        f"before voting closed for round {round_num}"
                    )
                    assert event.get("tally") is None, (
                        f"BLIND VOTING VIOLATION: tally in {event_type} "
                        f"before voting closed for round {round_num}"
                    )

    def test_vote_reveal_after_close_only(
        self, test_client: TestClient, sample_stream_request: dict
    ) -> None:
        """Test that vote_reveal event only appears after voting_closed."""
        sample_stream_request["council"]["trace"] = True

        with test_client.stream(
            "POST", "/v1/chat/completions", json=sample_stream_request
        ) as response:
            content = response.read().decode("utf-8")

            events = []
            for line in content.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        if data.get("object") == "council.event":
                            events.append(data)
                    except json.JSONDecodeError:
                        pass

            # Find indices of voting_closed and vote_reveal events per round
            for round_num in range(1, 10):  # Check up to 10 rounds
                voting_closed_idx = None
                vote_reveal_idx = None

                for idx, event in enumerate(events):
                    if event.get("round") == round_num:
                        if event.get("event_type") == "voting_closed":
                            voting_closed_idx = idx
                        elif event.get("event_type") == "vote_reveal":
                            vote_reveal_idx = idx

                # If both exist, vote_reveal must come after voting_closed
                if voting_closed_idx is not None and vote_reveal_idx is not None:
                    assert vote_reveal_idx > voting_closed_idx, (
                        f"vote_reveal came before voting_closed for round {round_num}"
                    )
