"""Tests for the transcript formatter."""

from __future__ import annotations

import pytest

from ai_consil.api.schemas import (
    CouncilEvent,
    CouncilEventType,
    Vote,
    VoteTally,
)
from ai_consil.storage.transcript_formatter import format_transcript_report


def _event(
    event_type: CouncilEventType,
    **kwargs,
) -> CouncilEvent:
    """Helper to create a CouncilEvent with defaults."""
    return CouncilEvent(
        id="council-test123",
        event_type=event_type,
        timestamp="2025-01-15T12:00:00+00:00",
        **kwargs,
    )


class TestTranscriptFormatterStructure:
    """Tests for basic markdown structure."""

    def test_report_header(self) -> None:
        """Report should contain title, session ID, and topic."""
        events = [
            _event(CouncilEventType.SESSION_START, content="Test topic"),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "Test topic", "council-abc123")

        assert "# Council Deliberation Report" in report
        assert "council-abc123" in report
        assert "Test topic" in report

    def test_empty_events(self) -> None:
        """Report should still produce a header with empty events."""
        report = format_transcript_report([], "Empty test", "council-empty")

        assert "# Council Deliberation Report" in report
        assert "Empty test" in report

    def test_round_sections(self) -> None:
        """Report should have round sections."""
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="agent-1",
                content="Analysis text",
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert "## Round 1" in report
        assert "### Agent Analyses" in report


class TestTranscriptFormatterAnalyses:
    """Tests for agent analysis rendering."""

    def test_full_analysis_not_truncated(self) -> None:
        """Full analysis text should appear, not truncated."""
        long_text = "A" * 1000
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="analyst-1",
                content=long_text,
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert long_text in report

    def test_multiple_agents_analyses(self) -> None:
        """Multiple agent analyses should all appear."""
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="cathie-wood",
                content="Bullish on innovation",
            ),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="michael-burry",
                content="Bubble territory",
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert "#### cathie-wood" in report
        assert "Bullish on innovation" in report
        assert "#### michael-burry" in report
        assert "Bubble territory" in report


class TestTranscriptFormatterQA:
    """Tests for Q&A rendering."""

    def test_question_and_answer(self) -> None:
        """Q&A exchanges should be formatted with arrow notation."""
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="agent-1",
                content="Analysis",
            ),
            _event(
                CouncilEventType.QUESTION,
                round=1,
                from_agent="cathie-wood",
                to_agent="michael-burry",
                content="What about growth?",
            ),
            _event(
                CouncilEventType.ANSWER,
                round=1,
                from_agent="michael-burry",
                to_agent="cathie-wood",
                content="The data shows otherwise.",
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert "### Questions & Answers" in report
        assert "cathie-wood" in report
        assert "michael-burry" in report
        assert "What about growth?" in report
        assert "The data shows otherwise." in report

    def test_question_without_answer(self) -> None:
        """A question without an answer should still appear."""
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.QUESTION,
                round=1,
                from_agent="agent-1",
                to_agent="agent-2",
                content="Unanswered question?",
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert "Unanswered question?" in report


class TestTranscriptFormatterVotes:
    """Tests for vote table rendering."""

    def test_vote_table(self) -> None:
        """Votes should be rendered in a markdown table."""
        votes = [
            Vote(agent_id="cathie-wood", position="bullish", confidence=0.92, reasoning="Innovation drives growth"),
            Vote(agent_id="michael-burry", position="bearish", confidence=0.85, reasoning="Overvalued by all metrics"),
        ]
        tally = VoteTally(counts={"bullish": 1, "bearish": 1})

        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="cathie-wood",
                content="Analysis",
            ),
            _event(CouncilEventType.VOTING_OPEN, round=1),
            _event(CouncilEventType.VOTING_CLOSED, round=1),
            _event(
                CouncilEventType.VOTE_REVEAL,
                round=1,
                votes=votes,
                tally=tally,
                metadata={"consensus": False},
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert "### Voting" in report
        assert "| Agent | Vote | Confidence | Reasoning |" in report
        assert "cathie-wood" in report
        assert "bullish" in report
        assert "0.92" in report
        assert "Innovation drives growth" in report
        assert "bearish" in report
        assert "**Tally:**" in report
        assert "**Consensus:** No" in report

    def test_vote_reasoning_with_pipe_escaped(self) -> None:
        """Pipe characters in reasoning should be escaped for markdown tables."""
        votes = [
            Vote(
                agent_id="agent-1",
                position="support",
                confidence=0.7,
                reasoning="Option A | Option B are both valid",
            ),
        ]
        tally = VoteTally(counts={"support": 1})

        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.VOTE_REVEAL,
                round=1,
                votes=votes,
                tally=tally,
                metadata={"consensus": True},
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        # The pipe inside reasoning should be escaped
        assert "Option A \\| Option B" in report


class TestTranscriptFormatterSynthesis:
    """Tests for synthesis section."""

    def test_synthesis_section(self) -> None:
        """Final synthesis should appear under its own heading."""
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="agent-1",
                content="Analysis",
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(
                CouncilEventType.SYNTHESIS,
                content="The council concludes that the market is overvalued.",
            ),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert "## Final Synthesis" in report
        assert "The council concludes that the market is overvalued." in report


class TestTranscriptFormatterMultiRound:
    """Tests for multi-round deliberations."""

    def test_two_rounds(self) -> None:
        """Multiple rounds should each get their own section."""
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="agent-1",
                content="Round 1 analysis",
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.ROUND_START, round=2),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=2,
                agent_id="agent-1",
                content="Round 2 analysis",
            ),
            _event(CouncilEventType.ROUND_END, round=2),
            _event(CouncilEventType.SYNTHESIS, content="Final answer"),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert "## Round 1" in report
        assert "Round 1 analysis" in report
        assert "## Round 2" in report
        assert "Round 2 analysis" in report
        assert "## Final Synthesis" in report


class TestTranscriptFormatterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_analysis_with_empty_content(self) -> None:
        """Agent with empty content should show (empty)."""
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.AGENT_ANALYSIS,
                round=1,
                agent_id="agent-1",
                content=None,
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        report = format_transcript_report(events, "topic", "council-test")

        assert "(empty)" in report

    def test_error_events_ignored_gracefully(self) -> None:
        """Error events should not crash the formatter."""
        events = [
            _event(CouncilEventType.SESSION_START, content="topic"),
            _event(CouncilEventType.ROUND_START, round=1),
            _event(
                CouncilEventType.ERROR,
                round=1,
                agent_id="agent-1",
                error="Provider timeout",
            ),
            _event(CouncilEventType.ROUND_END, round=1),
            _event(CouncilEventType.SESSION_END, content="done"),
        ]
        # Should not raise
        report = format_transcript_report(events, "topic", "council-test")
        assert "## Round 1" in report
