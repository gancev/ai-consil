"""Format council events into a human-readable markdown transcript report."""

from __future__ import annotations

from datetime import datetime, timezone

from ai_consil.api.schemas import CouncilEvent, CouncilEventType


def format_transcript_report(
    events: list[CouncilEvent],
    topic: str,
    session_id: str,
) -> str:
    """Convert a list of council events into a markdown report.

    Args:
        events: List of CouncilEvent objects from a deliberation session.
        topic: The deliberation topic/question.
        session_id: The session identifier.

    Returns:
        A formatted markdown string with the full deliberation transcript.
    """
    lines: list[str] = []

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# Council Deliberation Report")
    lines.append("")
    lines.append(f"**Session:** {session_id} | **Date:** {date_str}")
    lines.append("")
    lines.append("## Topic")
    lines.append("")
    lines.append(topic)
    lines.append("")

    # Group events by round
    current_round: int | None = None
    round_analyses: list[CouncilEvent] = []
    round_questions: list[tuple[CouncilEvent, CouncilEvent | None]] = []
    round_votes: CouncilEvent | None = None
    pending_question: CouncilEvent | None = None

    for event in events:
        etype = event.event_type

        if etype == CouncilEventType.ROUND_START:
            # Flush previous round
            if current_round is not None:
                _write_round(
                    lines, current_round, round_analyses, round_questions, round_votes
                )
            current_round = event.round
            round_analyses = []
            round_questions = []
            round_votes = None
            pending_question = None

        elif etype == CouncilEventType.AGENT_ANALYSIS:
            round_analyses.append(event)

        elif etype == CouncilEventType.QUESTION:
            # Save pending question; answer may follow
            if pending_question is not None:
                round_questions.append((pending_question, None))
            pending_question = event

        elif etype == CouncilEventType.ANSWER:
            if pending_question is not None:
                round_questions.append((pending_question, event))
                pending_question = None
            else:
                # Orphaned answer — still record it
                round_questions.append((event, None))

        elif etype == CouncilEventType.VOTE_REVEAL:
            round_votes = event

        elif etype == CouncilEventType.SYNTHESIS:
            # Flush last round before synthesis
            if current_round is not None:
                _write_round(
                    lines, current_round, round_analyses, round_questions, round_votes
                )
                current_round = None

            # Flush any trailing pending question
            if pending_question is not None:
                round_questions.append((pending_question, None))
                pending_question = None

            lines.append("## Final Synthesis")
            lines.append("")
            lines.append(event.content or "(empty)")
            lines.append("")

        elif etype == CouncilEventType.SESSION_END:
            # Flush if synthesis didn't already flush
            if current_round is not None:
                if pending_question is not None:
                    round_questions.append((pending_question, None))
                    pending_question = None
                _write_round(
                    lines, current_round, round_analyses, round_questions, round_votes
                )
                current_round = None

    # Handle edge case: no session_end or synthesis event
    if current_round is not None:
        if pending_question is not None:
            round_questions.append((pending_question, None))
        _write_round(
            lines, current_round, round_analyses, round_questions, round_votes
        )

    return "\n".join(lines)


def _write_round(
    lines: list[str],
    round_num: int,
    analyses: list[CouncilEvent],
    questions: list[tuple[CouncilEvent, CouncilEvent | None]],
    vote_reveal: CouncilEvent | None,
) -> None:
    """Append a single round's markdown section to lines."""
    lines.append(f"## Round {round_num}")
    lines.append("")

    # --- Analyses ---
    if analyses:
        lines.append("### Agent Analyses")
        lines.append("")
        for ev in analyses:
            agent_id = ev.agent_id or "unknown"
            lines.append(f"#### {agent_id}")
            lines.append("")
            lines.append(ev.content or "(empty)")
            lines.append("")

    # --- Q&A ---
    if questions:
        lines.append("### Questions & Answers")
        lines.append("")
        for q_event, a_event in questions:
            from_agent = q_event.from_agent or q_event.agent_id or "unknown"
            to_agent = q_event.to_agent or "unknown"
            lines.append(f"**{from_agent} \u2192 {to_agent}:** {q_event.content or ''}")
            lines.append("")
            if a_event and a_event.content:
                answerer = a_event.from_agent or a_event.agent_id or to_agent
                lines.append(f"> **{answerer}:** {a_event.content}")
                lines.append("")
        lines.append("")

    # --- Votes ---
    if vote_reveal and vote_reveal.votes:
        lines.append(f"### Voting \u2014 Round {round_num}")
        lines.append("")
        lines.append("| Agent | Vote | Confidence | Reasoning |")
        lines.append("|-------|------|------------|-----------|")
        for vote in vote_reveal.votes:
            reasoning = vote.reasoning.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {vote.agent_id} | {vote.position} | {vote.confidence:.2f} | {reasoning} |"
            )
        lines.append("")

        # Tally
        if vote_reveal.tally and vote_reveal.tally.counts:
            tally_parts = [
                f"{option}: {count}"
                for option, count in vote_reveal.tally.counts.items()
            ]
            tally_str = ", ".join(tally_parts)
            consensus = vote_reveal.metadata.get("consensus", False) if vote_reveal.metadata else False
            consensus_str = "Yes" if consensus else "No"
            lines.append(f"**Tally:** {tally_str} | **Consensus:** {consensus_str}")
            lines.append("")
