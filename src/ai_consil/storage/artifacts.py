"""Artifact storage for transcripts, vote ledgers, and session outputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_consil.api.schemas import CouncilConfig, CouncilEvent, CouncilTrace


@dataclass
class ArtifactPaths:
    """Paths to session artifacts."""

    base_dir: Path
    transcript: Path
    vote_ledger: Path
    final_answer: Path
    metadata: Path


class ArtifactStore:
    """Manages storage of session artifacts.

    Creates and manages the ./out/{session_id}/ directory structure
    with transcript, vote ledger, final answer, and metadata files.
    """

    def __init__(self, base_path: str = "./out") -> None:
        """Initialize the artifact store.

        Args:
            base_path: Base directory for all session artifacts.
        """
        self.base_path = Path(base_path)

    def _ensure_session_dir(self, session_id: str) -> ArtifactPaths:
        """Ensure session directory exists and return paths."""
        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        return ArtifactPaths(
            base_dir=session_dir,
            transcript=session_dir / "transcript.ndjson",
            vote_ledger=session_dir / "vote_ledger.json",
            final_answer=session_dir / "final_answer.md",
            metadata=session_dir / "metadata.json",
        )

    def append_event(self, session_id: str, event: CouncilEvent) -> None:
        """Append an event to the session transcript.

        Events are stored as newline-delimited JSON (NDJSON) for
        efficient append-only writing and streaming reads.

        Args:
            session_id: The session ID.
            event: The event to append.
        """
        paths = self._ensure_session_dir(session_id)

        event_dict = event.model_dump()
        # Convert enum to string for JSON serialization
        event_dict["event_type"] = event.event_type.value

        with open(paths.transcript, "a") as f:
            f.write(json.dumps(event_dict) + "\n")

    def write_vote_ledger(
        self,
        session_id: str,
        trace: CouncilTrace,
    ) -> None:
        """Write the vote ledger for a session.

        Args:
            session_id: The session ID.
            trace: The council trace containing vote information.
        """
        paths = self._ensure_session_dir(session_id)

        ledger: dict[str, Any] = {
            "session_id": session_id,
            "rounds": [],
            "final_tally": None,
            "consensus_reached": trace.consensus_reached,
        }

        for round_trace in trace.rounds:
            if round_trace.vote_result:
                ledger["rounds"].append({
                    "round": round_trace.round,
                    "voting_closed_at": round_trace.vote_result.voting_closed_at,
                    "votes": [v.model_dump() for v in round_trace.vote_result.votes],
                    "tally": round_trace.vote_result.tally.model_dump(),
                    "consensus": round_trace.vote_result.consensus,
                })

        if trace.final_vote:
            ledger["final_tally"] = trace.final_vote.tally.model_dump()

        with open(paths.vote_ledger, "w") as f:
            json.dump(ledger, f, indent=2)

    def write_final_answer(
        self,
        session_id: str,
        answer: str,
        topic: str,
    ) -> None:
        """Write the final answer as a markdown file.

        Args:
            session_id: The session ID.
            answer: The final synthesized answer.
            topic: The original topic/question.
        """
        paths = self._ensure_session_dir(session_id)

        content = f"""# Council Deliberation Result

## Topic
{topic}

## Final Answer
{answer}

---
*Session ID: {session_id}*
*Generated: {datetime.now(timezone.utc).isoformat()}*
"""

        with open(paths.final_answer, "w") as f:
            f.write(content)

    def write_metadata(
        self,
        session_id: str,
        config: CouncilConfig,
        topic: str,
        start_time: str,
        end_time: str | None = None,
    ) -> None:
        """Write session metadata.

        Args:
            session_id: The session ID.
            config: The council configuration used.
            topic: The deliberation topic.
            start_time: Session start timestamp.
            end_time: Session end timestamp (if complete).
        """
        paths = self._ensure_session_dir(session_id)

        metadata = {
            "session_id": session_id,
            "topic": topic,
            "config": config.model_dump(),
            "start_time": start_time,
            "end_time": end_time,
        }

        with open(paths.metadata, "w") as f:
            json.dump(metadata, f, indent=2)

    def read_transcript(self, session_id: str) -> list[dict[str, Any]]:
        """Read transcript events for a session.

        Args:
            session_id: The session ID.

        Returns:
            List of event dictionaries.
        """
        paths = self._ensure_session_dir(session_id)

        if not paths.transcript.exists():
            return []

        events = []
        with open(paths.transcript) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        return events

    def read_vote_ledger(self, session_id: str) -> dict[str, Any] | None:
        """Read the vote ledger for a session.

        Args:
            session_id: The session ID.

        Returns:
            Vote ledger dictionary or None if not found.
        """
        paths = self._ensure_session_dir(session_id)

        if not paths.vote_ledger.exists():
            return None

        with open(paths.vote_ledger) as f:
            return json.load(f)

    def get_session_path(self, session_id: str) -> str:
        """Get the path to a session's artifacts directory.

        Args:
            session_id: The session ID.

        Returns:
            Path to the session directory.
        """
        return str(self.base_path / session_id)

    def list_sessions(self) -> list[str]:
        """List all session IDs.

        Returns:
            List of session IDs.
        """
        if not self.base_path.exists():
            return []

        return [
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith("council-")
        ]


# Global artifact store instance
_artifact_store: ArtifactStore | None = None


def get_artifact_store(base_path: str = "./out") -> ArtifactStore:
    """Get the global artifact store instance.

    Args:
        base_path: Base path for artifacts (only used on first call).

    Returns:
        The artifact store instance.
    """
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(base_path)
    return _artifact_store
