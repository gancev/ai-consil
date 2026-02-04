"""Server-Sent Events (SSE) formatting and streaming utilities."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from ai_consil.api.schemas import (
    ChatCompletionChunk,
    CouncilEvent,
    DeltaMessage,
    StreamChoice,
)


def format_sse_event(data: dict[str, Any] | str) -> str:
    """Format data as an SSE event.

    Args:
        data: Data to format (dict will be JSON-encoded).

    Returns:
        SSE-formatted string.
    """
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = data

    return f"data: {data_str}\n\n"


def format_sse_done() -> str:
    """Format the SSE done event."""
    return "data: [DONE]\n\n"


def create_content_chunk(
    session_id: str,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    """Create a chat completion chunk for streaming content.

    Args:
        session_id: The session/completion ID.
        content: Content delta (if any).
        role: Role (usually only on first chunk).
        finish_reason: Finish reason (usually only on last chunk).

    Returns:
        A chat completion chunk.
    """
    delta = DeltaMessage(
        role=role,
        content=content,
    )

    choice = StreamChoice(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )

    return ChatCompletionChunk(
        id=session_id,
        created=int(time.time()),
        choices=[choice],
    )


async def stream_council_events(
    session_id: str,
    events: AsyncIterator[CouncilEvent],
    include_trace: bool = True,
) -> AsyncIterator[str]:
    """Stream council events as SSE.

    This generator yields SSE-formatted strings suitable for
    streaming HTTP responses.

    Args:
        session_id: The session ID.
        events: Async iterator of council events.
        include_trace: Whether to include trace events (council.event).

    Yields:
        SSE-formatted event strings.
    """
    # Send initial role chunk
    initial_chunk = create_content_chunk(session_id, role="assistant")
    yield format_sse_event(initial_chunk.model_dump())

    final_content = ""

    async for event in events:
        # Emit council event (if trace enabled)
        if include_trace:
            yield format_sse_event(event.model_dump())

        # Handle special events
        if event.event_type.value == "synthesis":
            # The synthesis content is the final answer
            final_content = event.content or ""

        elif event.event_type.value == "session_end":
            # Stream the final content
            if final_content:
                # Stream content in chunks for a more natural feel
                chunk_size = 50
                for i in range(0, len(final_content), chunk_size):
                    chunk_content = final_content[i:i + chunk_size]
                    content_chunk = create_content_chunk(
                        session_id,
                        content=chunk_content,
                    )
                    yield format_sse_event(content_chunk.model_dump())

    # Send final chunk with finish reason
    final_chunk = create_content_chunk(
        session_id,
        finish_reason="stop",
    )
    yield format_sse_event(final_chunk.model_dump())

    # Send done marker
    yield format_sse_done()


async def stream_non_council_response(
    session_id: str,
    content: str,
) -> AsyncIterator[str]:
    """Stream a simple non-council response.

    Used when council is not configured but streaming is requested.

    Args:
        session_id: The session ID.
        content: The content to stream.

    Yields:
        SSE-formatted event strings.
    """
    # Send initial role chunk
    initial_chunk = create_content_chunk(session_id, role="assistant")
    yield format_sse_event(initial_chunk.model_dump())

    # Stream content in chunks
    chunk_size = 50
    for i in range(0, len(content), chunk_size):
        chunk_content = content[i:i + chunk_size]
        content_chunk = create_content_chunk(
            session_id,
            content=chunk_content,
        )
        yield format_sse_event(content_chunk.model_dump())

    # Send final chunk
    final_chunk = create_content_chunk(
        session_id,
        finish_reason="stop",
    )
    yield format_sse_event(final_chunk.model_dump())

    yield format_sse_done()
