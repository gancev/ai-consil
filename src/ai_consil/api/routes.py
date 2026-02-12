"""FastAPI routes for OpenAI-compatible API."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ai_consil.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    CouncilConfig,
    Usage,
)
from ai_consil.config.loader import ConfigError, resolve_config
from ai_consil.council.orchestrator import CouncilOrchestrator
from ai_consil.storage.artifacts import get_artifact_store
from ai_consil.streaming.sse import stream_council_events

router = APIRouter()


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> StreamingResponse | JSONResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both standard completions and council deliberations.
    When `council` is provided, runs a multi-agent deliberation.

    Args:
        request: The chat completion request.

    Returns:
        Streaming SSE response or JSON response based on `stream` parameter.
    """
    # Resolve council configuration
    try:
        council_config = resolve_config(request.council)
    except ConfigError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # If no council config, return a simple response
    if council_config is None:
        return await _handle_non_council_request(request)

    # Run council deliberation
    return await _handle_council_request(request, council_config)


async def _handle_non_council_request(
    request: ChatCompletionRequest,
) -> StreamingResponse | JSONResponse:
    """Handle a request without council configuration.

    Returns a simple message explaining that council config is required.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    message_content = (
        "AI Council requires a `council` configuration to deliberate. "
        "Please provide a council configuration with agents, rounds, and other settings. "
        "See documentation for examples."
    )

    if request.stream:
        from ai_consil.streaming.sse import stream_non_council_response

        return StreamingResponse(
            stream_non_council_response(response_id, message_content),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return JSONResponse(
        content=ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(content=message_content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=sum(len(m.content.split()) for m in request.messages),
                completion_tokens=len(message_content.split()),
                total_tokens=sum(len(m.content.split()) for m in request.messages)
                + len(message_content.split()),
            ),
        ).model_dump()
    )


async def _handle_council_request(
    request: ChatCompletionRequest,
    config: CouncilConfig,
) -> StreamingResponse | JSONResponse:
    """Handle a council deliberation request."""
    # Extract topic from messages (last user message)
    topic = ""
    for msg in reversed(request.messages):
        if msg.role.value == "user":
            topic = msg.content
            break

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="No user message found in request",
        )

    # Create orchestrator
    orchestrator = CouncilOrchestrator(
        config=config,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # Get artifact store
    store = get_artifact_store()

    # Write initial metadata
    start_time = datetime.now(timezone.utc).isoformat()
    store.write_metadata(
        session_id=orchestrator.session_id,
        config=config,
        topic=topic,
        start_time=start_time,
    )

    if request.stream:
        return await _handle_streaming_council(
            orchestrator, topic, config, store, start_time
        )

    return await _handle_non_streaming_council(
        orchestrator, topic, config, store, start_time
    )


async def _handle_streaming_council(
    orchestrator: CouncilOrchestrator,
    topic: str,
    config: CouncilConfig,
    store: Any,
    start_time: str,
) -> StreamingResponse:
    """Handle streaming council deliberation."""

    async def generate():
        try:
            async for sse_chunk in stream_council_events(
                session_id=orchestrator.session_id,
                events=_events_with_storage(orchestrator, topic, store),
                include_trace=config.trace,
            ):
                yield sse_chunk

            # Write final artifacts
            trace = orchestrator._build_trace()
            store.write_vote_ledger(orchestrator.session_id, trace)
            if orchestrator.state:
                store.write_final_answer(
                    orchestrator.session_id,
                    orchestrator.state.final_answer,
                    topic,
                )
                store.write_metadata(
                    session_id=orchestrator.session_id,
                    config=config,
                    topic=topic,
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc).isoformat(),
                )

        except Exception as e:
            # Emit error event
            error_chunk = f'data: {{"error": "{str(e)}"}}\n\n'
            yield error_chunk
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _events_with_storage(orchestrator, topic, store):
    """Wrap event stream to also store events."""
    async for event in orchestrator.run_stream(topic):
        # Store event in transcript
        store.append_event(orchestrator.session_id, event)
        yield event


async def _handle_non_streaming_council(
    orchestrator: CouncilOrchestrator,
    topic: str,
    config: CouncilConfig,
    store: Any,
    start_time: str,
) -> JSONResponse:
    """Handle non-streaming council deliberation."""
    try:
        # Run deliberation
        final_answer, trace, _events = await orchestrator.run(topic)

        # Store artifacts
        store.write_vote_ledger(orchestrator.session_id, trace)
        store.write_final_answer(orchestrator.session_id, final_answer, topic)
        store.write_metadata(
            session_id=orchestrator.session_id,
            config=config,
            topic=topic,
            start_time=start_time,
            end_time=datetime.now(timezone.utc).isoformat(),
        )

        # Build response
        response = ChatCompletionResponse(
            id=orchestrator.session_id,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(content=final_answer),
                    finish_reason="stop",
                )
            ],
            council_trace=trace if config.trace else None,
        )

        return JSONResponse(content=response.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# List available providers
@router.get("/v1/providers")
async def list_providers() -> dict[str, Any]:
    """List available LLM providers."""
    from ai_consil.providers import registry

    return {
        "providers": registry.list_providers(),
        "available": registry.list_available(),
    }


# List sessions
@router.get("/v1/council/sessions")
async def list_sessions() -> dict[str, list[str]]:
    """List all council sessions."""
    store = get_artifact_store()
    return {"sessions": store.list_sessions()}


# Get session transcript
@router.get("/v1/council/sessions/{session_id}/transcript")
async def get_transcript(session_id: str) -> dict[str, Any]:
    """Get the transcript for a session."""
    store = get_artifact_store()
    events = store.read_transcript(session_id)

    if not events:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "events": events}


# Get session votes
@router.get("/v1/council/sessions/{session_id}/votes")
async def get_votes(session_id: str) -> dict[str, Any]:
    """Get the vote ledger for a session."""
    store = get_artifact_store()
    ledger = store.read_vote_ledger(session_id)

    if ledger is None:
        raise HTTPException(status_code=404, detail="Vote ledger not found")

    return ledger
