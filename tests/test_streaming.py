"""Streaming (SSE) tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


class TestStreamingResponse:
    """Tests for streaming SSE responses."""

    def test_stream_produces_sse(
        self, test_client: TestClient, sample_stream_request: dict
    ) -> None:
        """Test that streaming request produces SSE format."""
        with test_client.stream(
            "POST", "/v1/chat/completions", json=sample_stream_request
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            # Read some content
            content = b""
            for chunk in response.iter_bytes():
                content += chunk
                # Stop after getting some data
                if len(content) > 100:
                    break

            content_str = content.decode("utf-8")
            assert "data: " in content_str

    def test_stream_ends_with_done(
        self, test_client: TestClient, sample_stream_request: dict
    ) -> None:
        """Test that stream terminates with [DONE]."""
        with test_client.stream(
            "POST", "/v1/chat/completions", json=sample_stream_request
        ) as response:
            assert response.status_code == 200

            content = response.read().decode("utf-8")
            assert "data: [DONE]" in content

    def test_stream_council_events(
        self, test_client: TestClient, sample_stream_request: dict
    ) -> None:
        """Test that council events appear in stream."""
        sample_stream_request["council"]["trace"] = True

        with test_client.stream(
            "POST", "/v1/chat/completions", json=sample_stream_request
        ) as response:
            content = response.read().decode("utf-8")

            # Parse SSE events
            events = []
            for line in content.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        events.append(data)
                    except json.JSONDecodeError:
                        pass

            # Should have council events
            council_events = [e for e in events if e.get("object") == "council.event"]
            assert len(council_events) > 0

            # Should have session_start event
            event_types = [e.get("event_type") for e in council_events]
            assert "session_start" in event_types
            assert "session_end" in event_types

    def test_stream_final_content(
        self, test_client: TestClient, sample_stream_request: dict
    ) -> None:
        """Test that final answer is streamed as chat completion chunks."""
        with test_client.stream(
            "POST", "/v1/chat/completions", json=sample_stream_request
        ) as response:
            content = response.read().decode("utf-8")

            # Parse SSE events
            chunk_events = []
            for line in content.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        if data.get("object") == "chat.completion.chunk":
                            chunk_events.append(data)
                    except json.JSONDecodeError:
                        pass

            # Should have content chunks
            assert len(chunk_events) > 0

            # Check structure
            for chunk in chunk_events:
                assert "id" in chunk
                assert "choices" in chunk
                assert len(chunk["choices"]) == 1
                assert "delta" in chunk["choices"][0]

    def test_stream_without_trace(
        self, test_client: TestClient, sample_stream_request: dict
    ) -> None:
        """Test streaming without trace excludes council events."""
        sample_stream_request["council"]["trace"] = False

        with test_client.stream(
            "POST", "/v1/chat/completions", json=sample_stream_request
        ) as response:
            content = response.read().decode("utf-8")

            # Parse SSE events
            events = []
            for line in content.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        events.append(data)
                    except json.JSONDecodeError:
                        pass

            # Should NOT have council events when trace=False
            council_events = [e for e in events if e.get("object") == "council.event"]
            assert len(council_events) == 0
