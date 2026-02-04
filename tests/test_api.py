"""API endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, test_client: TestClient) -> None:
        """Test that health endpoint returns healthy status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestProvidersEndpoint:
    """Tests for the providers endpoint."""

    def test_list_providers(self, test_client: TestClient) -> None:
        """Test that providers endpoint returns provider list."""
        response = test_client.get("/v1/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "available" in data
        assert "mock" in data["providers"]
        assert "mock" in data["available"]  # Mock is always available


class TestChatCompletions:
    """Tests for the chat completions endpoint."""

    def test_no_council_config(self, test_client: TestClient) -> None:
        """Test request without council config returns guidance message."""
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "ai-consil",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "council" in data["choices"][0]["message"]["content"].lower()

    def test_basic_completion(
        self, test_client: TestClient, sample_request: dict
    ) -> None:
        """Test basic non-streaming completion with council config."""
        response = test_client.post("/v1/chat/completions", json=sample_request)
        assert response.status_code == 200
        data = response.json()

        # Check OpenAI-compatible structure
        assert "id" in data
        assert data["id"].startswith("council-")
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_completion_with_trace(
        self, test_client: TestClient, sample_request: dict
    ) -> None:
        """Test that trace is included when requested."""
        sample_request["council"]["trace"] = True
        response = test_client.post("/v1/chat/completions", json=sample_request)
        assert response.status_code == 200
        data = response.json()

        assert "council_trace" in data
        assert data["council_trace"] is not None
        assert "session_id" in data["council_trace"]
        assert "rounds" in data["council_trace"]

    def test_completion_without_trace(
        self, test_client: TestClient, sample_request: dict
    ) -> None:
        """Test that trace is excluded when not requested."""
        sample_request["council"]["trace"] = False
        response = test_client.post("/v1/chat/completions", json=sample_request)
        assert response.status_code == 200
        data = response.json()

        assert data.get("council_trace") is None

    def test_invalid_config_rejected(self, test_client: TestClient) -> None:
        """Test that invalid config returns 422."""
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "ai-consil",
                "messages": [{"role": "user", "content": "Test"}],
                "council": {"agents": []},  # Empty agents list
            },
        )
        assert response.status_code == 422

    def test_missing_user_message(
        self, test_client: TestClient, mock_council_config_dict: dict
    ) -> None:
        """Test that request without user message returns 400."""
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "ai-consil",
                "messages": [{"role": "system", "content": "You are helpful"}],
                "council": mock_council_config_dict,
            },
        )
        assert response.status_code == 400


class TestSessionEndpoints:
    """Tests for session-related endpoints."""

    def test_list_sessions(self, test_client: TestClient) -> None:
        """Test listing sessions."""
        response = test_client.get("/v1/council/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_get_nonexistent_transcript(self, test_client: TestClient) -> None:
        """Test getting transcript for nonexistent session."""
        response = test_client.get("/v1/council/sessions/nonexistent/transcript")
        assert response.status_code == 404

    def test_get_nonexistent_votes(self, test_client: TestClient) -> None:
        """Test getting votes for nonexistent session."""
        response = test_client.get("/v1/council/sessions/nonexistent/votes")
        assert response.status_code == 404
