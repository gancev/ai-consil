"""Pytest fixtures for ai-consil tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from ai_consil.api.schemas import AgentConfig, CouncilConfig, VotingSchedule
from ai_consil.main import app


@pytest.fixture
def test_client() -> TestClient:
    """Create a synchronous test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def mock_council_config() -> CouncilConfig:
    """Create a mock council configuration for testing."""
    return CouncilConfig(
        agents=[
            AgentConfig(
                id="analyst-test",
                role="analyst",
                provider="mock",
                model="mock-v1",
            ),
            AgentConfig(
                id="skeptic-test",
                role="skeptic",
                provider="mock",
                model="mock-v1",
            ),
        ],
        rounds=1,
        max_questions_per_agent=1,
        voting_schedule=VotingSchedule.EACH_ROUND,
        consensus_threshold=0.67,
        trace=True,
    )


@pytest.fixture
def mock_council_config_dict() -> dict:
    """Create a mock council configuration as a dictionary."""
    return {
        "agents": [
            {
                "id": "analyst-test",
                "role": "analyst",
                "provider": "mock",
                "model": "mock-v1",
            },
            {
                "id": "skeptic-test",
                "role": "skeptic",
                "provider": "mock",
                "model": "mock-v1",
            },
        ],
        "rounds": 1,
        "max_questions_per_agent": 1,
        "voting_schedule": "each_round",
        "consensus_threshold": 0.67,
        "trace": True,
    }


@pytest.fixture
def sample_request(mock_council_config_dict: dict) -> dict:
    """Create a sample chat completion request."""
    return {
        "model": "ai-consil",
        "messages": [
            {"role": "user", "content": "Should we use microservices or monolith?"}
        ],
        "stream": False,
        "temperature": 0.7,
        "council": mock_council_config_dict,
    }


@pytest.fixture
def sample_stream_request(mock_council_config_dict: dict) -> dict:
    """Create a sample streaming chat completion request."""
    return {
        "model": "ai-consil",
        "messages": [
            {"role": "user", "content": "Should we use microservices or monolith?"}
        ],
        "stream": True,
        "temperature": 0.7,
        "council": mock_council_config_dict,
    }
