"""Configuration validation tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_consil.api.schemas import AgentConfig, CouncilConfig, VotingSchedule
from ai_consil.config.loader import (
    ConfigError,
    load_config_from_file,
    resolve_config,
    validate_config,
)


class TestAgentConfig:
    """Tests for AgentConfig validation."""

    def test_valid_config(self) -> None:
        """Test valid agent configuration."""
        config = AgentConfig(
            id="analyst-1",
            role="analyst",
            provider="openai",
            model="gpt-4o",
        )
        assert config.id == "analyst-1"
        assert config.role == "analyst"
        assert config.provider == "openai"
        assert config.model == "gpt-4o"

    def test_empty_id_rejected(self) -> None:
        """Test that empty agent ID is rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(
                id="",
                role="analyst",
                provider="mock",
                model="mock-v1",
            )

    def test_whitespace_id_rejected(self) -> None:
        """Test that whitespace-only agent ID is rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(
                id="   ",
                role="analyst",
                provider="mock",
                model="mock-v1",
            )

    def test_custom_system_prompt(self) -> None:
        """Test custom system prompt is accepted."""
        config = AgentConfig(
            id="expert-1",
            role="domain_expert",
            system_prompt="You are an expert in quantum computing.",
            provider="mock",
            model="mock-v1",
        )
        assert config.system_prompt == "You are an expert in quantum computing."


class TestCouncilConfig:
    """Tests for CouncilConfig validation."""

    def test_valid_config(self) -> None:
        """Test valid council configuration."""
        config = CouncilConfig(
            agents=[
                AgentConfig(id="a1", role="analyst", provider="mock", model="m1"),
                AgentConfig(id="a2", role="skeptic", provider="mock", model="m1"),
            ],
            rounds=3,
            max_questions_per_agent=2,
            voting_schedule=VotingSchedule.EACH_ROUND,
            consensus_threshold=0.75,
            trace=True,
        )
        assert len(config.agents) == 2
        assert config.rounds == 3
        assert config.voting_schedule == VotingSchedule.EACH_ROUND

    def test_empty_agents_rejected(self) -> None:
        """Test that empty agents list is rejected."""
        with pytest.raises(ValidationError):
            CouncilConfig(agents=[])

    def test_rounds_must_be_positive(self) -> None:
        """Test that rounds must be positive."""
        with pytest.raises(ValidationError):
            CouncilConfig(
                agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
                rounds=0,
            )

        with pytest.raises(ValidationError):
            CouncilConfig(
                agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
                rounds=-1,
            )

    def test_rounds_max_limit(self) -> None:
        """Test that rounds has a maximum limit."""
        with pytest.raises(ValidationError):
            CouncilConfig(
                agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
                rounds=100,  # Over the limit of 10
            )

    def test_consensus_threshold_range(self) -> None:
        """Test that consensus threshold must be 0.0-1.0."""
        # Valid values
        config = CouncilConfig(
            agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
            consensus_threshold=0.0,
        )
        assert config.consensus_threshold == 0.0

        config = CouncilConfig(
            agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
            consensus_threshold=1.0,
        )
        assert config.consensus_threshold == 1.0

        # Invalid values
        with pytest.raises(ValidationError):
            CouncilConfig(
                agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
                consensus_threshold=-0.1,
            )

        with pytest.raises(ValidationError):
            CouncilConfig(
                agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
                consensus_threshold=1.5,
            )

    def test_voting_schedule_enum(self) -> None:
        """Test voting schedule enum validation."""
        for schedule in VotingSchedule:
            config = CouncilConfig(
                agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
                voting_schedule=schedule,
            )
            assert config.voting_schedule == schedule


class TestConfigLoader:
    """Tests for configuration file loading."""

    def test_load_valid_file(self) -> None:
        """Test loading a valid configuration file."""
        config_data = {
            "agents": [
                {"id": "a1", "role": "analyst", "provider": "mock", "model": "m1"},
            ],
            "rounds": 2,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()

            config = load_config_from_file(f.name)
            assert len(config.agents) == 1
            assert config.rounds == 2

    def test_load_nonexistent_file(self) -> None:
        """Test loading a nonexistent file raises error."""
        with pytest.raises(ConfigError, match="not found"):
            load_config_from_file("/nonexistent/path/config.json")

    def test_load_invalid_json(self) -> None:
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{ invalid json }")
            f.flush()

            with pytest.raises(ConfigError, match="Invalid JSON"):
                load_config_from_file(f.name)

    def test_load_invalid_config(self) -> None:
        """Test loading invalid configuration raises error."""
        config_data = {"agents": []}  # Empty agents

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()

            with pytest.raises(ConfigError, match="validation failed"):
                load_config_from_file(f.name)


class TestResolveConfig:
    """Tests for resolve_config function."""

    def test_resolve_none(self) -> None:
        """Test resolving None returns None."""
        result = resolve_config(None)
        assert result is None

    def test_resolve_config_object(self) -> None:
        """Test resolving a CouncilConfig object returns it as-is."""
        config = CouncilConfig(
            agents=[AgentConfig(id="a1", role="analyst", provider="mock", model="m1")],
        )
        result = resolve_config(config)
        assert result is config

    def test_resolve_dict(self) -> None:
        """Test resolving a dictionary validates and returns config."""
        config_dict = {
            "agents": [
                {"id": "a1", "role": "analyst", "provider": "mock", "model": "m1"},
            ],
        }
        result = resolve_config(config_dict)
        assert isinstance(result, CouncilConfig)
        assert len(result.agents) == 1

    def test_resolve_json_string(self) -> None:
        """Test resolving a JSON string validates and returns config."""
        config_json = json.dumps({
            "agents": [
                {"id": "a1", "role": "analyst", "provider": "mock", "model": "m1"},
            ],
        })
        result = resolve_config(config_json)
        assert isinstance(result, CouncilConfig)
        assert len(result.agents) == 1

    def test_resolve_file_path(self) -> None:
        """Test resolving a file path loads the config."""
        config_data = {
            "agents": [
                {"id": "a1", "role": "analyst", "provider": "mock", "model": "m1"},
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()

            result = resolve_config(f.name)
            assert isinstance(result, CouncilConfig)
            assert len(result.agents) == 1

    def test_resolve_invalid_string(self) -> None:
        """Test resolving an invalid string raises error."""
        with pytest.raises(ConfigError):
            resolve_config("not a file and not json")
