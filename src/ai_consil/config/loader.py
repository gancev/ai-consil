"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_consil.api.schemas import CouncilConfig


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


def load_config_from_file(path: str | Path) -> CouncilConfig:
    """Load council configuration from a JSON file.

    Args:
        path: Path to the configuration file.

    Returns:
        Validated CouncilConfig object.

    Raises:
        ConfigError: If file not found or validation fails.
    """
    path = Path(path)

    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    if not path.is_file():
        raise ConfigError(f"Path is not a file: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in configuration file: {e}")

    return validate_config(data)


def validate_config(data: dict[str, Any]) -> CouncilConfig:
    """Validate configuration data.

    Args:
        data: Configuration dictionary.

    Returns:
        Validated CouncilConfig object.

    Raises:
        ConfigError: If validation fails.
    """
    try:
        return CouncilConfig.model_validate(data)
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"  {loc}: {err['msg']}")
        raise ConfigError(
            f"Configuration validation failed:\n" + "\n".join(errors)
        )


def resolve_config(
    config: CouncilConfig | str | dict[str, Any] | None,
) -> CouncilConfig | None:
    """Resolve configuration from various input types.

    Args:
        config: Configuration as:
            - CouncilConfig object (returned as-is)
            - String path to JSON file
            - Dictionary to validate
            - None (returns None)

    Returns:
        Validated CouncilConfig or None.

    Raises:
        ConfigError: If validation fails.
    """
    if config is None:
        default_path = os.environ.get("AI_CONSIL_DEFAULT_CONFIG")
        if default_path:
            return load_config_from_file(default_path)
        return None

    if isinstance(config, CouncilConfig):
        return config

    if isinstance(config, str):
        # Check if it's a file path
        if os.path.exists(config):
            return load_config_from_file(config)

        # Try to parse as JSON string
        try:
            data = json.loads(config)
            return validate_config(data)
        except json.JSONDecodeError:
            raise ConfigError(
                f"Configuration string is neither a valid file path "
                f"nor valid JSON: {config[:100]}..."
            )

    if isinstance(config, dict):
        return validate_config(config)

    raise ConfigError(
        f"Invalid configuration type: {type(config).__name__}. "
        f"Expected CouncilConfig, str, dict, or None."
    )


def get_default_config() -> CouncilConfig:
    """Get a minimal default configuration for testing.

    Returns:
        A default CouncilConfig with mock agents.
    """
    from ai_consil.api.schemas import AgentConfig

    return CouncilConfig(
        agents=[
            AgentConfig(
                id="analyst-1",
                role="analyst",
                provider="mock",
                model="mock-v1",
            ),
            AgentConfig(
                id="skeptic-1",
                role="skeptic",
                provider="mock",
                model="mock-v1",
            ),
        ],
        rounds=2,
        max_questions_per_agent=1,
    )
