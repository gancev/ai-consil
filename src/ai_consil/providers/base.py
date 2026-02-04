"""Base provider interface and registry for extensible LLM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class ProviderMessage:
    """A message for provider communication."""

    role: str
    content: str


@dataclass
class ProviderResponse:
    """Response from a provider."""

    content: str
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None


class ProviderAdapter(ABC):
    """Abstract base class for LLM provider adapters.

    Implement this class to add support for new providers.
    """

    name: str = "base"

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize the provider adapter.

        Args:
            model: The model identifier to use.
            **kwargs: Additional provider-specific configuration.
        """
        self.model = model
        self.config = kwargs

    @abstractmethod
    async def complete(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Generate a completion for the given messages.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The provider's response.
        """
        ...

    async def stream(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion for the given messages.

        Default implementation falls back to non-streaming complete().
        Override for true streaming support.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Content chunks as they're generated.
        """
        response = await self.complete(messages, temperature, max_tokens)
        yield response.content

    @classmethod
    def is_available(cls) -> bool:
        """Check if this provider is available (e.g., API key configured).

        Override this to check for required environment variables.
        """
        return True

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Return list of required environment variables.

        Override this to specify what env vars are needed.
        """
        return []


# Type alias for provider factory functions
ProviderFactory = Callable[[str], ProviderAdapter]


class ProviderRegistry:
    """Registry for LLM provider adapters.

    Supports registering providers by name and creating instances.
    Extensible - third parties can register their own providers.
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[ProviderAdapter]] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        adapter_class: type[ProviderAdapter],
        aliases: list[str] | None = None,
    ) -> None:
        """Register a provider adapter.

        Args:
            name: Primary name for the provider.
            adapter_class: The adapter class to register.
            aliases: Optional list of alternative names.
        """
        self._providers[name.lower()] = adapter_class
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name.lower()

    def get(self, name: str, model: str, **kwargs: Any) -> ProviderAdapter:
        """Get a provider adapter instance.

        Args:
            name: Provider name or alias.
            model: Model identifier.
            **kwargs: Additional configuration.

        Returns:
            An initialized provider adapter.

        Raises:
            ValueError: If provider is not registered.
            RuntimeError: If provider is not available.
        """
        lookup_name = name.lower()

        # Resolve aliases
        if lookup_name in self._aliases:
            lookup_name = self._aliases[lookup_name]

        if lookup_name not in self._providers:
            available = list(self._providers.keys()) + list(self._aliases.keys())
            raise ValueError(
                f"Unknown provider '{name}'. Available providers: {sorted(available)}"
            )

        adapter_class = self._providers[lookup_name]

        if not adapter_class.is_available():
            env_vars = adapter_class.get_required_env_vars()
            raise RuntimeError(
                f"Provider '{name}' is not available. "
                f"Required environment variables: {env_vars}"
            )

        return adapter_class(model, **kwargs)

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return sorted(self._providers.keys())

    def list_available(self) -> list[str]:
        """List providers that are currently available."""
        return [
            name for name, cls in self._providers.items()
            if cls.is_available()
        ]

    def is_registered(self, name: str) -> bool:
        """Check if a provider is registered."""
        lookup_name = name.lower()
        return lookup_name in self._providers or lookup_name in self._aliases
