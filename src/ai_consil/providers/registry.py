"""Provider registry with built-in providers registered."""

from __future__ import annotations

from typing import Any

from ai_consil.providers.base import ProviderAdapter, ProviderRegistry

# Create the global registry
registry = ProviderRegistry()


def _register_builtin_providers() -> None:
    """Register all built-in providers."""
    # Mock provider (always available)
    from ai_consil.providers.mock import MockProviderAdapter

    registry.register("mock", MockProviderAdapter, aliases=["test", "demo"])

    # OpenAI
    from ai_consil.providers.openai_provider import OpenAIProviderAdapter

    registry.register("openai", OpenAIProviderAdapter, aliases=["gpt", "chatgpt"])

    # Anthropic
    from ai_consil.providers.anthropic_provider import AnthropicProviderAdapter

    registry.register("anthropic", AnthropicProviderAdapter, aliases=["claude"])

    # Google Gemini
    from ai_consil.providers.gemini_provider import GeminiProviderAdapter

    registry.register("gemini", GeminiProviderAdapter, aliases=["google", "bard"])

    # Groq
    from ai_consil.providers.groq_provider import GroqProviderAdapter

    registry.register("groq", GroqProviderAdapter, aliases=["groq-cloud"])

    # DeepSeek
    from ai_consil.providers.deepseek_provider import DeepSeekProviderAdapter

    registry.register("deepseek", DeepSeekProviderAdapter, aliases=["ds"])


# Register built-in providers on module import
_register_builtin_providers()


def get_provider(name: str, model: str, **kwargs: Any) -> ProviderAdapter:
    """Get a provider adapter instance.

    Args:
        name: Provider name or alias.
        model: Model identifier.
        **kwargs: Additional configuration.

    Returns:
        An initialized provider adapter.

    Example:
        >>> adapter = get_provider("openai", "gpt-4o")
        >>> response = await adapter.complete(messages)
    """
    return registry.get(name, model, **kwargs)


def register_provider(
    name: str,
    adapter_class: type[ProviderAdapter],
    aliases: list[str] | None = None,
) -> None:
    """Register a custom provider adapter.

    Args:
        name: Primary name for the provider.
        adapter_class: The adapter class to register.
        aliases: Optional list of alternative names.

    Example:
        >>> class MyCustomProvider(ProviderAdapter):
        ...     name = "custom"
        ...     async def complete(self, messages, **kwargs):
        ...         # Custom implementation
        ...         pass
        >>>
        >>> register_provider("my-custom", MyCustomProvider)
    """
    registry.register(name, adapter_class, aliases)
