"""LLM Provider adapters with extensible registry."""

from ai_consil.providers.base import ProviderAdapter, ProviderRegistry
from ai_consil.providers.registry import registry, get_provider, register_provider

__all__ = [
    "ProviderAdapter",
    "ProviderRegistry",
    "registry",
    "get_provider",
    "register_provider",
]
