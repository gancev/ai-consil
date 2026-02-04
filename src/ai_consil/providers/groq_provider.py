"""Groq provider adapter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ai_consil.providers.base import ProviderAdapter, ProviderMessage, ProviderResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class GroqProviderAdapter(ProviderAdapter):
    """Provider adapter for Groq API.

    Groq uses an OpenAI-compatible API, so this is similar to the OpenAI adapter.
    """

    name = "groq"

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize Groq provider.

        Args:
            model: Model identifier (e.g., "llama-3.1-70b-versatile", "mixtral-8x7b-32768").
            **kwargs: Additional configuration.
        """
        super().__init__(model, **kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the Groq client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
            except ImportError as e:
                raise ImportError(
                    "Groq package not installed. Install with: pip install groq"
                ) from e

            api_key = self.config.get("api_key") or os.environ.get("GROQ_API_KEY")
            self._client = AsyncGroq(api_key=api_key)
        return self._client

    async def complete(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Generate a completion using Groq API."""
        client = self._get_client()

        groq_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": groq_messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ProviderResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

    async def stream(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion using Groq API."""
        client = self._get_client()

        groq_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": groq_messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @classmethod
    def is_available(cls) -> bool:
        """Check if Groq API key is configured."""
        return bool(os.environ.get("GROQ_API_KEY"))

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Return required environment variables."""
        return ["GROQ_API_KEY"]
