"""DeepSeek provider adapter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ai_consil.providers.base import ProviderAdapter, ProviderMessage, ProviderResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class DeepSeekProviderAdapter(ProviderAdapter):
    """Provider adapter for DeepSeek API.

    DeepSeek uses an OpenAI-compatible API.
    """

    name = "deepseek"

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize DeepSeek provider.

        Args:
            model: Model identifier (e.g., "deepseek-chat", "deepseek-coder").
            **kwargs: Additional configuration.
        """
        super().__init__(model, **kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the DeepSeek client (using OpenAI SDK)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed (required for DeepSeek). "
                    "Install with: pip install openai"
                ) from e

            api_key = self.config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
            base_url = self.config.get("base_url", self.DEEPSEEK_BASE_URL)

            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._client

    async def complete(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Generate a completion using DeepSeek API."""
        client = self._get_client()

        deepseek_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": deepseek_messages,
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
        """Stream a completion using DeepSeek API."""
        client = self._get_client()

        deepseek_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": deepseek_messages,
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
        """Check if DeepSeek API key is configured."""
        return bool(os.environ.get("DEEPSEEK_API_KEY"))

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Return required environment variables."""
        return ["DEEPSEEK_API_KEY"]
