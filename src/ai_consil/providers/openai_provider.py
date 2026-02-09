"""OpenAI provider adapter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ai_consil.providers.base import ProviderAdapter, ProviderMessage, ProviderResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class OpenAIProviderAdapter(ProviderAdapter):
    """Provider adapter for OpenAI API."""

    name = "openai"

    # Reasoning models that don't support temperature
    REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "o5")

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize OpenAI provider.

        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-4-turbo").
            **kwargs: Additional configuration (api_key, base_url, etc.).
        """
        super().__init__(model, **kwargs)
        self._client: Any = None

    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model (o-series)."""
        return any(self.model.startswith(p) for p in self.REASONING_MODEL_PREFIXES)

    def _get_client(self) -> Any:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                ) from e

            api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            base_url = self.config.get("base_url")

            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._client

    async def complete(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Generate a completion using OpenAI API."""
        client = self._get_client()

        # Use Responses API when web_search is enabled
        if self.config.get("web_search"):
            return await self._complete_with_responses_api(client, messages)

        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }
        if not self._is_reasoning_model():
            kwargs["temperature"] = temperature
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

    async def _complete_with_responses_api(
        self,
        client: Any,
        messages: list[ProviderMessage],
    ) -> ProviderResponse:
        """Generate a completion using OpenAI Responses API with web search."""
        input_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "tools": [{"type": "web_search_preview"}],
        }

        response = await client.responses.create(**kwargs)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return ProviderResponse(
            content=response.output_text or "",
            finish_reason="stop",
            usage=usage,
        )

    async def stream(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion using OpenAI API."""
        client = self._get_client()

        # Web search uses Responses API — fall back to non-streaming
        if self.config.get("web_search"):
            response = await self._complete_with_responses_api(client, messages)
            yield response.content
            return

        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
        }
        if not self._is_reasoning_model():
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(os.environ.get("OPENAI_API_KEY"))

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Return required environment variables."""
        return ["OPENAI_API_KEY"]
