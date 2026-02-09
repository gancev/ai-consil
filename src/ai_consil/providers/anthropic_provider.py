"""Anthropic provider adapter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ai_consil.providers.base import ProviderAdapter, ProviderMessage, ProviderResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class AnthropicProviderAdapter(ProviderAdapter):
    """Provider adapter for Anthropic API."""

    name = "anthropic"

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize Anthropic provider.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514", "claude-3-opus-20240229").
            **kwargs: Additional configuration.
        """
        super().__init__(model, **kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                ) from e

            api_key = self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    def _convert_messages(
        self, messages: list[ProviderMessage]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert messages to Anthropic format, extracting system prompt."""
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        return system_prompt, anthropic_messages

    async def complete(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Generate a completion using Anthropic API."""
        client = self._get_client()
        system_prompt, anthropic_messages = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        if self.config.get("web_search"):
            kwargs["tools"] = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": self.config.get("max_searches", 5),
            }]

        response = await client.messages.create(**kwargs)

        # Extract text from all text blocks (web search responses have multiple blocks)
        content = ""
        if response.content:
            text_parts = [block.text for block in response.content if hasattr(block, "text")]
            content = "\n".join(text_parts)

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return ProviderResponse(
            content=content,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
        )

    async def stream(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion using Anthropic API."""
        client = self._get_client()
        system_prompt, anthropic_messages = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        if self.config.get("web_search"):
            kwargs["tools"] = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": self.config.get("max_searches", 5),
            }]

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    @classmethod
    def is_available(cls) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Return required environment variables."""
        return ["ANTHROPIC_API_KEY"]
