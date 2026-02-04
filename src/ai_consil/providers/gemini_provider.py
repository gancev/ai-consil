"""Google Gemini provider adapter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ai_consil.providers.base import ProviderAdapter, ProviderMessage, ProviderResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class GeminiProviderAdapter(ProviderAdapter):
    """Provider adapter for Google Gemini API."""

    name = "gemini"

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize Gemini provider.

        Args:
            model: Model identifier (e.g., "gemini-pro", "gemini-1.5-pro").
            **kwargs: Additional configuration.
        """
        super().__init__(model, **kwargs)
        self._client: Any = None
        self._model_instance: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError(
                    "Google GenerativeAI package not installed. "
                    "Install with: pip install google-generativeai"
                ) from e

            api_key = self.config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            self._client = genai
            self._model_instance = genai.GenerativeModel(self.model)
        return self._client

    def _convert_messages(
        self, messages: list[ProviderMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Gemini format."""
        system_prompt = None
        gemini_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                # Gemini uses "user" and "model" roles
                role = "user" if msg.role == "user" else "model"
                gemini_messages.append({
                    "role": role,
                    "parts": [msg.content],
                })

        return system_prompt, gemini_messages

    async def complete(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Generate a completion using Gemini API."""
        self._get_client()
        system_prompt, gemini_messages = self._convert_messages(messages)

        # Configure generation
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        # Create model with system instruction if provided
        if system_prompt:
            model = self._client.GenerativeModel(
                self.model,
                system_instruction=system_prompt,
            )
        else:
            model = self._model_instance

        # Start chat with history (all but last message)
        chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])

        # Send last message
        last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        response = await chat.send_message_async(
            last_message,
            generation_config=generation_config,
        )

        content = response.text if response.text else ""

        # Gemini doesn't provide detailed token counts in the same way
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        return ProviderResponse(
            content=content,
            finish_reason="stop",
            usage=usage,
        )

    async def stream(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion using Gemini API."""
        self._get_client()
        system_prompt, gemini_messages = self._convert_messages(messages)

        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        if system_prompt:
            model = self._client.GenerativeModel(
                self.model,
                system_instruction=system_prompt,
            )
        else:
            model = self._model_instance

        chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""

        response = await chat.send_message_async(
            last_message,
            generation_config=generation_config,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    @classmethod
    def is_available(cls) -> bool:
        """Check if Google API key is configured."""
        return bool(os.environ.get("GOOGLE_API_KEY"))

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Return required environment variables."""
        return ["GOOGLE_API_KEY"]
