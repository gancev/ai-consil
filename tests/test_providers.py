"""Provider tests."""

from __future__ import annotations

import pytest

from ai_consil.providers import get_provider, register_provider, registry
from ai_consil.providers.base import ProviderAdapter, ProviderMessage, ProviderResponse
from ai_consil.providers.mock import MockProviderAdapter


class TestMockProvider:
    """Tests for MockProviderAdapter."""

    def test_mock_is_available(self) -> None:
        """Test that mock provider is always available."""
        assert MockProviderAdapter.is_available()
        assert MockProviderAdapter.get_required_env_vars() == []

    @pytest.mark.asyncio
    async def test_mock_deterministic(self) -> None:
        """Test that mock provider with same seed produces same output."""
        adapter1 = MockProviderAdapter("mock-v1", seed=42)
        adapter2 = MockProviderAdapter("mock-v1", seed=42)

        messages = [ProviderMessage(role="user", content="Test question")]

        response1 = await adapter1.complete(messages)
        response2 = await adapter2.complete(messages)

        assert response1.content == response2.content

    @pytest.mark.asyncio
    async def test_mock_complete(self) -> None:
        """Test mock provider completion."""
        adapter = MockProviderAdapter("mock-v1")
        messages = [ProviderMessage(role="user", content="Analyze this proposal")]

        response = await adapter.complete(messages)

        assert isinstance(response, ProviderResponse)
        assert response.content
        assert response.finish_reason == "stop"
        assert response.usage is not None

    @pytest.mark.asyncio
    async def test_mock_stream(self) -> None:
        """Test mock provider streaming."""
        adapter = MockProviderAdapter("mock-v1")
        messages = [ProviderMessage(role="user", content="Analyze this")]

        chunks = []
        async for chunk in adapter.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_content = "".join(chunks)
        assert len(full_content) > 0

    @pytest.mark.asyncio
    async def test_mock_vote_response(self) -> None:
        """Test mock provider vote response format."""
        adapter = MockProviderAdapter("mock-v1")
        messages = [
            ProviderMessage(
                role="user",
                content="Cast your vote on this proposal. Use VOTE: format.",
            )
        ]

        response = await adapter.complete(messages)

        # Should contain vote format
        assert "VOTE:" in response.content
        assert "CONFIDENCE:" in response.content
        assert "REASONING:" in response.content


class TestProviderRegistry:
    """Tests for the provider registry."""

    def test_mock_registered(self) -> None:
        """Test that mock provider is registered."""
        assert registry.is_registered("mock")
        assert "mock" in registry.list_providers()

    def test_all_builtin_providers_registered(self) -> None:
        """Test that all built-in providers are registered."""
        expected = ["mock", "openai", "anthropic", "gemini", "groq", "deepseek"]
        for name in expected:
            assert registry.is_registered(name), f"Provider {name} not registered"

    def test_aliases_work(self) -> None:
        """Test that provider aliases work."""
        # Mock aliases
        assert registry.is_registered("test")
        assert registry.is_registered("demo")

        # OpenAI aliases
        assert registry.is_registered("gpt")
        assert registry.is_registered("chatgpt")

        # Gemini aliases
        assert registry.is_registered("google")

    def test_get_mock_provider(self) -> None:
        """Test getting mock provider."""
        adapter = get_provider("mock", "mock-v1")
        assert isinstance(adapter, MockProviderAdapter)
        assert adapter.model == "mock-v1"

    def test_get_unknown_provider_raises(self) -> None:
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent", "model")

    def test_list_available(self) -> None:
        """Test listing available providers."""
        available = registry.list_available()
        assert "mock" in available  # Mock is always available


class TestCustomProviderRegistration:
    """Tests for registering custom providers."""

    def test_register_custom_provider(self) -> None:
        """Test registering a custom provider."""

        class CustomProvider(ProviderAdapter):
            name = "custom-test"

            async def complete(
                self,
                messages: list[ProviderMessage],
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> ProviderResponse:
                return ProviderResponse(content="Custom response")

            @classmethod
            def is_available(cls) -> bool:
                return True

        register_provider("custom-test", CustomProvider, aliases=["ct"])

        assert registry.is_registered("custom-test")
        assert registry.is_registered("ct")

        adapter = get_provider("custom-test", "any-model")
        assert isinstance(adapter, CustomProvider)

    @pytest.mark.asyncio
    async def test_custom_provider_works(self) -> None:
        """Test that custom provider can be used."""

        class WorkingProvider(ProviderAdapter):
            name = "working-test"

            async def complete(
                self,
                messages: list[ProviderMessage],
                temperature: float = 0.7,
                max_tokens: int | None = None,
            ) -> ProviderResponse:
                return ProviderResponse(content=f"Echoing: {messages[-1].content}")

            @classmethod
            def is_available(cls) -> bool:
                return True

        register_provider("working-test", WorkingProvider)

        adapter = get_provider("working-test", "model")
        messages = [ProviderMessage(role="user", content="Hello")]
        response = await adapter.complete(messages)

        assert "Echoing: Hello" in response.content
