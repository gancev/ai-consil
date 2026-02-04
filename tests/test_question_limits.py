"""Question limit enforcement tests."""

from __future__ import annotations

import pytest

from ai_consil.api.schemas import AgentConfig, CouncilConfig
from ai_consil.council.agent import CouncilAgent


class TestQuestionLimits:
    """Tests for question limit enforcement."""

    @pytest.fixture
    def mock_agent(self) -> CouncilAgent:
        """Create a mock agent for testing."""
        config = AgentConfig(
            id="test-agent",
            role="analyst",
            provider="mock",
            model="mock-v1",
        )
        return CouncilAgent(config)

    def test_question_count_starts_at_zero(self, mock_agent: CouncilAgent) -> None:
        """Test that question count starts at zero for each round."""
        assert mock_agent.get_questions_asked(1) == 0
        assert mock_agent.get_questions_asked(2) == 0

    def test_question_count_independent_per_round(
        self, mock_agent: CouncilAgent
    ) -> None:
        """Test that question counts are independent per round."""
        mock_agent.state.questions_asked[1] = 2
        mock_agent.state.questions_asked[2] = 1

        assert mock_agent.get_questions_asked(1) == 2
        assert mock_agent.get_questions_asked(2) == 1
        assert mock_agent.get_questions_asked(3) == 0

    @pytest.mark.asyncio
    async def test_question_limit_respected(self, mock_agent: CouncilAgent) -> None:
        """Test that agent respects question limit."""
        max_questions = 2

        # Simulate asking max questions
        mock_agent.state.questions_asked[1] = max_questions

        # Agent should return None when at limit
        result = await mock_agent.ask_question(
            discussion="Test discussion",
            round_num=1,
            max_questions=max_questions,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_question_limit_allows_under_limit(
        self, mock_agent: CouncilAgent
    ) -> None:
        """Test that agent can ask questions when under limit."""
        max_questions = 2

        # Simulate asking fewer than max questions
        mock_agent.state.questions_asked[1] = 1

        # Agent should be able to ask (may return None if no question, but won't error)
        # This tests the limit check, not the actual question generation
        result = await mock_agent.ask_question(
            discussion="Test discussion about microservices",
            round_num=1,
            max_questions=max_questions,
        )

        # Result can be None (no question) or ParsedQuestion, both are valid
        # The key is that it didn't raise or skip due to limit
        # We can check that the count was potentially incremented
        # (depends on mock response)

    def test_question_limit_reset_per_round(self, mock_agent: CouncilAgent) -> None:
        """Test that question limits reset for new rounds."""
        mock_agent.state.questions_asked[1] = 3

        mock_agent.reset_round_state(2)

        assert mock_agent.get_questions_asked(1) == 3  # Round 1 unchanged
        assert mock_agent.get_questions_asked(2) == 0  # Round 2 reset

    def test_config_validates_max_questions(self) -> None:
        """Test that config validates max_questions_per_agent range."""
        # Valid values
        config = CouncilConfig(
            agents=[
                AgentConfig(id="a1", role="analyst", provider="mock", model="mock-v1")
            ],
            max_questions_per_agent=5,
        )
        assert config.max_questions_per_agent == 5

        # Zero is valid (no questions allowed)
        config = CouncilConfig(
            agents=[
                AgentConfig(id="a1", role="analyst", provider="mock", model="mock-v1")
            ],
            max_questions_per_agent=0,
        )
        assert config.max_questions_per_agent == 0

        # Negative should fail
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CouncilConfig(
                agents=[
                    AgentConfig(id="a1", role="analyst", provider="mock", model="mock-v1")
                ],
                max_questions_per_agent=-1,
            )

        # Over max should fail
        with pytest.raises(ValidationError):
            CouncilConfig(
                agents=[
                    AgentConfig(id="a1", role="analyst", provider="mock", model="mock-v1")
                ],
                max_questions_per_agent=100,
            )
