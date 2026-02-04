"""Mock provider adapter for testing and demos."""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING, Any

from ai_consil.providers.base import ProviderAdapter, ProviderMessage, ProviderResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class MockProviderAdapter(ProviderAdapter):
    """Mock provider for deterministic testing and demos.

    Generates predictable responses based on input hashing for reproducibility.
    """

    name = "mock"

    # Predefined response templates for different contexts
    ANALYSIS_TEMPLATES = [
        "After careful analysis, I believe this approach has merit. The key considerations are: "
        "1) feasibility given current constraints, 2) potential risks that need mitigation, "
        "and 3) alignment with stated objectives. My assessment is generally positive with caveats.",
        "From my perspective as {role}, the proposal raises several important points. "
        "The strengths include clear problem definition and practical scope. "
        "However, we should consider edge cases and implementation complexity.",
        "Examining this from a {role} standpoint: The core idea is sound, but execution "
        "will be critical. I recommend focusing on incremental delivery and continuous validation.",
    ]

    QUESTION_TEMPLATES = [
        "How do you account for {topic} in your analysis?",
        "What evidence supports your position on {topic}?",
        "Have you considered the implications of {topic}?",
        "Can you elaborate on how {topic} affects your conclusion?",
    ]

    ANSWER_TEMPLATES = [
        "That's a fair point. Regarding {topic}: my analysis accounts for this by "
        "considering multiple scenarios and their likelihood. The key factor is {factor}.",
        "Good question. On {topic}: I've factored this in through careful consideration "
        "of precedents and current conditions. The evidence suggests {conclusion}.",
        "To address {topic}: this is indeed relevant. My position incorporates this "
        "consideration, weighing it against other factors like {factor}.",
    ]

    VOTE_REASONING_TEMPLATES = [
        "Based on the deliberation, I {position} because the arguments presented "
        "demonstrate {reasoning}. The evidence and counter-arguments have been weighed.",
        "After considering all perspectives, I {position}. The key factors influencing "
        "my decision are: {reasoning}. This aligns with my role as {role}.",
        "My vote to {position} reflects the balance of arguments. While there are "
        "valid concerns, {reasoning} tips the scale in this direction.",
    ]

    SYNTHESIS_TEMPLATES = [
        "The council has deliberated on this matter. The majority view supports "
        "{conclusion}. Key points of agreement include {points}. Areas of ongoing "
        "discussion involve {concerns}.",
        "After {rounds} rounds of deliberation, the council reaches the following "
        "synthesis: {conclusion}. This reflects input from all members, balancing "
        "{points} against {concerns}.",
    ]

    def __init__(self, model: str, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize mock provider.

        Args:
            model: Model identifier (used for response variation).
            seed: Random seed for reproducibility.
            **kwargs: Additional configuration.
        """
        super().__init__(model, **kwargs)
        self.seed = seed
        self._rng = random.Random(seed)

    def _hash_input(self, text: str) -> int:
        """Create a deterministic hash from input text."""
        return int(hashlib.md5(text.encode()).hexdigest(), 16)

    def _select_template(self, templates: list[str], context: str) -> str:
        """Select a template deterministically based on context."""
        hash_val = self._hash_input(context)
        idx = hash_val % len(templates)
        return templates[idx]

    def _generate_response(self, messages: list[ProviderMessage]) -> str:
        """Generate a mock response based on message context."""
        if not messages:
            return "No input provided."

        last_message = messages[-1].content.lower()
        context = " ".join(m.content for m in messages)

        # Detect context type and generate appropriate response
        if "vote" in last_message or "position" in last_message:
            return self._generate_vote_response(context)
        elif "question" in last_message or "ask" in last_message or "?" in last_message:
            if "answer" in last_message or "respond" in last_message:
                return self._generate_answer(context)
            return self._generate_question(context)
        elif "synthesize" in last_message or "conclude" in last_message:
            return self._generate_synthesis(context)
        else:
            return self._generate_analysis(context)

    def _generate_analysis(self, context: str) -> str:
        """Generate an analysis response."""
        template = self._select_template(self.ANALYSIS_TEMPLATES, context)
        # Extract role from context if present
        role = "analyst"
        if "skeptic" in context.lower():
            role = "skeptic"
        elif "advocate" in context.lower():
            role = "advocate"
        elif "pragmatist" in context.lower():
            role = "pragmatist"
        return template.format(role=role)

    def _generate_question(self, context: str) -> str:
        """Generate a question."""
        template = self._select_template(self.QUESTION_TEMPLATES, context)
        topics = ["scalability", "security", "cost", "timeline", "risks", "alternatives"]
        topic = topics[self._hash_input(context) % len(topics)]
        return template.format(topic=topic)

    def _generate_answer(self, context: str) -> str:
        """Generate an answer to a question."""
        template = self._select_template(self.ANSWER_TEMPLATES, context)
        topics = ["this aspect", "the concern raised", "implementation details"]
        factors = ["careful planning", "industry best practices", "empirical evidence"]
        conclusions = ["a measured approach is best", "the benefits outweigh risks", "further analysis needed"]

        topic = topics[self._hash_input(context) % len(topics)]
        factor = factors[self._hash_input(context + "f") % len(factors)]
        conclusion = conclusions[self._hash_input(context + "c") % len(conclusions)]

        return template.format(topic=topic, factor=factor, conclusion=conclusion)

    def _generate_vote_response(self, context: str) -> str:
        """Generate a vote with reasoning in structured format."""
        hash_val = self._hash_input(context)

        # Determine position based on hash
        positions = ["support", "oppose", "abstain"]
        weights = [0.5, 0.35, 0.15]  # More likely to support or oppose than abstain
        cumulative = 0.0
        normalized_hash = (hash_val % 1000) / 1000.0
        position = positions[-1]
        for pos, weight in zip(positions, weights):
            cumulative += weight
            if normalized_hash < cumulative:
                position = pos
                break

        # Generate confidence (0.5 to 0.95)
        confidence = 0.5 + (hash_val % 45) / 100.0

        # Generate reasoning
        reasonings = [
            "the evidence presented is compelling",
            "the risks have been adequately addressed",
            "the benefits clearly outweigh the costs",
            "there are still unresolved concerns",
            "more information is needed before deciding",
        ]
        reasoning = reasonings[hash_val % len(reasonings)]

        role = "analyst"
        if "skeptic" in context.lower():
            role = "skeptic"
        elif "advocate" in context.lower():
            role = "advocate"

        template = self._select_template(self.VOTE_REASONING_TEMPLATES, context)
        full_reasoning = template.format(
            position=position,
            reasoning=reasoning,
            role=role,
        )

        # Return structured format that can be parsed
        return f"VOTE:{position}|CONFIDENCE:{confidence:.2f}|REASONING:{full_reasoning}"

    def _generate_synthesis(self, context: str) -> str:
        """Generate a synthesis/conclusion."""
        template = self._select_template(self.SYNTHESIS_TEMPLATES, context)
        conclusions = [
            "a phased approach with careful monitoring",
            "proceeding with the proposed plan with modifications",
            "further investigation before commitment",
        ]
        points = [
            "feasibility and strategic alignment",
            "risk mitigation and resource availability",
            "stakeholder support and clear objectives",
        ]
        concerns = [
            "timeline pressures and resource constraints",
            "edge cases and failure modes",
            "long-term maintainability",
        ]

        hash_val = self._hash_input(context)
        return template.format(
            conclusion=conclusions[hash_val % len(conclusions)],
            points=points[hash_val % len(points)],
            concerns=concerns[hash_val % len(concerns)],
            rounds="multiple",
        )

    async def complete(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """Generate a mock completion."""
        content = self._generate_response(messages)

        # Simulate token counts
        prompt_tokens = sum(len(m.content.split()) for m in messages) * 2
        completion_tokens = len(content.split()) * 2

        return ProviderResponse(
            content=content,
            finish_reason="stop",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    async def stream(
        self,
        messages: list[ProviderMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a mock completion word by word."""
        content = self._generate_response(messages)
        words = content.split()

        for i, word in enumerate(words):
            if i > 0:
                yield " "
            yield word

    @classmethod
    def is_available(cls) -> bool:
        """Mock provider is always available."""
        return True

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """No env vars required."""
        return []
