"""Shared pytest fixtures.

The whole suite runs offline: no API keys, no network, no model downloads.
``MockLLMClient`` implements the :class:`BaseLLMClient` contract and returns a
deterministic response so tests stay fast and reproducible.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import pytest

from llm_diagnostic.core.llm_client import BaseLLMClient, LLMResponse


class MockLLMClient(BaseLLMClient):
    """Deterministic, offline LLM client for tests.

    Pass ``response_fn`` to derive the answer from the prompt (useful to make a
    test "pass" by echoing the expected output); otherwise a fixed string is
    returned. Cost is always 0.0 so no test can incur a charge.
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        response_text: str = "mock response",
        response_fn: Optional[Callable[[str], str]] = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.response_text = response_text
        self.response_fn = response_fn
        self.calls: List[str] = []

    def _build(self, prompt: str) -> LLMResponse:
        self.calls.append(prompt)
        text = self.response_fn(prompt) if self.response_fn else self.response_text
        return LLMResponse(
            text=text,
            model=self.model_name,
            tokens_used=len(prompt.split()),
            latency_ms=1.0,
            cost_usd=0.0,
            metadata={"mock": True},
        )

    def generate(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.0, **kwargs
    ) -> LLMResponse:
        return self._build(prompt)

    def batch_generate(self, prompts, max_tokens: int = 500, temperature: float = 0.0, **kwargs):
        return [self._build(p) for p in prompts]


@pytest.fixture
def mock_llm_client():
    """A plain deterministic client."""
    return MockLLMClient()


@pytest.fixture
def make_mock_llm_client():
    """Factory so a test can customise the response behaviour."""

    def _factory(**kwargs) -> MockLLMClient:
        return MockLLMClient(**kwargs)

    return _factory
