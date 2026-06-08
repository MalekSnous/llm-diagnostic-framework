"""Tests for the LLM client factory and routing (no real API calls)."""

import sys
import types

import pytest

from llm_diagnostic.core.llm_client import get_llm_client


@pytest.fixture(autouse=True)
def _fake_openai(monkeypatch):
    """Stub the openai SDK so OpenAIClient can be constructed offline."""
    fake = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    fake.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    yield


def test_get_client_routes_openai():
    from llm_diagnostic.core.llm_client import OpenAIClient

    client = get_llm_client("gpt-4o-mini")
    assert isinstance(client, OpenAIClient)
    assert client.model_name == "gpt-4o-mini"


def test_openai_pricing_table_present():
    client = get_llm_client("gpt-4o-mini")
    assert "gpt-4o-mini" in client.pricing
    assert "input" in client.pricing["gpt-4o-mini"]


def test_openai_cost_is_per_1k_not_per_1m():
    """Regression: pricing must be per-1K, so costs aren't inflated ~1000x.

    gpt-4o is $2.50/$10.00 per 1M tokens. For 1K in + 1K out the cost must be
    ~$0.0125, not ~$12.50.
    """
    client = get_llm_client("gpt-4o")
    cost = client._calculate_cost(1000, 1000)
    assert abs(cost - 0.0125) < 1e-9
    # Sanity bound: 1K+1K tokens on any current model is well under $1.
    assert cost < 1.0
