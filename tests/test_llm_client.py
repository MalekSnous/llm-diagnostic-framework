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
