"""Tests for improvement strategies (offline, mocked)."""

from llm_diagnostic.core.llm_client import LLMResponse
from llm_diagnostic.failure_tests.base_test import TestCase, TestResult
from llm_diagnostic.improvements.prompt_engineering import PromptEngineeringStrategy


def _baseline_results(test_cases):
    return [
        TestResult(
            test_case_id=tc.id,
            prediction="",
            reference=tc.expected_output,
            success=False,
            metrics={"accuracy": 0.0},
            response=LLMResponse("", "mock", tokens_used=1, latency_ms=1.0, cost_usd=0.0),
        )
        for tc in test_cases
    ]


def test_prompt_engineering_configure():
    strategy = PromptEngineeringStrategy()
    config = strategy.configure(technique="chain-of-thought")
    assert config.parameters["technique"] == "chain-of-thought"


def test_prompt_engineering_apply_echo(make_mock_llm_client):
    """When the model echoes the expected answer, accuracy should be perfect."""
    test_cases = [
        TestCase(id="t0", input="What is 2+2?", expected_output="4"),
        TestCase(id="t1", input="Capital of France?", expected_output="Paris"),
    ]
    # Echo the expected output so the substring-match success criterion passes.
    client = make_mock_llm_client(response_fn=lambda prompt: "4 Paris")
    strategy = PromptEngineeringStrategy()
    config = strategy.configure(technique="few-shot")

    results = strategy.apply(
        test_cases=test_cases,
        baseline_results=_baseline_results(test_cases),
        config=config,
        llm_client=client,
    )

    assert len(results) == 2
    assert all(r.metrics["accuracy"] == 1.0 for r in results)


def test_prompt_engineering_task_specific_examples(make_mock_llm_client):
    """configure(examples=...) must inject the caller's examples into the prompt."""
    examples = [
        "Example question: q1\nExample answer: a1",
        "Example question: q2\nExample answer: a2",
    ]
    test_cases = [TestCase(id="t0", input="Translate this.", expected_output="ok")]
    client = make_mock_llm_client(response_fn=lambda p: "ok")

    strategy = PromptEngineeringStrategy()
    config = strategy.configure(technique="few-shot", examples=examples, num_examples=2)
    strategy.apply(test_cases, _baseline_results(test_cases), config, client)

    prompt_sent = client.calls[0]
    assert "Example question: q1" in prompt_sent
    assert "Example question: q2" in prompt_sent
    assert "Translate this." in prompt_sent
    # The built-in medical examples must NOT leak into a custom-example task.
    assert "Patient with diabetes" not in prompt_sent


def test_cost_benefit_analysis_runs(make_mock_llm_client):
    test_cases = [TestCase(id="t0", input="q", expected_output="a")]
    client = make_mock_llm_client(response_fn=lambda p: "a")
    strategy = PromptEngineeringStrategy()
    config = strategy.configure()
    strategy.apply(test_cases, _baseline_results(test_cases), config, client)
    # A result was recorded for later cost/benefit comparison.
    assert len(strategy.results) == 1
    assert strategy.results[0].total_cost == 0.0
