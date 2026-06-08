"""Tests for the metric calculations in ``llm_diagnostic.core.evaluator``."""

from llm_diagnostic.core.evaluator import EvaluationMetrics, Evaluator
from llm_diagnostic.core.llm_client import LLMResponse


def test_exact_match():
    assert Evaluator.exact_match(["a", "b"], ["a", "b"]) == 1.0
    assert Evaluator.exact_match(["a", "x"], ["a", "b"]) == 0.5


def test_classification_metrics():
    metrics = Evaluator.classification_metrics(["cat", "dog", "cat"], ["cat", "dog", "dog"])
    assert metrics.metrics["accuracy"] == 2 / 3
    assert "f1" in metrics.metrics


def test_entity_extraction_metrics():
    metrics = Evaluator.entity_extraction_metrics(
        predictions=[["a", "b"]],
        references=[["a", "c"]],
    )
    # 1 TP (a), 1 FP (b), 1 FN (c) -> precision = recall = 0.5
    assert metrics.metrics["precision"] == 0.5
    assert metrics.metrics["recall"] == 0.5
    assert metrics.metrics["f1"] == 0.5


def test_structure_validation_metrics_uses_add_details():
    """Regression test: add_details() must exist (was missing -> AttributeError)."""
    metrics = Evaluator.structure_validation_metrics(
        ['{"ok": 1}', "not json"], expected_format="json"
    )
    assert metrics.metrics["parse_success_rate"] == 0.5
    assert metrics.metrics["valid_outputs"] == 1
    assert "parse_errors" in metrics.details
    assert len(metrics.details["parse_errors"]) == 1


def test_evaluation_metrics_add_details():
    m = EvaluationMetrics()
    m.add_details("notes", {"k": "v"})
    assert m.details["notes"] == {"k": "v"}
    assert m.to_dict()["details"]["notes"] == {"k": "v"}


def test_parse_entity_list_drops_prose():
    out = Evaluator.parse_entity_list("diabetes, hypertension and metformin")
    assert "diabetes" in out and "hypertension" in out and "metformin" in out
    # A long prose sentence should not be kept as an entity.
    prose = Evaluator.parse_entity_list("The patient was a brilliant mind pushing many boundaries")
    assert prose == []


def test_fuzzy_entity_metrics_containment():
    m = Evaluator.fuzzy_entity_metrics(["type 2 diabetes", "metformin"], ["diabetes", "metformin"])
    assert m.metrics["recall"] == 1.0
    assert m.metrics["precision"] == 1.0
    assert m.metrics["f1"] == 1.0


def test_fuzzy_entity_metrics_penalises_verbosity():
    """A verbose dump that happens to contain the answers must NOT score 100%.

    This is the Phi-2 artifact: recall-only substring matching gave 100%, but
    precision collapses once spurious candidates are counted.
    """
    reference = ["diabetes", "metformin"]
    verbose = Evaluator.parse_entity_list(
        "diabetes, metformin, aspirin, surgery, cancer, fever, headache, insulin, asthma"
    )
    m = Evaluator.fuzzy_entity_metrics(verbose, reference)
    assert m.metrics["recall"] == 1.0  # both answers present
    assert m.metrics["precision"] < 0.5  # but lots of junk
    assert m.metrics["f1"] < 0.7  # so F1 is not inflated


def test_cost_analysis():
    responses = [
        LLMResponse("a", "m", tokens_used=10, latency_ms=5.0, cost_usd=0.01),
        LLMResponse("b", "m", tokens_used=20, latency_ms=15.0, cost_usd=0.02),
    ]
    metrics = Evaluator.cost_analysis(responses)
    assert metrics.metrics["total_cost_usd"] == 0.03
    assert metrics.metrics["avg_latency_ms"] == 10.0
