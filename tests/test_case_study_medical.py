"""Offline tests for the medical entity-extraction dataset and scoring."""

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
STUDY = ROOT / "case_studies" / "medical_entity_extraction" / "run_study.py"
TIERS = {"easy", "medium", "hard", "expert"}


def _load_study():
    spec = importlib.util.spec_from_file_location("med_study", STUDY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_size_and_integrity():
    study = _load_study()
    assert len(study.CASES) >= 90  # grew from the original 8
    for c in study.CASES:
        assert c["text"] and c["entities"], c
        assert c["difficulty"] in TIERS
    # every tier represented
    assert TIERS.issubset(set(Counter(study.DIFFICULTIES)))


def test_create_test_cases_carries_difficulty():
    study = _load_study()
    cases = study.create_test_cases()
    assert len(cases) == len(study.CASES)
    assert all(c.metadata["difficulty"] in TIERS for c in cases)


def test_baseline_and_difficulty_breakdown(make_mock_llm_client):
    study = _load_study()
    cases = study.create_test_cases()

    # Echo the gold entities so scoring is high and the breakdown is exercised.
    def respond(prompt):
        for c in study.CASES:
            if c["text"] in prompt:
                return ", ".join(c["entities"])
        return "none"

    client = make_mock_llm_client(response_fn=respond)
    results, f1, cost = study.run_baseline(cases, client)
    assert f1 > 0.9
    assert cost == 0.0
    breakdown = study.difficulty_breakdown(results)
    assert TIERS.issubset(set(breakdown))
