"""Offline tests for the text-to-SQL dataset and execution-accuracy scoring."""

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
STUDY = ROOT / "case_studies" / "text_to_sql" / "run_study.py"
TIERS = {"easy", "medium", "hard", "expert"}


def _load_study():
    spec = importlib.util.spec_from_file_location("sql_study", STUDY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_size_and_integrity():
    study = _load_study()
    assert len(study.CASES) >= 90
    for c in study.CASES:
        assert c["question"] and c["gold_sql"], c
        assert c["difficulty"] in TIERS
    assert TIERS.issubset(set(Counter(study.DIFFICULTIES)))


def test_every_gold_query_executes_and_returns_rows():
    study = _load_study()
    for c in study.CASES:
        rows, error = study.execute_sql(c["gold_sql"])
        assert error is None, f"gold query failed: {c['question']!r} -> {error}"
        assert rows, f"gold query returned no rows: {c['question']!r}"
        assert rows != [(None,)], f"gold query returned NULL: {c['question']!r}"


def test_create_test_cases_carries_difficulty_and_schema():
    study = _load_study()
    cases = study.create_test_cases()
    assert len(cases) == len(study.CASES)
    assert all(c.metadata["difficulty"] in TIERS for c in cases)
    # The schema is necessary context and must appear in every prompt.
    assert all("CREATE TABLE customers" in c.input for c in cases)


def test_score_execution_accuracy():
    study = _load_study()
    gold = "SELECT COUNT(*) FROM customers"

    # Semantically equivalent query, different text + fenced output -> full credit.
    equivalent = "```sql\nSELECT COUNT(id) FROM customers;\n```"
    assert study.score(equivalent, gold) == {"accuracy": 1.0, "valid_sql": 1.0}

    # Valid SQL, wrong answer -> valid but inaccurate.
    wrong = "SELECT COUNT(*) FROM orders"
    assert study.score(wrong, gold) == {"accuracy": 0.0, "valid_sql": 1.0}

    # Prose, broken SQL, or non-SELECT statements never crash, never score.
    for bad in ["I don't know.", "SELECT * FROM nope", "DROP TABLE customers", ""]:
        assert study.score(bad, gold) == {"accuracy": 0.0, "valid_sql": 0.0}


def test_score_is_order_insensitive_but_column_strict():
    study = _load_study()
    gold = "SELECT name FROM customers WHERE country = 'France'"
    reordered = "SELECT name FROM customers WHERE country = 'France' ORDER BY name DESC"
    assert study.score(reordered, gold)["accuracy"] == 1.0
    # Extra columns are a real mismatch (the question asked for names only).
    extra = "SELECT name, city FROM customers WHERE country = 'France'"
    assert study.score(extra, gold)["accuracy"] == 0.0


def test_baseline_and_difficulty_breakdown(make_mock_llm_client):
    study = _load_study()
    cases = study.create_test_cases()

    # Echo the gold SQL so scoring is perfect and the breakdown is exercised.
    def respond(prompt):
        for c in study.CASES:
            if c["question"] in prompt:
                return c["gold_sql"]
        return "no idea"

    client = make_mock_llm_client(response_fn=respond)
    results, accuracy, cost = study.run_baseline(cases, client)
    assert accuracy > 0.99
    assert cost == 0.0
    breakdown = study.difficulty_breakdown(results)
    assert TIERS.issubset(set(breakdown))
