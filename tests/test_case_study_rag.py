"""Offline tests for the RAG document-QA case study.

Covers dataset integrity (every answerable, non-derived answer really occurs in
its gold source documents — the RAG analogue of the SQL study executing its
gold queries), the boundary-aware scoring, and both the baseline and RAG arms
driven by the mock LLM client. No API keys, no heavy models: the RAG arm runs
on the TF-IDF embedder.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
STUDY_DIR = ROOT / "case_studies" / "rag_document_qa"
KB_PATH = ROOT / "data" / "acmecloud_kb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def study():
    return _load("rag_study_under_test", STUDY_DIR / "run_study.py")


@pytest.fixture(scope="module")
def dataset():
    return _load("rag_dataset_under_test", STUDY_DIR / "dataset.py")


# ------------------------------------------------------------------ dataset
def test_dataset_shape(dataset):
    assert len(dataset.CASES) == 100
    for tier in dataset.TIERS:
        count = sum(1 for c in dataset.CASES if c["difficulty"] == tier)
        assert count == 25, f"{tier}: {count} cases"


def test_unanswerable_cases_are_expert_only(dataset):
    unanswerable = [c for c in dataset.CASES if c["unanswerable"]]
    assert len(unanswerable) == 8
    assert all(c["difficulty"] == "expert" for c in unanswerable)
    assert all(not c["sources"] for c in unanswerable)
    assert all(c["accepted"] == dataset.ABSTAIN_ANSWERS for c in unanswerable)


def test_gold_sources_exist_on_disk(dataset):
    for case in dataset.CASES:
        for doc_id in case["sources"]:
            assert (KB_PATH / f"{doc_id}.md").is_file(), f"missing gold doc: {doc_id}"
        if not case["unanswerable"]:
            assert case["sources"], f"answerable case without sources: {case['question']}"


def test_answers_are_grounded_in_gold_docs(study, dataset):
    """Every answerable, non-derived accepted answer occurs in a gold source doc.

    Uses the study's own matcher, so what the scorer accepts is exactly what
    the corpus must contain — the dataset cannot drift from the docs.
    """
    for case in dataset.CASES:
        if case["unanswerable"] or case["derived"]:
            continue
        gold_text = " ".join(
            (KB_PATH / f"{d}.md").read_text(encoding="utf-8") for d in case["sources"]
        )
        assert study.is_correct(gold_text, case), f"answer not in gold docs: {case['question']}"


# ------------------------------------------------------------------ scoring
def test_numeric_matching_uses_digit_boundaries(study):
    case = {"accepted": ["50"], "unanswerable": False}
    assert study.is_correct("The Growth plan allows 50 concurrent pipelines.", case)
    assert study.is_correct("It allows 50.", case)
    assert not study.is_correct("It includes 500 free ACUs.", case)
    assert not study.is_correct("The limit is 3.50 today.", case)
    assert not study.is_correct("About 1,050 total.", case)


def test_decimal_matching_is_exact(study):
    case = {"accepted": ["99.9"], "unanswerable": False}
    assert study.is_correct("The SLA is 99.9%.", case)
    assert not study.is_correct("The SLA is 99.95%.", case)


def test_currency_and_text_matching(study):
    assert study.is_correct("Each extra ACU costs $0.08.", {"accepted": ["0.08"]})
    assert study.is_correct(
        "Rotate it with `acme auth rotate`.", {"accepted": ["acme auth rotate"]}
    )
    assert not study.is_correct("I have no idea.", {"accepted": ["acme auth rotate"]})


def test_abstention_scoring(study, dataset):
    unanswerable = next(c for c in dataset.CASES if c["unanswerable"])
    assert study.is_correct("Not in the documentation", unanswerable)
    assert study.is_correct("Sorry, I don't know.", unanswerable)
    assert not study.is_correct("The limit is 100 Flows per project.", unanswerable)


# ----------------------------------------------------------------- sampling
def test_sample_keeps_every_tier(study):
    sampled = study._sample(study.CASES, 20)
    assert len(sampled) == 20
    assert {c["difficulty"] for c in sampled} == set(study.TIERS)
    assert study._sample(study.CASES, 0) == study.CASES


# ------------------------------------------------------------------- arms
def _echo_client(make_mock_llm_client, cases):
    """Mock client that answers each question with its first accepted answer."""

    def respond(prompt):
        for case in cases:
            if case["question"] in prompt:
                return f"The answer is {case['accepted'][0]}."
        return "unknown"

    return make_mock_llm_client(response_fn=respond)


def test_baseline_arm_scores_echoed_answers(study, make_mock_llm_client):
    cases = study._sample(study.CASES, 12)
    test_cases = study.create_test_cases(cases)
    client = _echo_client(make_mock_llm_client, cases)
    results = study.run_baseline(test_cases, cases, client)
    assert len(results) == 12
    assert all(r.metrics["accuracy"] == 1.0 for r in results)
    breakdown = study.difficulty_breakdown(results, cases)
    assert set(breakdown) == set(study.TIERS)


def test_rag_arm_runs_offline_with_tfidf(study, make_mock_llm_client):
    from llm_diagnostic.rag import RAGPipeline, TfidfEmbedder, load_directory

    cases = study._sample(study.CASES, 8)
    test_cases = study.create_test_cases(cases)
    pipeline = RAGPipeline(embedder=TfidfEmbedder(), top_k=4)
    pipeline.index(load_directory(KB_PATH))

    client = _echo_client(make_mock_llm_client, cases)
    results, recall = study.run_rag_arm("RAG", pipeline, test_cases, cases, client)
    assert all(r.metrics["accuracy"] == 1.0 for r in results)
    assert 0.0 <= recall <= 1.0
    answerable = [r for r, c in zip(results, cases) if c["sources"]]
    assert all("retrieval_recall" in r.metrics for r in answerable)
