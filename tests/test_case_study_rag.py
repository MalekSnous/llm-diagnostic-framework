"""Offline tests for the RAG document-QA case study (baseline path only).

The RAG path needs chromadb/sentence-transformers (heavy, optional) and is not
exercised here; we validate the data, test-case construction and the
substring-match scoring used by both baseline and RAG.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent
STUDY = ROOT / "case_studies" / "rag_document_qa" / "run_study.py"


def _load_study():
    spec = importlib.util.spec_from_file_location("rag_study", STUDY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_documents_load():
    study = _load_study()
    docs = study.load_documents()
    assert len(docs) > 1
    assert any("AcmeCloud" in d for d in docs)


def test_create_test_cases():
    study = _load_study()
    cases = study.create_test_cases()
    assert len(cases) == len(study.QA_PAIRS)
    assert all(c.metadata["accepted"] for c in cases)


def test_baseline_scoring(make_mock_llm_client):
    study = _load_study()
    cases = study.create_test_cases()
    # Echo the first accepted answer of every question -> perfect baseline score.
    answers = {c.id: c.metadata["accepted"][0] for c in cases}

    def respond(prompt):
        for c in cases:
            if c.input in prompt:
                return f"The answer is {answers[c.id]}."
        return "unknown"

    client = make_mock_llm_client(response_fn=respond)
    _results, accuracy, cost = study.run_baseline(cases, client)
    assert accuracy == 1.0
    assert cost == 0.0
