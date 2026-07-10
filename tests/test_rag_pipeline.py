"""Offline tests for the modular RAG pipeline (llm_diagnostic.rag).

Everything here runs with the core install only: the TF-IDF embedder
(scikit-learn) and the in-memory vector store need no network, no model
downloads, and no API keys. The dense embedder and cross-encoder reranker
(``rag`` extra) are exercised through the same interfaces via fakes.
"""

from pathlib import Path

import numpy as np
import pytest

from llm_diagnostic.rag import (
    Chunk,
    Document,
    InMemoryVectorStore,
    RAGPipeline,
    TfidfEmbedder,
    chunk_documents,
    get_embedder,
    load_directory,
)
from llm_diagnostic.rag.chunking import chunk_document
from llm_diagnostic.rag.pipeline import ABSTAIN_PHRASE
from llm_diagnostic.rag.reranker import BaseReranker

KB_PATH = Path(__file__).parent.parent / "data" / "acmecloud_kb"


# ---------------------------------------------------------------- ingestion
def test_load_directory_reads_kb():
    docs = load_directory(KB_PATH)
    assert len(docs) >= 10
    ids = [d.doc_id for d in docs]
    assert ids == sorted(ids)  # stable ordering
    assert "plans_and_billing" in ids
    plans = next(d for d in docs if d.doc_id == "plans_and_billing")
    assert plans.title == "Plans and Billing"  # H1 extracted
    assert "ACU" in plans.text


def test_load_directory_missing_path():
    with pytest.raises(FileNotFoundError):
        load_directory(KB_PATH / "does_not_exist")


# ----------------------------------------------------------------- chunking
def test_chunks_respect_max_chars_and_trace_source():
    doc = Document(
        doc_id="d", title="T", text="\n\n".join(f"Paragraph {i}. " * 5 for i in range(20))
    )
    chunks = chunk_document(doc, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 + 50 + 2 for c in chunks)  # content + overlap tail
    assert all(c.doc_id == "d" for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]


def test_chunk_overlap_carries_context():
    doc = Document(doc_id="d", title="T", text="A" * 200 + "\n\n" + "B" * 200)
    chunks = chunk_document(doc, max_chars=250, overlap=40)
    assert len(chunks) == 2
    assert "A" * 40 in chunks[1].text  # tail of chunk 0 repeated


def test_oversized_paragraph_is_split():
    doc = Document(doc_id="d", title="T", text="X" * 2000)
    chunks = chunk_document(doc, max_chars=500, overlap=0)
    assert all(len(c.text) <= 500 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 2000


# ------------------------------------------------------ embeddings + store
def test_tfidf_retrieval_finds_the_right_doc():
    docs = load_directory(KB_PATH)
    chunks = chunk_documents(docs)
    embedder = TfidfEmbedder()
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    store = InMemoryVectorStore()
    store.add(chunks, embedder.embed(texts))

    query_vec = embedder.embed(["How many days until an AKT token expires?"])[0]
    hits = store.search(query_vec, top_k=3)
    assert hits[0][1] >= hits[-1][1]  # sorted by score
    assert any(chunk.doc_id == "authentication" for chunk, _ in hits)


def test_embeddings_are_normalised():
    embedder = TfidfEmbedder()
    embedder.fit(["alpha beta", "gamma delta"])
    matrix = embedder.embed(["alpha beta gamma"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_tfidf_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().embed(["text"])


def test_get_embedder_auto_falls_back_or_is_dense():
    embedder = get_embedder("auto")
    assert embedder.name in ("tfidf", "dense")
    with pytest.raises(ValueError):
        get_embedder("nope")


def test_store_rejects_mismatched_shapes():
    store = InMemoryVectorStore()
    chunk = Chunk(chunk_id="d#0", doc_id="d", doc_title="T", text="x")
    with pytest.raises(ValueError):
        store.add([chunk], np.zeros((2, 3), dtype=np.float32))


# ----------------------------------------------------------------- pipeline
def _indexed_pipeline(**kwargs) -> RAGPipeline:
    pipeline = RAGPipeline(embedder=TfidfEmbedder(), **kwargs)
    pipeline.index(load_directory(KB_PATH))
    return pipeline


def test_pipeline_answer_grounds_prompt_in_retrieved_chunks(make_mock_llm_client):
    pipeline = _indexed_pipeline(top_k=3)
    client = make_mock_llm_client(response_text="14 days [authentication]")
    answer = pipeline.answer("After how many days does a default AKT expire?", client)

    assert answer.text.startswith("14 days")
    assert len(answer.retrieved) == 3
    assert "authentication" in answer.sources
    # The prompt actually contains the retrieved chunks and the abstain contract.
    prompt = client.calls[0]
    assert ABSTAIN_PHRASE in prompt
    for item in answer.retrieved:
        assert item.chunk.text in prompt


class ReverseReranker(BaseReranker):
    """Deterministic fake: reverses first-stage order (worst first)."""

    name = "reverse"

    def rerank(self, query, chunks):
        return list(reversed(chunks))


def test_reranker_changes_final_order():
    plain = _indexed_pipeline(top_k=4)
    reranked = RAGPipeline(
        embedder=plain.embedder, store=plain.store, reranker=ReverseReranker(), top_k=4, fetch_k=8
    )
    question = "What is the maximum webhook payload size?"
    plain_ids = [r.chunk.chunk_id for r in plain.retrieve(question)]
    rerank_ids = [r.chunk.chunk_id for r in reranked.retrieve(question)]
    assert len(rerank_ids) == 4
    assert plain_ids != rerank_ids  # the reranker actually reordered the shortlist
    assert [r.rank for r in reranked.retrieve(question)] == [0, 1, 2, 3]
