"""End-to-end RAG pipeline: index a corpus, retrieve, (optionally) rerank, answer.

The pipeline is deliberately explicit about its stages so a case study can
report on each one (which chunks were retrieved, in what order, at what cost)
instead of treating RAG as a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.llm_client import BaseLLMClient, LLMResponse
from .chunking import Chunk, chunk_documents
from .embeddings import BaseEmbedder
from .ingestion import Document
from .reranker import BaseReranker
from .vector_store import BaseVectorStore, InMemoryVectorStore

ABSTAIN_PHRASE = "Not in the documentation"

ANSWER_PROMPT = """You are a support assistant. Answer the question using ONLY the documentation excerpts below.
Answer concisely (one short sentence) and cite the source id in brackets, e.g. [plans_and_billing].
If the excerpts do not contain the answer, reply exactly: {abstain}

Documentation excerpts:
{context}

Question: {question}

Answer:"""


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk as returned to the generator, with its retrieval provenance."""

    chunk: Chunk
    score: float  # first-stage cosine similarity
    rank: int  # final rank fed to the prompt (0 = most relevant)


@dataclass
class RAGAnswer:
    """The generated answer plus everything needed to audit it."""

    text: str
    retrieved: List[RetrievedChunk]
    prompt: str
    response: Optional[LLMResponse] = None

    @property
    def sources(self) -> List[str]:
        """doc_ids of the retrieved chunks, deduplicated, best first."""
        seen: List[str] = []
        for item in self.retrieved:
            if item.chunk.doc_id not in seen:
                seen.append(item.chunk.doc_id)
        return seen


@dataclass
class RAGPipeline:
    """Query pipeline over an indexed corpus.

    ``top_k`` chunks reach the prompt. With a reranker, ``fetch_k`` candidates
    (default 3×top_k) are retrieved first and the reranker picks the best
    ``top_k``; without one, first-stage order is used directly.
    """

    embedder: BaseEmbedder
    store: BaseVectorStore = field(default_factory=InMemoryVectorStore)
    reranker: Optional[BaseReranker] = None
    top_k: int = 4
    fetch_k: Optional[int] = None

    def index(self, documents: List[Document], max_chars: int = 800, overlap: int = 120) -> int:
        """Chunk + embed + store the corpus. Returns the number of chunks."""
        chunks = chunk_documents(documents, max_chars=max_chars, overlap=overlap)
        texts = [chunk.text for chunk in chunks]
        self.embedder.fit(texts)
        self.store.add(chunks, self.embedder.embed(texts))
        return len(chunks)

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """First-stage retrieval, then optional reranking, down to top_k."""
        n_candidates = self.fetch_k or (3 * self.top_k if self.reranker else self.top_k)
        query_vec = self.embedder.embed([question])[0]
        hits = self.store.search(query_vec, top_k=n_candidates)
        scores = {hit[0].chunk_id: hit[1] for hit in hits}

        candidates = [hit[0] for hit in hits]
        if self.reranker is not None:
            candidates = self.reranker.rerank(question, candidates)
        return [
            RetrievedChunk(chunk=chunk, score=scores[chunk.chunk_id], rank=rank)
            for rank, chunk in enumerate(candidates[: self.top_k])
        ]

    def build_prompt(self, question: str, retrieved: List[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"[source: {item.chunk.doc_id}]\n{item.chunk.text}" for item in retrieved
        )
        return ANSWER_PROMPT.format(abstain=ABSTAIN_PHRASE, context=context, question=question)

    def answer(self, question: str, llm_client: BaseLLMClient, max_tokens: int = 150) -> RAGAnswer:
        """Full query path: retrieve → rerank → prompt → generate."""
        retrieved = self.retrieve(question)
        prompt = self.build_prompt(question, retrieved)
        response = llm_client.generate(prompt, max_tokens=max_tokens)
        return RAGAnswer(text=response.text, retrieved=retrieved, prompt=prompt, response=response)
