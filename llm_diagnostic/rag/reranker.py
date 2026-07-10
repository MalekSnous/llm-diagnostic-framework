"""Second-stage reranking of retrieved chunks.

A cross-encoder scores each (query, chunk) pair jointly, which is far more
precise than the bi-encoder similarity used for first-stage retrieval — at the
price of one forward pass per candidate, so it only runs on the shortlist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from .chunking import Chunk


class BaseReranker(ABC):
    """Reorders candidate chunks by relevance to the query."""

    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """Return the chunks sorted most-relevant first."""


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker via sentence-transformers (``rag`` extra)."""

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

        self.model_name = model_name
        self._model: Any = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, chunk.text) for chunk in chunks])
        order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
        return [chunks[i] for i in order]
