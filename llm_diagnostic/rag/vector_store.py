"""Vector store interface + an exact in-memory implementation.

For corpora up to tens of thousands of chunks, exact cosine search over a numpy
matrix is faster and simpler than an ANN service. Swap in ChromaDB/FAISS behind
the same interface when the corpus outgrows memory (see the ``rag`` extra).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from .chunking import Chunk


class BaseVectorStore(ABC):
    """Stores chunk embeddings and answers nearest-neighbour queries."""

    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Index chunks with their (n_chunks, dim) embedding matrix."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, cosine_similarity) pairs, best first."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of indexed chunks."""


class InMemoryVectorStore(BaseVectorStore):
    """Exact cosine search over a dense matrix (embeddings are pre-normalised)."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {embeddings.shape[0]} embeddings")
        self._chunks.extend(chunks)
        matrix = embeddings.astype(np.float32)
        self._matrix = matrix if self._matrix is None else np.vstack([self._matrix, matrix])

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            return []
        scores = self._matrix @ query_embedding.astype(np.float32).ravel()
        k = min(top_k, len(self._chunks))
        best = np.argsort(scores)[::-1][:k]
        return [(self._chunks[i], float(scores[i])) for i in best]

    def __len__(self) -> int:
        return len(self._chunks)
