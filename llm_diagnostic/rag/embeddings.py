"""Embedding backends behind one interface.

``TfidfEmbedder`` (scikit-learn, a core dependency) keeps the pipeline fully
offline-runnable; ``SentenceTransformerEmbedder`` (the ``rag`` extra) is the
quality option. ``get_embedder("auto")`` picks dense when installed and falls
back to TF-IDF otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class BaseEmbedder(ABC):
    """Maps texts to L2-normalised vectors; cosine similarity == dot product."""

    name: str = "base"

    def fit(self, corpus: List[str]) -> None:  # noqa: B027 — optional hook, no-op by default
        """Fit on the corpus before indexing. No-op for pretrained models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) float array of normalised embeddings."""


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(matrix / norms)


class TfidfEmbedder(BaseEmbedder):
    """Lexical embeddings from a TF-IDF vectorizer fitted on the corpus."""

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.embed() called before fit()")
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return _normalise(matrix)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense embeddings via sentence-transformers (requires the ``rag`` extra)."""

    name = "dense"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self.model_name = model_name
        self._model: Any = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalise(np.asarray(vectors, dtype=np.float32))


def get_embedder(kind: str = "auto") -> BaseEmbedder:
    """Build an embedder: "tfidf", "dense", or "auto" (dense with fallback)."""
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind == "dense":
        return SentenceTransformerEmbedder()
    if kind == "auto":
        try:
            return SentenceTransformerEmbedder()
        except ImportError:
            return TfidfEmbedder()
    raise ValueError(f"unknown embedder kind: {kind!r} (expected auto|dense|tfidf)")
