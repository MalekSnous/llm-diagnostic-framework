"""Modular RAG pipeline: ingestion → chunking → embedding → retrieval → reranking → answer.

Each stage is a small, swappable component so the case studies can ablate them
independently (e.g. TF-IDF vs dense embeddings, with or without reranking).
Heavy dependencies (sentence-transformers, chromadb) are imported lazily inside
the components that need them; the TF-IDF + in-memory path runs with the core
install only.
"""

from .chunking import Chunk, chunk_documents
from .embeddings import BaseEmbedder, TfidfEmbedder, get_embedder
from .ingestion import Document, load_directory
from .pipeline import RAGAnswer, RAGPipeline, RetrievedChunk
from .reranker import BaseReranker, CrossEncoderReranker
from .vector_store import BaseVectorStore, InMemoryVectorStore

__all__ = [
    "BaseEmbedder",
    "BaseReranker",
    "BaseVectorStore",
    "Chunk",
    "CrossEncoderReranker",
    "Document",
    "InMemoryVectorStore",
    "RAGAnswer",
    "RAGPipeline",
    "RetrievedChunk",
    "TfidfEmbedder",
    "chunk_documents",
    "get_embedder",
    "load_directory",
]
