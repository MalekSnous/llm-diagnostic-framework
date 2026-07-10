"""Paragraph-aware chunking with overlap.

Paragraphs (blank-line separated blocks) are packed into chunks of at most
``max_chars``; a paragraph never gets split unless it alone exceeds the limit.
Consecutive chunks share ``overlap`` trailing characters of context so a fact
sitting on a chunk boundary is still retrievable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .ingestion import Document


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit, traceable back to its source document."""

    chunk_id: str  # "<doc_id>#<n>"
    doc_id: str
    doc_title: str
    text: str


def _paragraphs(text: str) -> List[str]:
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def _split_oversized(paragraph: str, max_chars: int) -> List[str]:
    """Hard-split a paragraph longer than max_chars (rare: tables, long lists)."""
    return [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]


def chunk_document(document: Document, max_chars: int = 800, overlap: int = 120) -> List[Chunk]:
    """Chunk one document into overlapping, paragraph-aligned chunks."""
    pieces: List[str] = []
    for paragraph in _paragraphs(document.text):
        if len(paragraph) > max_chars:
            pieces.extend(_split_oversized(paragraph, max_chars))
        else:
            pieces.append(paragraph)

    chunks: List[Chunk] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and len(candidate) > max_chars:
            chunks.append(_make_chunk(document, len(chunks), current))
            # Carry the tail of the previous chunk as overlap context.
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = candidate
    if current:
        chunks.append(_make_chunk(document, len(chunks), current))
    return chunks


def _make_chunk(document: Document, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id}#{index}",
        doc_id=document.doc_id,
        doc_title=document.title,
        text=text.strip(),
    )


def chunk_documents(
    documents: Iterable[Document], max_chars: int = 800, overlap: int = 120
) -> List[Chunk]:
    """Chunk a whole corpus."""
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars, overlap=overlap))
    return chunks
