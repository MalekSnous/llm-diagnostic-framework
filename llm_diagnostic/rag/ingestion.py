"""Document ingestion: load a corpus of text/markdown files from disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

TEXT_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    """One source document of the knowledge base."""

    doc_id: str  # stable identifier, e.g. the file stem ("plans_and_billing")
    title: str
    text: str


def _title_of(text: str, fallback: str) -> str:
    """First markdown H1 if present, else the fallback (file stem)."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_directory(path: Union[str, Path]) -> List[Document]:
    """Load every .md/.txt file under ``path`` (non-recursive) as a Document.

    Files are sorted by name so doc_ids and downstream chunk ids are stable
    across runs.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"knowledge-base directory not found: {directory}")

    documents = []
    for file in sorted(directory.iterdir()):
        if file.suffix.lower() not in TEXT_SUFFIXES or not file.is_file():
            continue
        text = file.read_text(encoding="utf-8").strip()
        if text:
            documents.append(
                Document(doc_id=file.stem, title=_title_of(text, file.stem), text=text)
            )
    if not documents:
        raise FileNotFoundError(f"no .md/.txt documents in {directory}")
    return documents
