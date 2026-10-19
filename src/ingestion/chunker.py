"""Deterministic chunking strategies for short policy documents."""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.models import Chunk, Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _make_chunk(document: Document, strategy: str, index: int, text: str) -> Chunk:
    clean_text = " ".join(text.split())
    return Chunk(
        chunk_id=f"{document.doc_id}::{strategy}::{index:03d}",
        doc_id=document.doc_id,
        chunk_index=index,
        strategy=strategy,
        text=clean_text,
        token_count=len(clean_text.split()),
    )


def document_chunks(document: Document) -> list[Chunk]:
    """Treat each already-short policy row as one atomic chunk."""

    return [_make_chunk(document, "document", 0, f"{document.title}. {document.content}")]


def fixed_window_chunks(
    document: Document,
    max_words: int = 110,
    overlap_words: int = 20,
) -> list[Chunk]:
    """Split a document with a sliding word window."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be in [0, max_words)")

    words = f"{document.title}. {document.content}".split()
    chunks: list[Chunk] = []
    step = max_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        chunks.append(_make_chunk(document, "fixed", len(chunks), " ".join(window)))
        if start + max_words >= len(words):
            break
    return chunks


def semantic_chunks(document: Document, max_chars: int = 360) -> list[Chunk]:
    """Group sentences into small, context-preserving semantic units."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    sentences = [
        part.strip()
        for part in _SENTENCE_SPLIT_RE.split(document.content)
        if part.strip()
    ]
    groups: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        if current and current_length + len(sentence) + 1 > max_chars:
            groups.append(" ".join(current))
            current = []
            current_length = 0
        current.append(sentence)
        current_length += len(sentence) + 1
    if current:
        groups.append(" ".join(current))

    header = f"{document.title}. Product: {document.card_tier}. Year: {document.effective_year}."
    return [
        _make_chunk(document, "semantic", index, f"{header} {group}")
        for index, group in enumerate(groups)
    ]


def chunk_documents(
    documents: Iterable[Document],
    strategy: str = "document",
) -> list[Chunk]:
    """Apply one chunking strategy to every document."""

    chunks: list[Chunk] = []
    for document in documents:
        if strategy == "document":
            chunks.extend(document_chunks(document))
        elif strategy == "fixed":
            chunks.extend(fixed_window_chunks(document))
        elif strategy == "semantic":
            chunks.extend(semantic_chunks(document))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    return chunks

