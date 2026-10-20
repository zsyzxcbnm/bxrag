"""Deterministic chunking strategies for short policy documents."""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.exceptions import IngestionError
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
    max_words: int = 40,
    overlap_words: int = 10,
) -> list[Chunk]:
    """Split a document with a sliding word window."""

    if max_words <= 0:
        raise IngestionError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise IngestionError("overlap_words must be in [0, max_words)")

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


def sentence_aware_chunks(document: Document, max_chars: int = 360) -> list[Chunk]:
    """Pack complete sentences into chunks without embedding-based segmentation.

    The strategy preserves sentence boundaries and prepends document metadata.
    It does not perform semantic clustering or infer section structure.
    """

    if max_chars <= 0:
        raise IngestionError("max_chars must be positive")

    sentences = [
        part.strip() for part in _SENTENCE_SPLIT_RE.split(document.content) if part.strip()
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
        _make_chunk(document, "sentence", index, f"{header} {group}")
        for index, group in enumerate(groups)
    ]


def semantic_chunks(document: Document, max_chars: int = 360) -> list[Chunk]:
    """Deprecated alias for :func:`sentence_aware_chunks`."""

    return sentence_aware_chunks(document, max_chars=max_chars)


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
        elif strategy in {"sentence", "semantic"}:
            chunks.extend(sentence_aware_chunks(document))
        else:
            raise IngestionError(f"Unknown chunking strategy: {strategy}")
    return chunks
