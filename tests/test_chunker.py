"""Chunking strategy tests."""

from __future__ import annotations

from src.ingestion.chunker import chunk_documents, document_chunks, fixed_window_chunks
from src.ingestion.normalizer import normalize_document


def _document():
    return normalize_document(
        {
            "doc_id": "DOC-X",
            "title": "Apex Platinum - Test Policy (2024)",
            "category": "Test",
            "card_tier": "Apex Platinum",
            "effective_year": "2024",
            "content": (
                "Policy Code: TST-001-APEX-2024. Applicable Product: Apex Platinum. "
                "Details: " + " ".join(["word"] * 300)
            ),
        }
    )


def test_document_strategy_produces_one_chunk() -> None:
    chunks = document_chunks(_document())
    assert len(chunks) == 1
    assert chunks[0].chunk_id.startswith("DOC-X::document::")


def test_fixed_window_produces_multiple_chunks() -> None:
    chunks = fixed_window_chunks(_document(), max_words=80, overlap_words=10)
    assert len(chunks) > 1
    assert all(chunk.strategy == "fixed" for chunk in chunks)


def test_all_strategies_dispatch() -> None:
    chunks = []
    for strategy in ("document", "fixed", "semantic"):
        chunks.extend(chunk_documents([_document()], strategy))
    assert {chunk.strategy for chunk in chunks} == {"document", "fixed", "semantic"}

