"""RRF and document aggregation tests."""

from __future__ import annotations

from src.models import SearchHit
from src.retrieval.fusion import aggregate_doc_hits, rrf_fuse


def _hit(doc_id: str, score: float, rank: int) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        chunk_id=f"{doc_id}::0",
        title=doc_id,
        content=doc_id,
        score=score,
        source="test",
        rank=rank,
    )


def test_aggregate_keeps_best_chunk_per_document() -> None:
    hits = [_hit("a", 0.5, 1), _hit("a", 0.9, 2), _hit("b", 0.7, 3)]
    aggregated = aggregate_doc_hits(hits)
    assert [hit.doc_id for hit in aggregated] == ["a", "b"]
    assert aggregated[0].score == 0.9


def test_rrf_prefers_documents_present_in_both_lists() -> None:
    dense = [_hit("a", 0.9, 1), _hit("b", 0.8, 2)]
    sparse = [_hit("b", 3.0, 1), _hit("c", 2.0, 2)]
    fused = rrf_fuse(dense, sparse, k=60)
    assert [hit.doc_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].source == "rrf"
    assert fused[0].score > fused[1].score

