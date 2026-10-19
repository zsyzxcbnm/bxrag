"""Rank fusion and document-level aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from src.models import SearchHit


def aggregate_doc_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Collapse multiple chunks to one document, keeping its best chunk."""

    best: dict[str, SearchHit] = {}
    for hit in hits:
        previous = best.get(hit.doc_id)
        if previous is None or hit.score > previous.score:
            best[hit.doc_id] = hit
    ordered = sorted(best.values(), key=lambda item: item.score, reverse=True)
    return [replace(hit, rank=index + 1) for index, hit in enumerate(ordered)]


def rrf_fuse(
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[SearchHit]:
    """Fuse two ranked lists with Reciprocal Rank Fusion."""

    if k <= 0:
        raise ValueError("RRF k must be positive")
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("RRF weights must be non-negative")
    if dense_weight == 0 and sparse_weight == 0:
        raise ValueError("At least one RRF weight must be positive")

    scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, SearchHit] = {}
    for source_hits, weight in ((dense_hits, dense_weight), (sparse_hits, sparse_weight)):
        for rank, hit in enumerate(aggregate_doc_hits(source_hits), start=1):
            scores[hit.doc_id] += weight / (k + rank)
            representatives.setdefault(hit.doc_id, hit)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[SearchHit] = []
    for index, (doc_id, score) in enumerate(ranked, start=1):
        hit = representatives[doc_id]
        fused.append(replace(hit, score=score, source="rrf", rank=index))
    return fused
