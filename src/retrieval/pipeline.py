"""End-to-end retrieval pipeline with optional metadata filters."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from src.config import Settings
from src.db_protocol import DatabaseLike
from src.exceptions import RetrievalError
from src.ingestion.embedder import Embedder
from src.models import ParsedQuery, RetrievalResult, SearchHit
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import aggregate_doc_hits, rrf_fuse
from src.retrieval.query import parse_query
from src.retrieval.rerank import Reranker, build_reranker
from src.retrieval.sparse import SparseRetriever

RetrievalMode = Literal["dense_only", "sparse_only", "hybrid", "hybrid_rerank"]


class RetrievalPipeline:
    """Compose query parsing, dense/sparse retrieval, RRF and reranking."""

    def __init__(
        self,
        database: DatabaseLike,
        embedder: Embedder,
        settings: Settings,
        reranker: Reranker | None = None,
    ) -> None:
        self.database = database
        self.embedder = embedder
        self.settings = settings
        self.dense = DenseRetriever(database)
        self.sparse = SparseRetriever(database)
        self.reranker = reranker or build_reranker(settings)
        self._query_cache: dict[str, np.ndarray] = {}

    def clear_query_cache(self) -> None:
        """Clear cached query vectors before a fresh latency measurement."""

        self._query_cache.clear()

    @staticmethod
    def _filters(parsed: ParsedQuery) -> dict[str, str]:
        filters: dict[str, str] = {}
        if parsed.card_tier:
            filters["card_tier"] = parsed.card_tier
        if parsed.effective_year:
            filters["effective_year"] = parsed.effective_year
        return filters

    def _dense_with_fallback(
        self,
        query_vector: np.ndarray,
        strategy: str,
        filters: dict[str, str],
    ) -> tuple[list[SearchHit], float, bool]:
        hits, elapsed = self.dense.search(
            query_vector,
            self.settings.dense_limit,
            strategy,
            filters or None,
        )
        if filters and len(aggregate_doc_hits(hits)) < 3:
            fallback_hits, fallback_elapsed = self.dense.search(
                query_vector,
                self.settings.dense_limit,
                strategy,
                None,
            )
            if len(fallback_hits) > len(hits):
                return fallback_hits, elapsed + fallback_elapsed, True
        return hits, elapsed, False

    def _sparse_with_fallback(
        self,
        query: str,
        strategy: str,
        filters: dict[str, str],
    ) -> tuple[list[SearchHit], float, bool]:
        hits, elapsed = self.sparse.search(
            query,
            self.settings.sparse_limit,
            strategy,
            filters or None,
        )
        if filters and len(aggregate_doc_hits(hits)) < 3:
            fallback_hits, fallback_elapsed = self.sparse.search(
                query,
                self.settings.sparse_limit,
                strategy,
                None,
            )
            if len(fallback_hits) > len(hits):
                return fallback_hits, elapsed + fallback_elapsed, True
        return hits, elapsed, False

    def search(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        chunk_strategy: str = "document",
        use_metadata: bool = False,
    ) -> RetrievalResult:
        """Run one of four retrieval modes and return ranked document hits."""

        if mode not in {"dense_only", "sparse_only", "hybrid", "hybrid_rerank"}:
            raise RetrievalError(f"Unknown retrieval mode: {mode}")

        started = time.perf_counter()
        parsed = parse_query(query)
        filters = self._filters(parsed) if use_metadata else {}
        timings: dict[str, float] = {}
        limit = top_k if top_k is not None else self.settings.final_k

        query_vector: np.ndarray | None = None
        if mode in {"dense_only", "hybrid", "hybrid_rerank"}:
            embed_started = time.perf_counter()
            query_vector = self._query_cache.get(query)
            if query_vector is None:
                query_vector = self.embedder.embed_query(query)
                self._query_cache[query] = query_vector
            timings["embed_ms"] = (time.perf_counter() - embed_started) * 1000
        else:
            timings["embed_ms"] = 0.0

        dense_hits: list[SearchHit] = []
        sparse_hits: list[SearchHit] = []
        needs_dense = mode in {"dense_only", "hybrid", "hybrid_rerank"}
        needs_sparse = mode in {"sparse_only", "hybrid", "hybrid_rerank"}
        if needs_dense:
            if query_vector is None:
                raise RetrievalError("Dense retrieval requires a query vector")
            dense_hits, elapsed, _ = self._dense_with_fallback(
                query_vector, chunk_strategy, filters
            )
            timings["dense_ms"] = elapsed
        if needs_sparse:
            sparse_hits, elapsed, _ = self._sparse_with_fallback(query, chunk_strategy, filters)
            timings["sparse_ms"] = elapsed

        if mode == "dense_only":
            hits = aggregate_doc_hits(dense_hits)[:limit]
        elif mode == "sparse_only":
            hits = aggregate_doc_hits(sparse_hits)[:limit]
        else:
            fusion_started = time.perf_counter()
            hits = rrf_fuse(
                dense_hits,
                sparse_hits,
                k=self.settings.rrf_k,
                dense_weight=self.settings.dense_weight,
                sparse_weight=self.settings.sparse_weight,
            )
            timings["fusion_ms"] = (time.perf_counter() - fusion_started) * 1000
            if mode == "hybrid":
                hits = hits[:limit]
            else:
                rerank_started = time.perf_counter()
                hits = self.reranker.rerank(
                    query,
                    parsed,
                    hits[: self.settings.rerank_candidates],
                    top_n=limit,
                    use_metadata=use_metadata,
                )
                timings["rerank_ms"] = (time.perf_counter() - rerank_started) * 1000

        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return RetrievalResult(hits=hits, parsed_query=parsed, mode=mode, timings_ms=timings)
