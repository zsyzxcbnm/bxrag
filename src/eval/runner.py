"""Run reproducible baseline and enhanced retrieval experiments."""

from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import Settings
from src.eval.metrics import evaluate_ranking, first_relevant_rank
from src.models import QueryExample
from src.retrieval.pipeline import RetrievalMode, RetrievalPipeline

MODES: tuple[RetrievalMode, ...] = ("dense_only", "sparse_only", "hybrid", "hybrid_rerank")
EVAL_TOP_K = 10


def percentile(values: list[float], percentile_value: float) -> float:
    """Linear-interpolation percentile with no external dependency."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class EvalRunner:
    """Execute the four retrieval ablations and aggregate metrics."""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        settings: Settings,
        repeats: int = 3,
        backend: str | None = None,
        top_k: int = EVAL_TOP_K,
        use_metadata: bool = False,
    ) -> None:
        if top_k < 10:
            raise ValueError("top_k must be at least 10 to compute Recall@10 and nDCG@10")
        self.pipeline = pipeline
        self.settings = settings
        self.repeats = max(1, repeats)
        self.backend = backend or settings.backend
        self.top_k = top_k
        self.use_metadata = use_metadata

    def run(self, queries: list[QueryExample]) -> dict[str, Any]:
        """Run all modes and return a JSON-serializable result payload."""

        self.pipeline.embedder.embed_query("warmup retrieval query")
        payload: dict[str, Any] = {
            "metadata": {
                "generated_at": datetime.now(UTC).isoformat(),
                "query_count": len(queries),
                "backend": self.backend,
                "embedding_provider": self.settings.embedding_provider,
                "embedding_model": self.pipeline.embedder.name,
                "embedding_dim": self.pipeline.embedder.dim,
                "embedding_device": getattr(self.pipeline.embedder, "device", "n/a"),
                "reranker": self.pipeline.reranker.name,
                "rerank_candidates": self.settings.rerank_candidates,
                "rrf_k": self.settings.rrf_k,
                "dense_weight": self.settings.dense_weight,
                "sparse_weight": self.settings.sparse_weight,
                "top_k": self.top_k,
                "use_metadata": self.use_metadata,
            },
            "summary": {},
            "per_query": {},
        }

        for mode in MODES:
            self.pipeline.clear_query_cache()
            mode_metrics: dict[str, list[float]] = {}
            retrieval_latencies: list[float] = []
            embedding_latencies: list[float] = []
            stage_latencies: dict[str, list[float]] = {}
            per_query: dict[str, Any] = {}

            for query in queries:
                first_result = self.pipeline.search(
                    query.query,
                    mode=mode,
                    top_k=self.top_k,
                    use_metadata=self.use_metadata,
                )
                ranked_ids = [hit.doc_id for hit in first_result.hits]
                metrics = evaluate_ranking(ranked_ids, query.relevant_doc_ids)
                for name, value in metrics.items():
                    mode_metrics.setdefault(name, []).append(value)

                query_latencies = [first_result.timings_ms["total_ms"]]
                query_retrieval = [
                    sum(
                        value
                        for stage, value in first_result.timings_ms.items()
                        if stage not in {"embed_ms", "total_ms"}
                    )
                ]
                query_embedding = [first_result.timings_ms.get("embed_ms", 0.0)]
                for stage, value in first_result.timings_ms.items():
                    stage_latencies.setdefault(stage, []).append(value)
                for _ in range(self.repeats - 1):
                    repeated = self.pipeline.search(
                        query.query,
                        mode=mode,
                        top_k=self.top_k,
                        use_metadata=self.use_metadata,
                    )
                    query_latencies.append(repeated.timings_ms["total_ms"])
                    query_retrieval.append(
                        sum(
                            value
                            for stage, value in repeated.timings_ms.items()
                            if stage not in {"embed_ms", "total_ms"}
                        )
                    )
                    for stage, value in repeated.timings_ms.items():
                        stage_latencies.setdefault(stage, []).append(value)
                retrieval_latencies.extend(query_retrieval)
                embedding_latencies.extend(query_embedding)

                per_query[query.query_id] = {
                    "query": query.query,
                    "relevant_doc_ids": list(query.relevant_doc_ids),
                    "ranked_doc_ids": ranked_ids,
                    "first_relevant_rank": first_relevant_rank(ranked_ids, query.relevant_doc_ids),
                    "metrics": metrics,
                    "latency_ms": {
                        "median": statistics.median(query_latencies),
                        "retrieval_median": statistics.median(query_retrieval),
                        "embedding_median": statistics.median(query_embedding),
                        "min": min(query_latencies),
                        "max": max(query_latencies),
                    },
                    "top_hits": [
                        {
                            "doc_id": hit.doc_id,
                            "title": hit.title,
                            "score": round(hit.score, 6),
                            "source": hit.source,
                            "card_tier": hit.card_tier,
                            "effective_year": hit.effective_year,
                        }
                        for hit in first_result.hits
                    ],
                    "parsed_query": {
                        "card_tier": first_result.parsed_query.card_tier,
                        "effective_year": first_result.parsed_query.effective_year,
                    },
                }

            summary_metrics = {
                name: round(statistics.fmean(values), 6) for name, values in mode_metrics.items()
            }
            summary_latency = {
                "p50_ms": round(percentile(retrieval_latencies, 50), 3),
                "p95_ms": round(percentile(retrieval_latencies, 95), 3),
                "p99_ms": round(percentile(retrieval_latencies, 99), 3),
                "mean_ms": round(statistics.fmean(retrieval_latencies), 3),
                "samples": len(retrieval_latencies),
                "embedding_p50_ms": round(percentile(embedding_latencies, 50), 3),
                "embedding_p99_ms": round(percentile(embedding_latencies, 99), 3),
            }
            payload["summary"][mode] = {
                "metrics": summary_metrics,
                "latency": summary_latency,
                "stage_mean_ms": {
                    stage: round(statistics.fmean(values), 3)
                    for stage, values in stage_latencies.items()
                },
            }
            payload["per_query"][mode] = per_query

        return payload


def write_evaluation(payload: dict[str, Any], path: Path) -> None:
    """Write raw experiment data for reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
