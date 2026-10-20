"""Ablation: pure RRF, pure Cross-Encoder and metadata heuristic."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.config import Settings
from src.db import Database
from src.eval.metrics import evaluate_ranking
from src.eval.runner import EVAL_TOP_K, percentile
from src.ingestion.embedder import Embedder
from src.models import QueryExample
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.rerank import CrossEncoderReranker, HeuristicReranker


def run_ablation(
    settings: Settings,
    embedder: Embedder,
    queries: list[QueryExample],
    repeats: int = 3,
) -> dict[str, Any]:
    """Compare no-metadata retrieval with the metadata heuristic and reranker."""

    database = Database(
        dsn=settings.postgres_dsn,
        embedding_dim=embedder.dim,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    heuristic_pipeline = RetrievalPipeline(
        database.clone(),
        embedder,
        settings,
        reranker=HeuristicReranker(),
    )
    cross_pipeline = RetrievalPipeline(
        database.clone(),
        embedder,
        settings,
        reranker=CrossEncoderReranker(
            settings.reranker_model,
            device=settings.reranker_device,
            fp16=settings.reranker_fp16,
        ),
    )

    # Warm the Cross-Encoder/model paths before measuring.
    if queries:
        cross_pipeline.search(queries[0].query, mode="hybrid_rerank", use_metadata=False)
        heuristic_pipeline.search(queries[0].query, mode="hybrid_rerank", use_metadata=True)

    modes = [
        ("no_metadata_rrf", heuristic_pipeline, "hybrid", False),
        ("no_metadata_cross_encoder", cross_pipeline, "hybrid_rerank", False),
        ("metadata_heuristic", heuristic_pipeline, "hybrid_rerank", True),
    ]
    payload: dict[str, Any] = {
        "metadata": {
            "query_count": len(queries),
            "repeats": repeats,
            "embedding_model": embedder.name,
            "embedding_dim": embedder.dim,
            "cross_encoder": settings.reranker_model,
            "cross_encoder_device": settings.reranker_device,
            "cross_encoder_fp16": settings.reranker_fp16,
            "top_k": EVAL_TOP_K,
        },
        "summary": {},
        "per_query": {},
    }

    for name, pipeline, mode, use_metadata in modes:
        pipeline.clear_query_cache()
        metric_values: dict[str, list[float]] = {}
        retrieval_latencies: list[float] = []
        embedding_latencies: list[float] = []
        stage_latencies: dict[str, list[float]] = {}
        per_query: dict[str, Any] = {}

        for query in queries:
            first = pipeline.search(
                query.query,
                mode=mode,  # type: ignore[arg-type]
                top_k=EVAL_TOP_K,
                use_metadata=use_metadata,
            )
            ranked = [hit.doc_id for hit in first.hits]
            metrics = evaluate_ranking(ranked, query.relevant_doc_ids)
            for metric, value in metrics.items():
                metric_values.setdefault(metric, []).append(value)

            retrieval = sum(
                value
                for stage, value in first.timings_ms.items()
                if stage not in {"embed_ms", "total_ms"}
            )
            retrieval_latencies.append(retrieval)
            embedding_latencies.append(first.timings_ms.get("embed_ms", 0.0))
            for stage, value in first.timings_ms.items():
                stage_latencies.setdefault(stage, []).append(value)

            for _ in range(repeats - 1):
                repeated = pipeline.search(
                    query.query,
                    mode=mode,  # type: ignore[arg-type]
                    top_k=EVAL_TOP_K,
                    use_metadata=use_metadata,
                )
                retrieval_latencies.append(
                    sum(
                        value
                        for stage, value in repeated.timings_ms.items()
                        if stage not in {"embed_ms", "total_ms"}
                    )
                )
                for stage, value in repeated.timings_ms.items():
                    stage_latencies.setdefault(stage, []).append(value)

            per_query[query.query_id] = {
                "ranked_doc_ids": ranked,
                "metrics": metrics,
            }

        payload["summary"][name] = {
            "metrics": {
                metric: round(statistics.fmean(values), 6)
                for metric, values in metric_values.items()
            },
            "retrieval_latency": {
                "p50_ms": round(percentile(retrieval_latencies, 50), 3),
                "p99_ms": round(percentile(retrieval_latencies, 99), 3),
            },
            "embedding_latency": {
                "p50_ms": round(percentile(embedding_latencies, 50), 3),
                "p99_ms": round(percentile(embedding_latencies, 99), 3),
            },
            "stage_mean_ms": {
                stage: round(statistics.fmean(values), 3)
                for stage, values in stage_latencies.items()
            },
        }
        payload["per_query"][name] = per_query

    database.close()
    return payload


def write_ablation(payload: dict[str, Any], reports_dir: Path) -> Path:
    """Write the metadata-rule ablation as JSON, CSV and Markdown."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ablation_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows: list[list[Any]] = []
    for name, item in payload["summary"].items():
        metrics = item["metrics"]
        rows.append(
            [
                name,
                metrics.get("recall@1", 0),
                metrics.get("recall@3", 0),
                metrics.get("recall@10", 0),
                metrics.get("mrr", 0),
                metrics.get("ndcg@5", 0),
                metrics.get("ndcg@10", 0),
                item["retrieval_latency"]["p50_ms"],
                item["retrieval_latency"]["p99_ms"],
            ]
        )

    with (reports_dir / "ablation_metadata.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pipeline",
                "recall@1",
                "recall@3",
                "recall@10",
                "mrr",
                "ndcg@5",
                "ndcg@10",
                "p50_ms",
                "p99_ms",
            ]
        )
        writer.writerows(rows)

    lines = [
        "| Pipeline | Recall@1 | Recall@3 | Recall@10 | MRR | nDCG@5 | nDCG@10 | "
        "Retrieval P50 | Retrieval P99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]:.3f} | {row[2]:.3f} | {row[3]:.3f} | "
            f"{row[4]:.3f} | {row[5]:.3f} | {row[6]:.3f} | "
            f"{row[7]:.2f} ms | {row[8]:.2f} ms |"
        )
    table = "\n".join(lines)
    report = f"""# Metadata Rule Ablation

This experiment separates three effects:

- `no_metadata_rrf`: pure dense + sparse hybrid retrieval and RRF, with no
  card-tier/year filters and no manual reranker.
- `no_metadata_cross_encoder`: the same retrieval, followed by the pure BGE
  Cross-Encoder without heuristic blending.
- `metadata_heuristic`: an opt-in configuration with card-tier/year filters
  and the metadata-aware heuristic reranker.

{table}

The metadata rules are dataset-informed and are not the project default. This
table makes their performance explicit and compares them with the rules-free
RRF and Cross-Encoder controls. Every row retrieves ten documents; @5 and @10
metrics are derived from the same ranking.
"""
    path = reports_dir / "ablation_metadata.md"
    path.write_text(report, encoding="utf-8")
    return path
