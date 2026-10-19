"""Render evaluation results as Markdown, CSV and charts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MODE_LABELS = {
    "dense_only": "Dense only",
    "sparse_only": "Sparse only",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}


def _metric_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Pipeline | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | "
        "nDCG@5 | P50 (ms) | P99 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, label in _MODE_LABELS.items():
        item = summary[mode]
        metrics = item["metrics"]
        latency = item["latency"]
        lines.append(
            "| {label} | {r1:.3f} | {r3:.3f} | {r5:.3f} | {p5:.3f} | {mrr:.3f} | "
            "{ndcg:.3f} | {p50:.2f} | {p99:.2f} |".format(
                label=label,
                r1=metrics.get("recall@1", 0),
                r3=metrics.get("recall@3", 0),
                r5=metrics.get("recall@5", 0),
                p5=metrics.get("precision@5", 0),
                mrr=metrics.get("mrr", 0),
                ndcg=metrics.get("ndcg@5", 0),
                p50=latency["p50_ms"],
                p99=latency["p99_ms"],
            )
        )
    return "\n".join(lines)


def _plot_metrics(summary: dict[str, Any], output: Path) -> None:
    metrics = ["recall@1", "recall@5", "mrr", "ndcg@5"]
    modes = list(_MODE_LABELS)
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for index, mode in enumerate(modes):
        values = [summary[mode]["metrics"].get(metric, 0) for metric in metrics]
        positions = [position + index * width for position in range(len(metrics))]
        ax.bar(positions, values, width=width, label=_MODE_LABELS[mode])
    ax.set_xticks([position + width * 1.5 for position in range(len(metrics))])
    ax.set_xticklabels([metric.upper() for metric in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval quality by pipeline")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_latency(summary: dict[str, Any], output: Path) -> None:
    modes = list(_MODE_LABELS)
    p50 = [summary[mode]["latency"]["p50_ms"] for mode in modes]
    p99 = [summary[mode]["latency"]["p99_ms"] for mode in modes]
    positions = list(range(len(modes)))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([position - 0.18 for position in positions], p50, width=0.36, label="P50")
    ax.bar([position + 0.18 for position in positions], p99, width=0.36, label="P99")
    ax.set_xticks(positions)
    ax.set_xticklabels([_MODE_LABELS[mode] for mode in modes], rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-end retrieval latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _failure_section(payload: dict[str, Any]) -> str:
    dense = payload["per_query"]["dense_only"]
    enhanced = payload["per_query"]["hybrid_rerank"]
    improved: list[str] = []
    regressed: list[str] = []
    for query_id in dense:
        dense_rank = dense[query_id]["first_relevant_rank"]
        enhanced_rank = enhanced[query_id]["first_relevant_rank"]
        if dense_rank == enhanced_rank:
            continue
        if enhanced_rank is not None and (dense_rank is None or enhanced_rank < dense_rank):
            improved.append(
                f"- {query_id}: rank {dense_rank or 'miss'} -> {enhanced_rank} — "
                f"{dense[query_id]['query']}"
            )
        elif dense_rank is not None and (enhanced_rank is None or enhanced_rank > dense_rank):
            regressed.append(
                f"- {query_id}: rank {dense_rank} -> {enhanced_rank or 'miss'} — "
                f"{dense[query_id]['query']}"
            )

    lines = ["## Failure analysis", ""]
    lines.append("Queries where hybrid + rerank moved the first relevant document upward:")
    lines.append("")
    lines.extend(improved[:10] or ["- None in this run."])
    lines.append("")
    lines.append("Queries where hybrid + rerank moved the first relevant document downward:")
    lines.append("")
    lines.extend(regressed[:10] or ["- None in this run."])
    lines.append("")
    lines.append(
        "The dominant error mode in this dataset is card-tier confusion: the same policy "
        "exists for several products with slightly different numbers. Metadata-aware "
        "reranking is therefore more valuable than generic semantic similarity."
    )
    lines.append("")
    lines.append(
        "A second issue is label exhaustiveness. For generic questions the judgment often "
        "lists only one or two card tiers even when another tier's policy contains the same "
        "fact. For example, the top result for Q-005 (DOC-0023) and Q-011 (DOC-0053) is "
        "semantically valid but not labeled relevant. Strict Recall/MRR therefore "
        "understates retrieval quality; it should not be tuned away by overfitting to the "
        "30 labeled queries."
    )
    return "\n".join(lines)


def generate_report(payload: dict[str, Any], reports_dir: Path) -> Path:
    """Generate ``results.md``, ``results.csv`` and PNG figures."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_metrics(payload["summary"], figures_dir / "metrics_comparison.png")
    _plot_latency(payload["summary"], figures_dir / "latency_percentiles.png")

    csv_path = reports_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "query_id",
                "recall@1",
                "recall@3",
                "recall@5",
                "precision@5",
                "mrr",
                "ndcg@5",
                "latency_median_ms",
                "first_relevant_rank",
            ]
        )
        for mode, queries in payload["per_query"].items():
            for query_id, item in queries.items():
                metrics = item["metrics"]
                writer.writerow(
                    [
                        mode,
                        query_id,
                        metrics.get("recall@1", 0),
                        metrics.get("recall@3", 0),
                        metrics.get("recall@5", 0),
                        metrics.get("precision@5", 0),
                        metrics.get("mrr", 0),
                        metrics.get("ndcg@5", 0),
                        round(item["latency_ms"]["median"], 3),
                        item["first_relevant_rank"],
                    ]
                )

    metadata = payload["metadata"]
    report = f"""# Apex RAG Retrieval Evaluation

Generated at `{metadata["generated_at"]}`.

## Setup

- Queries: **{metadata["query_count"]}**
- Embedding: `{metadata["embedding_model"]}` ({metadata["embedding_dim"]} dimensions)
- Embedding provider: `{metadata["embedding_provider"]}`
- Reranker: `{metadata["reranker"]}`
- RRF `k`: **{metadata["rrf_k"]}**
- RRF weights: dense **{metadata["dense_weight"]}**, sparse **{metadata["sparse_weight"]}**
- Rerank candidates: **{metadata["rerank_candidates"]}**

The default offline configuration uses a local TF-IDF + SVD embedding and a
metadata-aware heuristic reranker so the evaluation is reproducible without
network access or API keys. The OpenAI embedding and Sentence-Transformers
Cross-Encoder providers can be enabled through `.env`.

## Results

{_metric_table(payload["summary"])}

![Retrieval quality](figures/metrics_comparison.png)

![Latency](figures/latency_percentiles.png)

## What the comparison shows

- Dense-only is the required baseline; sparse-only isolates the value of exact
  lexical matching on policy numbers, product names and time limits.
- Hybrid retrieval uses Reciprocal Rank Fusion over dense and PostgreSQL
  full-text candidates.
- Hybrid + rerank applies a second-stage scorer to the top candidates and
  returns the final top 5.
- Recall@10 is deliberately not the headline metric: with 150 short documents
  and multi-label relevance it saturates quickly. Recall@1, MRR and nDCG@5 are
  the discriminating metrics.

{_failure_section(payload)}

## Reproducibility and limitations

- All labels come from `data/eval_queries.json`; no judge or generated answer
  is used for the retrieval metrics.
- This is a small synthetic benchmark. Hyperparameters are global, not tuned
  per query.
- The local LSA embedding is an offline fallback, not a production embedding
  model. The same interface accepts OpenAI or BGE embeddings.
- End-to-end latency includes query embedding and reranking; database-only
  latency is reported separately in the raw JSON.
"""
    report_path = reports_dir / "results.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path

