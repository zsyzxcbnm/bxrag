"""Compare document, fixed-window and sentence-aware chunking strategies."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import Settings
from src.db_memory import MemoryDatabase
from src.eval.metrics import evaluate_ranking
from src.eval.runner import EVAL_TOP_K, percentile
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import Embedder
from src.ingestion.loader import load_documents
from src.models import QueryExample
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.rerank import HeuristicReranker

STRATEGIES = ("document", "fixed", "sentence")

_QUALITY_METRICS = ("recall@1", "mrr", "ndcg@5")


def _metric_deltas(
    payload: dict[str, Any],
    left: str,
    right: str,
) -> dict[str, float]:
    """Return metric differences for ``left - right``."""

    left_metrics = payload["strategies"][left]["metrics"]
    right_metrics = payload["strategies"][right]["metrics"]
    return {
        metric: float(left_metrics.get(metric, 0.0)) - float(right_metrics.get(metric, 0.0))
        for metric in _QUALITY_METRICS
    }


def _lead_text(deltas: dict[str, float]) -> str:
    """Describe positive metric deltas in report prose."""

    labels = {"recall@1": "Recall@1", "mrr": "MRR", "ndcg@5": "nDCG@5"}
    return ", ".join(
        f"{labels[metric]} {delta:+.3f}" for metric, delta in deltas.items() if abs(delta) >= 0.0005
    )


def _interpret(payload: dict[str, Any]) -> dict[str, str]:
    """Derive the English and Chinese conclusions from measured metrics."""

    document_vs_sentence = _metric_deltas(payload, "document", "sentence")
    document_vs_fixed = _metric_deltas(payload, "document", "fixed")
    sentence_vs_document = {key: -value for key, value in document_vs_sentence.items()}
    median_chars = float(payload.get("median_document_chars", 0.0))
    corpus_note_en = (
        f"The supplied documents are already short (median {median_chars:.0f} characters)."
        if median_chars
        else "The supplied documents are already short."
    )
    corpus_note_zh = (
        f"提供的文档本身已经较短（字符数中位数 {median_chars:.0f}）。"
        if median_chars
        else "提供的文档本身已经较短。"
    )

    sentence_material_gain = (
        sum(delta >= 0.01 for delta in sentence_vs_document.values()) >= 2
        and min(sentence_vs_document.values()) >= -0.005
    )
    document_at_least_as_good = all(delta >= -0.005 for delta in document_vs_sentence.values())

    if sentence_material_gain:
        en = (
            "Sentence-aware chunking is the stronger strategy in this run "
            f"({_lead_text(sentence_vs_document)} versus document-level). "
            "The measured gain is large enough to justify the extra chunks."
        )
        zh = (
            "本次实验中句子感知切分更强"
            f"（相对文档级：{_lead_text(sentence_vs_document)}），"
            "增益足以抵消新增 chunk 带来的索引复杂度。"
        )
        recommendation_en = (
            "Sentence-aware chunking is the better measured default for this corpus."
        )
        recommendation_zh = "在这份语料上，句子感知切分是更合适的默认策略。"
    elif document_at_least_as_good:
        if all(abs(delta) <= 0.005 for delta in document_vs_sentence.values()):
            en = (
                "Document-level and sentence-aware chunking are effectively tied on the "
                "measured quality metrics."
            )
            zh = "文档级与句子感知切分在本次评测指标上基本持平。"
        else:
            en = (
                "Document-level chunking is at least as good as sentence-aware chunking "
                f"and leads by {_lead_text(document_vs_sentence)}."
            )
            zh = (
                "文档级切分不弱于句子感知切分，"
                f"并在以下指标上领先：{_lead_text(document_vs_sentence)}。"
            )
        recommendation_en = (
            "Document-level chunking remains the simplest defensible default on this corpus."
        )
        recommendation_zh = "当前语料下，文档级仍是最简单、最稳妥的默认策略。"
    else:
        en = (
            "Document-level and sentence-aware chunking trade wins across metrics "
            f"(document minus sentence: {_lead_text(document_vs_sentence)})."
        )
        zh = (
            "文档级与句子感知切分在不同指标上各有胜负"
            f"（文档级减句子感知：{_lead_text(document_vs_sentence)}）。"
        )
        recommendation_en = (
            "Neither strategy dominates; retain document-level as the simple default and "
            "revisit the choice when the corpus contains longer documents."
        )
        recommendation_zh = "没有一种策略全面占优；暂保留文档级作为简单默认，待语料变长后重评。"

    fixed_is_worse = all(delta >= 0.0 for delta in document_vs_fixed.values())
    if fixed_is_worse:
        fixed_note_en = (
            "Fixed-window chunking is weaker than document-level chunking "
            f"(document minus fixed: {_lead_text(document_vs_fixed)})."
        )
        fixed_note_zh = (
            "固定窗口切分弱于文档级"
            f"（文档级减固定窗口：{_lead_text(document_vs_fixed)}）。"
        )
    else:
        fixed_note_en = (
            "Fixed-window chunking does not lose uniformly: "
            f"document minus fixed is {_lead_text(document_vs_fixed)}."
        )
        fixed_note_zh = (
            "固定窗口并非全面落后"
            f"（文档级减固定窗口：{_lead_text(document_vs_fixed)}）。"
        )

    return {
        "en_quality": f"{corpus_note_en} {en} {fixed_note_en} {recommendation_en}",
        "zh_quality": f"{corpus_note_zh}{zh}{fixed_note_zh}{recommendation_zh}",
    }


def compare_chunking(
    settings: Settings,
    embedder: Embedder,
    queries: list[QueryExample],
) -> dict[str, Any]:
    """Evaluate the same hybrid+rerank pipeline under three chunking strategies."""

    documents = load_documents(settings.data_dir / "kb_documents.json")
    results: dict[str, Any] = {
        "strategies": {},
        "queries": len(queries),
        "median_document_chars": statistics.median(
            len(document.content) for document in documents
        ),
        "median_document_tokens": statistics.median(
            len(document.content.split()) for document in documents
        ),
        "embedding_model": embedder.name,
        "top_k": EVAL_TOP_K,
        "metadata_rules": False,
    }
    for strategy in STRATEGIES:
        chunks = chunk_documents(documents, strategy)
        vectors = embedder.embed([chunk.text for chunk in chunks])
        database = MemoryDatabase(embedder.dim)
        database.init_schema()
        database.upsert_documents(documents)
        database.upsert_chunks(chunks, vectors)
        pipeline = RetrievalPipeline(
            database=database,
            embedder=embedder,
            settings=settings,
            reranker=HeuristicReranker(),
        )

        metric_values: dict[str, list[float]] = {}
        latencies: list[float] = []
        per_query: dict[str, Any] = {}
        for query in queries:
            result = pipeline.search(
                query.query,
                mode="hybrid_rerank",
                chunk_strategy=strategy,
                top_k=EVAL_TOP_K,
            )
            ranked_ids = [hit.doc_id for hit in result.hits]
            metrics = evaluate_ranking(ranked_ids, query.relevant_doc_ids)
            for name, value in metrics.items():
                metric_values.setdefault(name, []).append(value)
            latencies.append(result.timings_ms["total_ms"])
            per_query[query.query_id] = {
                "ranked_doc_ids": ranked_ids,
                "first_relevant_rank": next(
                    (
                        index
                        for index, doc_id in enumerate(ranked_ids, start=1)
                        if doc_id in set(query.relevant_doc_ids)
                    ),
                    None,
                ),
                "metrics": metrics,
            }

        results["strategies"][strategy] = {
            "chunk_count": len(chunks),
            "avg_tokens": round(
                statistics.fmean(chunk.token_count for chunk in chunks),
                2,
            ),
            "metrics": {
                name: round(statistics.fmean(values), 6) for name, values in metric_values.items()
            },
            "latency": {
                "p50_ms": round(percentile(latencies, 50), 3),
                "p99_ms": round(percentile(latencies, 99), 3),
            },
            "per_query": per_query,
        }
    return results


def write_chunking_report(payload: dict[str, Any], reports_dir: Path) -> Path:
    """Write a Markdown/CSV/PNG chunking comparison."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "chunking.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    labels = {
        "document": "Document-level",
        "fixed": "Fixed window",
        "sentence": "Sentence-aware",
    }
    lines = [
        "| Strategy | Chunks | Avg tokens | Recall@1 | Recall@5 | Recall@10 | MRR | "
        "nDCG@5 | P50 (ms) | P99 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        item = payload["strategies"][strategy]
        metrics = item["metrics"]
        lines.append(
            "| {label} | {chunks} | {tokens:.1f} | {r1:.3f} | {r5:.3f} | {r10:.3f} | "
            "{mrr:.3f} | {ndcg:.3f} | {p50:.2f} | {p99:.2f} |".format(
                label=labels[strategy],
                chunks=item["chunk_count"],
                tokens=item["avg_tokens"],
                r1=metrics.get("recall@1", 0),
                r5=metrics.get("recall@5", 0),
                r10=metrics.get("recall@10", 0),
                mrr=metrics.get("mrr", 0),
                ndcg=metrics.get("ndcg@5", 0),
                p50=item["latency"]["p50_ms"],
                p99=item["latency"]["p99_ms"],
            )
        )
    table = "\n".join(lines)
    interpretation = _interpret(payload)

    with (reports_dir / "chunking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy",
                "chunks",
                "avg_tokens",
                "recall@1",
                "recall@5",
                "recall@10",
                "mrr",
                "ndcg@5",
            ]
        )
        for strategy in STRATEGIES:
            item = payload["strategies"][strategy]
            metrics = item["metrics"]
            writer.writerow(
                [
                    strategy,
                    item["chunk_count"],
                    item["avg_tokens"],
                    metrics.get("recall@1", 0),
                    metrics.get("recall@5", 0),
                    metrics.get("recall@10", 0),
                    metrics.get("mrr", 0),
                    metrics.get("ndcg@5", 0),
                ]
            )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    metric_names = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@5"]
    width = 0.24
    for index, strategy in enumerate(STRATEGIES):
        values = [payload["strategies"][strategy]["metrics"].get(name, 0) for name in metric_names]
        positions = [position + index * width for position in range(len(metric_names))]
        ax.bar(positions, values, width=width, label=labels[strategy])
    ax.set_xticks([position + width for position in range(len(metric_names))])
    ax.set_xticklabels([name.upper() for name in metric_names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Chunking strategy comparison")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "chunking_comparison.png", dpi=160)
    plt.close(fig)

    report = f"""# Chunking Strategy Comparison

This comparison holds the retrieval stack fixed and changes only the chunking
strategy. It uses the in-memory backend with `{payload["embedding_model"]}` and
metadata rules disabled, so every strategy runs under identical conditions.
Each strategy returns **{payload["top_k"]}** documents. `Sentence-aware` packs
complete sentences into a maximum character budget and repeats document
metadata in each chunk; it is not embedding-based semantic segmentation.

{table}

![Chunking comparison](figures/chunking_comparison.png)

## Interpretation

{interpretation["en_quality"]}

The strategies remain in the repository so the choice is measured rather than
assumed, and can be repeated after the corpus grows into longer policy
documents.
"""
    path = reports_dir / "chunking.md"
    path.write_text(report, encoding="utf-8")

    zh_table_lines = [
        "| 策略 | Chunk 数 | 平均 token | Recall@1 | Recall@5 | Recall@10 | MRR | "
        "nDCG@5 | P50 | P99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    zh_labels = {
        "document": "文档级",
        "fixed": "固定窗口",
        "sentence": "句子感知",
    }
    for strategy in STRATEGIES:
        item = payload["strategies"][strategy]
        metrics = item["metrics"]
        zh_table_lines.append(
            "| {label} | {chunks} | {tokens:.1f} | {r1:.3f} | {r5:.3f} | {r10:.3f} | "
            "{mrr:.3f} | {ndcg:.3f} | {p50:.2f} ms | {p99:.2f} ms |".format(
                label=zh_labels[strategy],
                chunks=item["chunk_count"],
                tokens=item["avg_tokens"],
                r1=metrics.get("recall@1", 0),
                r5=metrics.get("recall@5", 0),
                r10=metrics.get("recall@10", 0),
                mrr=metrics.get("mrr", 0),
                ndcg=metrics.get("ndcg@5", 0),
                p50=item["latency"]["p50_ms"],
                p99=item["latency"]["p99_ms"],
            )
        )
    zh_table = "\n".join(zh_table_lines)
    zh_report = f"""# 分块策略对比（中文版）

本实验固定检索栈，只改变分块策略；使用同一离线 Memory Backend、
`{payload["embedding_model"]}` 和关闭元数据规则的配置，保证条件一致。
每种策略返回 **{payload["top_k"]}** 条结果。
“句子感知”会按完整句子和字符上限打包 chunk，并在每个 chunk 前重复文档元数据；
它不是基于 embedding 的语义聚类，也不是标题层级切分。

{zh_table}

![分块策略对比](figures/chunking_comparison_zh.png)

## 结论

{interpretation["zh_quality"]}

英文版见 [chunking.md](chunking.md)。
"""
    (reports_dir / "chunking.zh-CN.md").write_text(zh_report, encoding="utf-8")
    return path
