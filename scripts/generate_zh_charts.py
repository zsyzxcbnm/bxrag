"""Generate Chinese-labeled charts from the saved evaluation JSON files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
REPORTS = Path(os.environ.get("REPORTS_DIR", "reports"))
if not REPORTS.is_absolute():
    REPORTS = ROOT / REPORTS
FIGURES = REPORTS / "figures"

MODE_ZH = {
    "dense_only": "仅稠密检索",
    "sparse_only": "仅稀疏检索",
    "hybrid": "混合检索(RRF)",
    "hybrid_rerank": "混合检索+重排序",
}

STRATEGY_ZH = {
    "document": "文档级",
    "fixed": "固定窗口",
    "semantic": "语义切分",
}

COLORS = ["#3274a1", "#e1812c", "#3a923a", "#c03d3e", "#7b5aa6"]


def _load(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def _load_optional(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_quality(results: dict[str, Any]) -> None:
    summary = results["summary"]
    metric_keys = ["recall@1", "recall@5", "mrr", "ndcg@5"]
    metric_labels = ["Recall@1", "Recall@5", "MRR", "nDCG@5"]
    modes = list(MODE_ZH)
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for index, mode in enumerate(modes):
        values = [summary[mode]["metrics"][key] for key in metric_keys]
        positions = [position + index * width for position in range(len(metric_keys))]
        ax.bar(positions, values, width=width, label=MODE_ZH[mode], color=COLORS[index])
    ax.set_xticks([position + width * 1.5 for position in range(len(metric_keys))])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("分数")
    ax.set_title("Apex 信用卡知识库：检索质量对比（BGE-large GPU）", fontsize=14)
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "metrics_comparison_zh.png")


def chart_latency(results: dict[str, Any]) -> None:
    summary = results["summary"]
    modes = list(MODE_ZH)
    labels = [MODE_ZH[mode] for mode in modes]
    retrieval_p50 = [summary[mode]["latency"]["p50_ms"] for mode in modes]
    retrieval_p99 = [summary[mode]["latency"]["p99_ms"] for mode in modes]
    e2e_p50 = [
        summary[mode]["latency"]["p50_ms"] + summary[mode]["latency"].get("embedding_p50_ms", 0)
        for mode in modes
    ]
    e2e_p99 = [
        summary[mode]["latency"]["p99_ms"] + summary[mode]["latency"].get("embedding_p99_ms", 0)
        for mode in modes
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    positions = list(range(len(modes)))
    for ax, p50, p99, title in (
        (axes[0], retrieval_p50, retrieval_p99, "检索路径延迟（不含 Embedding）"),
        (axes[1], e2e_p50, e2e_p99, "近似端到端延迟（检索+Embedding）"),
    ):
        ax.bar([p - 0.18 for p in positions], p50, width=0.36, label="P50", color="#3274a1")
        ax.bar([p + 0.18 for p in positions], p99, width=0.36, label="P99", color="#e1812c")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("毫秒 (ms)")
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    _save(fig, "latency_comparison_zh.png")


def chart_chunking(chunking: dict[str, Any]) -> None:
    strategies = list(STRATEGY_ZH)
    metrics = ["recall@1", "recall@5", "mrr", "ndcg@5"]
    labels = ["Recall@1", "Recall@5", "MRR", "nDCG@5"]
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, strategy in enumerate(strategies):
        values = [chunking["strategies"][strategy]["metrics"][key] for key in metrics]
        positions = [position + index * width for position in range(len(metrics))]
        ax.bar(positions, values, width=width, label=STRATEGY_ZH[strategy], color=COLORS[index])
    ax.set_xticks([position + width for position in range(len(metrics))])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("分数")
    ax.set_title("分块策略对比（离线 Memory Backend + TF-IDF）", fontsize=13)
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "chunking_comparison_zh.png")


def chart_index_tuning(payload: dict[str, Any]) -> None:
    rows = payload["results"]
    labels = [f"m={row['m']}, efc={row['ef_construction']}, efs={row['ef_search']}" for row in rows]
    p50 = [row["p50_ms"] for row in rows]
    p99 = [row["p99_ms"] for row in rows]
    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar([p - 0.18 for p in positions], p50, width=0.36, label="P50", color="#3274a1")
    ax.bar([p + 0.18 for p in positions], p99, width=0.36, label="P99", color="#e1812c")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("毫秒 (ms)")
    ax.set_title("HNSW 参数扫描：候选检索延迟（Recall@10 均为 1.000）", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "index_tuning_zh.png")


def chart_embeddings() -> None:
    files = {
        "TF-IDF": "results_tfidf_pgvector.json",
        "BGE-large": "results_bge_large_pgvector.json",
        "Qwen3-0.6B": "results_qwen3_pgvector.json",
    }
    loaded = {name: _load_optional(file) for name, file in files.items()}
    if any(item is None for item in loaded.values()):
        return
    data = {name: item["summary"] for name, item in loaded.items() if item is not None}
    metric_keys = ["recall@1", "mrr", "ndcg@5"]
    metric_labels = ["Recall@1", "MRR", "nDCG@5"]
    names = list(files)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mode, title in (
        (axes[0], "dense_only", "仅稠密检索"),
        (axes[1], "hybrid_rerank", "混合检索+重排序"),
    ):
        width = 0.22
        for index, name in enumerate(names):
            values = [data[name][mode]["metrics"][key] for key in metric_keys]
            positions = [position + index * width for position in range(len(metric_keys))]
            ax.bar(positions, values, width=width, label=name, color=COLORS[index])
        ax.set_xticks([position + width for position in range(len(metric_keys))])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("分数")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, "embedding_comparison_zh.png")


def chart_rerankers() -> None:
    payload = _load_optional("ablation_metadata.json")
    if payload is None:
        return
    ablation = payload["summary"]
    heuristic = ablation["metadata_heuristic"]
    cross = ablation["no_metadata_cross_encoder"]
    metric_keys = ["recall@1", "recall@3", "mrr", "ndcg@5"]
    metric_labels = ["Recall@1", "Recall@3", "MRR", "nDCG@5"]
    names = ["元数据启发式", "BGE Cross-Encoder"]
    values = [heuristic, cross]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    width = 0.32
    positions = list(range(len(metric_keys)))
    for index, (name, item) in enumerate(zip(names, values, strict=True)):
        axes[0].bar(
            [p + (index - 0.5) * width for p in positions],
            [item["metrics"][key] for key in metric_keys],
            width=width,
            label=name,
            color=COLORS[index],
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("分数")
    axes[0].set_title("Hybrid + 重排序质量", fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        [0 - 0.18, 1 - 0.18],
        [heuristic["retrieval_latency"]["p50_ms"], cross["retrieval_latency"]["p50_ms"]],
        width=0.36,
        label="P50",
        color="#3274a1",
    )
    axes[1].bar(
        [0 + 0.18, 1 + 0.18],
        [heuristic["retrieval_latency"]["p99_ms"], cross["retrieval_latency"]["p99_ms"]],
        width=0.36,
        label="P99",
        color="#e1812c",
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylabel("毫秒 (ms)")
    axes[1].set_title("重排序检索延迟", fontsize=12)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, "reranker_comparison_zh.png")


def chart_ablation() -> None:
    payload = _load_optional("ablation_metadata.json")
    if payload is None:
        return
    names = {
        "no_metadata_rrf": "纯 RRF",
        "no_metadata_cross_encoder": "纯 Cross-Encoder",
        "metadata_heuristic": "元数据启发式",
    }
    metric_keys = ["recall@1", "recall@3", "mrr", "ndcg@5"]
    metric_labels = ["Recall@1", "Recall@3", "MRR", "nDCG@5"]
    modes = list(names)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for index, mode in enumerate(modes):
        values = [payload["summary"][mode]["metrics"][key] for key in metric_keys]
        positions = [position + index * width for position in range(len(metric_keys))]
        axes[0].bar(positions, values, width=width, label=names[mode], color=COLORS[index])
    axes[0].set_xticks([position + width for position in range(len(metric_keys))])
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("分数")
    axes[0].set_title("元数据规则消融：检索质量", fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    positions = list(range(len(modes)))
    p50 = [payload["summary"][mode]["retrieval_latency"]["p50_ms"] for mode in modes]
    p99 = [payload["summary"][mode]["retrieval_latency"]["p99_ms"] for mode in modes]
    axes[1].bar([p - 0.18 for p in positions], p50, width=0.36, label="P50", color="#3274a1")
    axes[1].bar([p + 0.18 for p in positions], p99, width=0.36, label="P99", color="#e1812c")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([names[mode] for mode in modes], rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("毫秒 (ms)")
    axes[1].set_title("元数据规则消融：检索延迟", fontsize=12)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, "ablation_metadata_zh.png")


def chart_architecture() -> None:
    steps = [
        "知识库\nJSON",
        "校验与\n元数据",
        "文档分块",
        "BGE/Qwen3\n向量化",
        "PostgreSQL\npgvector",
        "稠密+稀疏\n召回",
        "RRF\n融合",
        "重排序",
        "Top-5\n证据",
    ]
    fig, ax = plt.subplots(figsize=(14.8, 3.8))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 4)
    ax.axis("off")
    box_width, box_height = 1.35, 1.35
    gap = 0.25
    start_x = 0.25
    for index, step in enumerate(steps):
        x = start_x + index * (box_width + gap)
        box = FancyBboxPatch(
            (x, 1.35),
            box_width,
            box_height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.1,
            edgecolor="#4a6b82",
            facecolor="#eaf2f8" if index < 5 else "#fdf2e6",
        )
        ax.add_patch(box)
        ax.text(x + box_width / 2, 2.03, step, ha="center", va="center", fontsize=9)
        if index < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + box_width + 0.02, 2.03),
                (x + box_width + gap - 0.02, 2.03),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#708090",
            )
            ax.add_patch(arrow)
    ax.text(3.7, 3.35, "离线导入 / 向量化", ha="center", fontsize=11, color="#3274a1")
    ax.text(10.7, 3.35, "在线检索 / 重排序", ha="center", fontsize=11, color="#b35c00")
    _save(fig, "architecture_zh.png")


def main() -> None:
    _setup_font()
    results = _load("results.json")
    chunking = _load("chunking.json")
    chart_quality(results)
    chart_latency(results)
    chart_chunking(chunking)
    index_tuning = _load_optional("index_tuning.json")
    if index_tuning is not None:
        chart_index_tuning(index_tuning)
    chart_embeddings()
    chart_rerankers()
    chart_ablation()
    chart_architecture()
    print("中文图表已生成到", FIGURES)


if __name__ == "__main__":
    main()
