"""Retrieval metrics for binary multi-label relevance judgments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of labeled relevant documents found in the first ``k`` results."""

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(relevant & set(ranked_ids[:k])) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the first ``k`` results that are relevant."""

    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    return len(relevant & set(ranked_ids[:k])) / k


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant document."""

    relevant = set(relevant_ids)
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@K."""

    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute all required metrics for one query."""

    relevant = set(relevant_ids)
    metrics: dict[str, float] = {
        "mrr": reciprocal_rank(ranked_ids, relevant),
    }
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(ranked_ids, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked_ids, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevant, k)
    return metrics


def first_relevant_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> int | None:
    """Return the one-based rank of the first relevant document, if any."""

    relevant = set(relevant_ids)
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return index
    return None

