"""Metric correctness tests."""

from __future__ import annotations

import math

from src.eval.metrics import (
    evaluate_ranking,
    first_relevant_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_and_precision_at_k() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"a", "c"}
    assert recall_at_k(ranked, relevant, 1) == 0.5
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert precision_at_k(ranked, relevant, 3) == 2 / 3


def test_mrr_and_first_rank() -> None:
    ranked = ["x", "b", "a"]
    relevant = {"a", "b"}
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert first_relevant_rank(ranked, relevant) == 2
    assert first_relevant_rank(["x"], relevant) is None


def test_ndcg_binary() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"a", "c"}
    expected = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert math.isclose(ndcg_at_k(ranked, relevant, 3), expected)
    assert ndcg_at_k(ranked, set(), 3) == 0.0


def test_evaluate_ranking_contains_required_metrics() -> None:
    metrics = evaluate_ranking(["a", "b"], {"b"}, k_values=(1, 3, 5, 10))
    assert metrics["mrr"] == 0.5
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["ndcg@5"] > 0.0

