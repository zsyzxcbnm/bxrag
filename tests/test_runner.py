"""Evaluation runner regression tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from src.config import Settings
from src.eval.runner import EVAL_TOP_K, EvalRunner
from src.models import ParsedQuery, QueryExample, RetrievalResult, SearchHit


@dataclass
class _Embedder:
    name: str = "stub"
    dim: int = 2

    def embed_query(self, query: str) -> np.ndarray:
        del query
        return np.ones(self.dim, dtype=np.float32)


@dataclass
class _Reranker:
    name: str = "stub"


@dataclass
class _Pipeline:
    embedder: _Embedder = field(default_factory=_Embedder)
    reranker: _Reranker = field(default_factory=_Reranker)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def clear_query_cache(self) -> None:
        return None

    def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls.append({"query": query, **kwargs})
        limit = int(kwargs["top_k"])
        hits = [
            SearchHit(
                doc_id=f"D{index}",
                chunk_id=f"D{index}::0",
                title=f"Doc {index}",
                content=f"content {index}",
                score=1.0 / (index + 1),
                source="stub",
                rank=index + 1,
            )
            for index in range(limit)
        ]
        return RetrievalResult(
            hits=hits,
            parsed_query=ParsedQuery(original=query, normalized=query),
            mode=str(kwargs["mode"]),
            timings_ms={"total_ms": 0.1, "embed_ms": 0.0},
        )


def test_runner_fetches_ten_results_for_recall_at_ten() -> None:
    pipeline = _Pipeline()
    runner = EvalRunner(
        pipeline=pipeline,  # type: ignore[arg-type]
        settings=Settings(),
        repeats=1,
    )
    payload = runner.run(
        [
            QueryExample(
                query_id="Q-1",
                query="test query",
                relevant_doc_ids=("D8",),
                gold_response="",
            )
        ]
    )

    assert EVAL_TOP_K == 10
    assert all(call["top_k"] == 10 for call in pipeline.calls)
    assert all(call["use_metadata"] is False for call in pipeline.calls)
    ranked = payload["per_query"]["hybrid_rerank"]["Q-1"]["ranked_doc_ids"]
    assert len(ranked) == 10
    metrics = payload["summary"]["hybrid_rerank"]["metrics"]
    assert metrics["recall@5"] == 0.0
    assert metrics["recall@10"] == 1.0
    assert metrics["ndcg@10"] > 0.0


def test_runner_rejects_fewer_than_ten_results() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        EvalRunner(
            pipeline=_Pipeline(),  # type: ignore[arg-type]
            settings=Settings(),
            top_k=5,
        )
