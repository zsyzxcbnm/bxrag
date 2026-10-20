"""Reranker fallback tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.models import ParsedQuery, SearchHit
from src.retrieval.rerank import CrossEncoderReranker, HeuristicReranker


def _hit(
    doc_id: str,
    title: str,
    content: str,
    card_tier: str,
    rank: int,
) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        chunk_id=f"{doc_id}::0",
        title=title,
        content=content,
        score=1.0 / rank,
        source="test",
        rank=rank,
        card_tier=card_tier,
    )


def _hits() -> list[SearchHit]:
    return [
        _hit(
            "business",
            "Apex Business Select annual fee",
            "The annual fee is 495.",
            "Apex Business Select",
            1,
        ),
        _hit(
            "platinum",
            "Apex Platinum annual fee",
            "The annual fee is 550.",
            "Apex Platinum",
            2,
        ),
        _hit(
            "rewards",
            "Apex Platinum rewards",
            "Reward points do not expire.",
            "Apex Platinum",
            3,
        ),
    ]


class _FailingModel:
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        del pairs
        raise RuntimeError("prediction failed")


@pytest.mark.parametrize("failure", ["load", "predict"])
def test_cross_encoder_fallback_honors_use_metadata(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    parsed = ParsedQuery(
        original="Apex Platinum annual fee",
        normalized="apex platinum annual fee",
        card_tier="Apex Platinum",
    )
    reranker = CrossEncoderReranker("unused")
    if failure == "load":
        monkeypatch.setattr(reranker, "_load", lambda: None)
    else:
        monkeypatch.setattr(reranker, "_load", lambda: _FailingModel())

    expected = HeuristicReranker().rerank(
        parsed.original,
        parsed,
        _hits(),
        top_n=3,
        use_metadata=False,
    )
    actual = reranker.rerank(
        parsed.original,
        parsed,
        _hits(),
        top_n=3,
        use_metadata=False,
    )
    metadata_order = [
        hit.doc_id
        for hit in HeuristicReranker().rerank(
            parsed.original,
            parsed,
            _hits(),
            top_n=3,
            use_metadata=True,
        )
    ]

    assert [hit.doc_id for hit in actual] == [hit.doc_id for hit in expected]
    assert [hit.doc_id for hit in actual] != metadata_order
