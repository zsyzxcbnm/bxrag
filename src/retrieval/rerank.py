"""Second-stage reranking with a dependency-free fallback."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from src.config import Settings
from src.models import ParsedQuery, SearchHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "which",
    "will",
    "with",
}

_ANSWER_TERMS: dict[str, tuple[str, ...]] = {
    "fee": ("fee", "fees", "cost", "charge", "pricing", "$"),
    "deadline": ("deadline", "days", "within", "submit", "file", "report"),
    "time": ("days", "hours", "business", "calendar", "time", "within"),
    "limit": ("limit", "maximum", "maximum", "cap", "capped", "up to"),
    "coverage": ("coverage", "covered", "protect", "protection", "deductible", "up to"),
    "apr": ("apr", "interest", "rate"),
    "eligibility": ("eligible", "eligibility", "qualify", "after", "months"),
}


def _token_list(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _tokens(text: str) -> set[str]:
    return set(_token_list(text))


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False))


class Reranker(Protocol):
    """Interface for second-stage rerankers."""

    name: str

    def rerank(
        self,
        query: str,
        parsed: ParsedQuery,
        hits: Sequence[SearchHit],
        top_n: int,
    ) -> list[SearchHit]:
        """Return the highest-scoring candidates."""


class HeuristicReranker:
    """A fast lexical/metadata reranker used when no neural model is installed."""

    name = "heuristic"

    def rerank(
        self,
        query: str,
        parsed: ParsedQuery,
        hits: Sequence[SearchHit],
        top_n: int,
    ) -> list[SearchHit]:
        query_token_list = _token_list(query)
        query_tokens = set(query_token_list)
        query_numbers = set(_NUMBER_RE.findall(query.lower()))
        query_bigrams = _bigrams(query_token_list)
        requested_types = {
            term
            for term in _ANSWER_TERMS
            if re.search(rf"\b{re.escape(term)}\b", query.lower())
        }

        scored: list[SearchHit] = []
        for hit in hits:
            text = f"{hit.title} {hit.content}".lower()
            doc_token_list = _token_list(text)
            doc_tokens = _tokens(text)
            title_tokens = _tokens(hit.title)
            numbers = set(_NUMBER_RE.findall(text))
            bigrams = _bigrams(doc_token_list)

            token_overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
            title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
            bigram_overlap = len(query_bigrams & bigrams) / max(1, len(query_bigrams))
            number_overlap = len(query_numbers & numbers) / max(1, len(query_numbers))
            answer_type_matches = sum(
                1
                for term in requested_types
                if any(synonym in text for synonym in _ANSWER_TERMS[term])
            )
            answer_type_score = answer_type_matches / max(1, len(requested_types))
            tier_match = 1.0 if parsed.card_tier and hit.card_tier == parsed.card_tier else 0.0
            tier_penalty = (
                -0.18 if parsed.card_tier and hit.card_tier != parsed.card_tier else 0.0
            )
            year_match = (
                1.0
                if parsed.effective_year and hit.effective_year == parsed.effective_year
                else 0.0
            )
            prior = 1.0 / (1.0 + 0.08 * max(0, hit.rank - 1))

            score = (
                0.45 * prior
                + 0.18 * token_overlap
                + 0.10 * title_overlap
                + 0.07 * bigram_overlap
                + 0.03 * number_overlap
                + 0.12 * tier_match
                + 0.02 * year_match
                + 0.15 * answer_type_score
                + tier_penalty
            )
            scored.append(replace(hit, score=score, source="heuristic-rerank"))

        scored.sort(key=lambda item: (item.score, item.rank), reverse=True)
        return [replace(hit, rank=index + 1) for index, hit in enumerate(scored[:top_n])]


class CrossEncoderReranker:
    """Optional Sentence-Transformers Cross-Encoder with a safe fallback."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self.model_name = model_name
        self._model: object | None = None
        self._failed = False
        self._fallback = HeuristicReranker()

    def _load(self) -> object | None:
        if self._failed:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception:
            self._failed = True
            return None
        return self._model

    def rerank(
        self,
        query: str,
        parsed: ParsedQuery,
        hits: Sequence[SearchHit],
        top_n: int,
    ) -> list[SearchHit]:
        model = self._load()
        if model is None:
            return self._fallback.rerank(query, parsed, hits, top_n)

        pairs = [(query, f"{hit.title}. {hit.content}") for hit in hits]
        try:
            predictions = model.predict(pairs)  # type: ignore[attr-defined]
        except Exception:
            self._failed = True
            return self._fallback.rerank(query, parsed, hits, top_n)

        scored: list[SearchHit] = []
        for hit, raw_score in zip(hits, predictions, strict=True):
            tier_bonus = 0.15 if parsed.card_tier and hit.card_tier == parsed.card_tier else 0.0
            year_bonus = (
                0.05
                if parsed.effective_year and hit.effective_year == parsed.effective_year
                else 0.0
            )
            scored.append(
                replace(
                    hit,
                    score=float(raw_score) + tier_bonus + year_bonus,
                    source="cross-encoder",
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return [replace(hit, rank=index + 1) for index, hit in enumerate(scored[:top_n])]


def build_reranker(settings: Settings) -> Reranker:
    """Build the configured reranker."""

    if settings.reranker_provider == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model)
    return HeuristicReranker()

