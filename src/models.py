"""Shared typed records used across ingestion, retrieval and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A normalized knowledge-base document."""

    doc_id: str
    title: str
    content: str
    category: str
    card_tier: str
    effective_year: str
    policy_code: str
    policy_family: str
    topic: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryExample:
    """A labeled evaluation query."""

    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    gold_response: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable text unit linked back to its parent document."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    strategy: str
    text: str
    token_count: int


@dataclass(slots=True)
class SearchHit:
    """A scored retrieval result at document level."""

    doc_id: str
    chunk_id: str
    title: str
    content: str
    score: float
    source: str
    rank: int = 0
    category: str = ""
    card_tier: str = ""
    effective_year: str = ""
    policy_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """Structured attributes extracted from a natural-language query."""

    original: str
    normalized: str
    card_tier: str | None = None
    effective_year: str | None = None
    policy_code: str | None = None


@dataclass(slots=True)
class RetrievalResult:
    """Results and stage timings for one retrieval request."""

    hits: list[SearchHit]
    parsed_query: ParsedQuery
    mode: str
    timings_ms: dict[str, float]
