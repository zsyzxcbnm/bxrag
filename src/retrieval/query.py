"""Deterministic query understanding for product and year filters."""

from __future__ import annotations

import re

from src.models import ParsedQuery

_PRODUCT_ALIASES: tuple[tuple[str, str], ...] = (
    ("apex platinum", "Apex Platinum"),
    ("platinum", "Apex Platinum"),
    ("apex everyday freedom", "Apex Everyday Freedom"),
    ("everyday freedom", "Apex Everyday Freedom"),
    ("apex preferred travel", "Apex Preferred Travel"),
    ("preferred travel", "Apex Preferred Travel"),
    ("apex business select", "Apex Business Select"),
    ("business select", "Apex Business Select"),
)

_POLICY_CODE_RE = re.compile(r"\b[A-Z]{2,4}-\d{3}-APEX-(?:19|20)\d{2}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def parse_query(query: str) -> ParsedQuery:
    """Extract card tier, year and policy code from a short support question."""

    normalized = " ".join(query.lower().split())
    card_tier: str | None = None
    for alias, product in _PRODUCT_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            card_tier = product
            break

    year_match = _YEAR_RE.search(query)
    policy_match = _POLICY_CODE_RE.search(query)
    return ParsedQuery(
        original=query,
        normalized=normalized,
        card_tier=card_tier,
        effective_year=year_match.group(1) if year_match else None,
        policy_code=policy_match.group(0).upper() if policy_match else None,
    )
