"""Normalize raw JSON records into typed documents."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from src.models import Document

_POLICY_CODE_RE = re.compile(r"Policy Code:\s*([^\.]+)")
_YEAR_SUFFIX_RE = re.compile(r"-\d{4}$")
_TRAILING_YEAR_RE = re.compile(r"\s*\((?:19|20)\d{2}\)\s*$")


def _required_text(raw: dict[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"Document field {key!r} is required")
    return value


def extract_topic(title: str) -> str:
    """Return the human-readable policy topic from a title."""

    topic = title.split(" - ", 1)[1] if " - " in title else title
    return _TRAILING_YEAR_RE.sub("", topic).strip()


def normalize_document(raw: dict[str, Any]) -> Document:
    """Validate a raw JSON document and add derived metadata."""

    doc_id = _required_text(raw, "doc_id")
    title = _required_text(raw, "title")
    content = _required_text(raw, "content")
    category = _required_text(raw, "category")
    card_tier = _required_text(raw, "card_tier")
    effective_year = str(raw.get("effective_year", "")).strip()
    if not effective_year:
        raise ValueError(f"Document {doc_id} is missing effective_year")

    match = _POLICY_CODE_RE.search(content)
    policy_code = match.group(1).strip() if match else ""
    policy_family = _YEAR_SUFFIX_RE.sub("", policy_code)
    topic = extract_topic(title)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    metadata: dict[str, Any] = {
        "policy_code": policy_code,
        "policy_family": policy_family,
        "topic": topic,
        "content_hash": content_hash,
    }
    return Document(
        doc_id=doc_id,
        title=title,
        content=content,
        category=category,
        card_tier=card_tier,
        effective_year=effective_year,
        policy_code=policy_code,
        policy_family=policy_family,
        topic=topic,
        content_hash=content_hash,
        metadata=metadata,
    )

