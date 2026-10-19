"""Load and validate the provided knowledge base and benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.normalizer import normalize_document
from src.models import Document, QueryExample


def _load_json(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def load_documents(path: Path) -> list[Document]:
    """Load, normalize and validate all knowledge-base documents."""

    raw_documents = _load_json(path)
    documents = [normalize_document(raw) for raw in raw_documents]
    ids = [document.doc_id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate doc_id values found in knowledge base")
    return documents


def load_queries(path: Path, known_doc_ids: set[str] | None = None) -> list[QueryExample]:
    """Load benchmark queries and verify that labels point to known documents."""

    raw_queries = _load_json(path)
    queries: list[QueryExample] = []
    for raw in raw_queries:
        query_id = str(raw.get("query_id", "")).strip()
        query = str(raw.get("query", "")).strip()
        raw_relevant = raw.get("relevant_doc_ids", [])
        if not isinstance(raw_relevant, list):
            raise ValueError(f"Query {query_id} relevant_doc_ids must be a list")
        relevant = tuple(str(item) for item in raw_relevant)
        if not query_id or not query:
            raise ValueError("Each query requires query_id and query")
        if not relevant:
            raise ValueError(f"Query {query_id} has no relevant_doc_ids")
        if known_doc_ids is not None:
            missing = sorted(set(relevant) - known_doc_ids)
            if missing:
                raise ValueError(f"Query {query_id} references unknown docs: {missing}")
        queries.append(
            QueryExample(
                query_id=query_id,
                query=query,
                relevant_doc_ids=relevant,
                gold_response=str(raw.get("gold_response", "")).strip(),
            )
        )
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query_id values found in benchmark")
    return queries

