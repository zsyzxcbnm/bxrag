"""PostgreSQL full-text sparse retrieval stage."""

from __future__ import annotations

from src.db_protocol import DatabaseLike
from src.models import SearchHit


class SparseRetriever:
    """Query PostgreSQL full-text search using ``tsvector`` and ``ts_rank_cd``."""

    def __init__(self, database: DatabaseLike) -> None:
        self.database = database

    def search(
        self,
        query: str,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
    ) -> tuple[list[SearchHit], float]:
        return self.database.search_sparse(query, limit, chunk_strategy, filters)
