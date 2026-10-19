"""Dense-vector retrieval stage."""

from __future__ import annotations

import numpy as np

from src.db_protocol import DatabaseLike
from src.models import SearchHit


class DenseRetriever:
    """Query PostgreSQL/pgvector with a normalized dense vector."""

    def __init__(self, database: DatabaseLike) -> None:
        self.database = database

    def search(
        self,
        query_vector: np.ndarray,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
        force_index: bool = False,
    ) -> tuple[list[SearchHit], float]:
        return self.database.search_dense(
            query_vector,
            limit,
            chunk_strategy,
            filters,
            force_index=force_index,
        )

