"""Structural interface shared by PostgreSQL and in-memory backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from src.models import Chunk, Document, SearchHit


class DatabaseLike(Protocol):
    """Minimal database surface used by retrieval and ingestion."""

    def init_schema(self, reset: bool = False) -> None:
        """Create or reset the storage layer."""

    def clone(self) -> DatabaseLike:
        """Return an independent or safely shareable backend handle."""

    def upsert_documents(self, documents: Sequence[Document]) -> None:
        """Insert or update documents."""

    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        """Insert or update chunks and vectors."""

    def search_dense(
        self,
        query_vector: np.ndarray,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
        force_index: bool = False,
    ) -> tuple[list[SearchHit], float]:
        """Return dense candidates and elapsed milliseconds."""

    def search_sparse(
        self,
        query: str,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
    ) -> tuple[list[SearchHit], float]:
        """Return sparse candidates and elapsed milliseconds."""

    def stats(self) -> dict[str, Any]:
        """Return storage statistics."""

