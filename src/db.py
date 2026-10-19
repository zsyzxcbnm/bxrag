"""PostgreSQL + pgvector storage and SQL retrieval primitives."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.models import Chunk, Document, SearchHit


class Database:
    """Thin, typed database gateway for documents and chunks."""

    def __init__(
        self,
        dsn: str,
        embedding_dim: int,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 80,
    ) -> None:
        self.dsn = dsn
        self.embedding_dim = int(embedding_dim)
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_ef_search = int(hnsw_ef_search)

    def connect(self, register_adapter: bool = True) -> psycopg.Connection[dict[str, Any]]:
        """Open a connection with vector adapters and dictionary rows."""

        connection = psycopg.connect(self.dsn, row_factory=dict_row)
        if register_adapter:
            register_vector(connection)
        return connection

    def init_schema(self, reset: bool = False) -> None:
        """Create extension, tables and indexes."""

        with self.connect(register_adapter=False) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(connection)
            if reset:
                connection.execute("DROP TABLE IF EXISTS chunks CASCADE")
                connection.execute("DROP TABLE IF EXISTS documents CASCADE")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    card_tier TEXT NOT NULL,
                    effective_year TEXT NOT NULL,
                    policy_code TEXT NOT NULL DEFAULT '',
                    policy_family TEXT NOT NULL DEFAULT '',
                    topic TEXT NOT NULL DEFAULT '',
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    content_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        chunk_strategy TEXT NOT NULL,
                        chunk_text TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        embedding vector({dim}) NOT NULL,
                        fts tsvector GENERATED ALWAYS AS (
                            to_tsvector('english', chunk_text)
                        ) STORED,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(dim=sql.Literal(self.embedding_dim))
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_card_tier ON documents(card_tier)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(effective_year)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(chunk_strategy)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin(fts)"
            )
            connection.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef})
                    """
                ).format(m=sql.Literal(self.hnsw_m), ef=sql.Literal(self.hnsw_ef_construction))
            )
            connection.execute("ANALYZE documents")
            connection.execute("ANALYZE chunks")

    def upsert_documents(self, documents: Sequence[Document]) -> None:
        """Idempotently insert or update normalized documents."""

        rows = [
            (
                document.doc_id,
                document.title,
                document.content,
                document.category,
                document.card_tier,
                document.effective_year,
                document.policy_code,
                document.policy_family,
                document.topic,
                Jsonb(document.metadata),
                document.content_hash,
            )
            for document in documents
        ]
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO documents (
                    doc_id, title, content, category, card_tier, effective_year,
                    policy_code, policy_family, topic, metadata, content_hash
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    card_tier = EXCLUDED.card_tier,
                    effective_year = EXCLUDED.effective_year,
                    policy_code = EXCLUDED.policy_code,
                    policy_family = EXCLUDED.policy_family,
                    topic = EXCLUDED.topic,
                    metadata = EXCLUDED.metadata,
                    content_hash = EXCLUDED.content_hash,
                    updated_at = now()
                """,
                rows,
            )

    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        """Idempotently insert or update chunks and their vectors."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.chunk_index,
                chunk.strategy,
                chunk.text,
                chunk.token_count,
                np.asarray(embedding, dtype=np.float32),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, chunk_index, chunk_strategy,
                    chunk_text, token_count, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    chunk_index = EXCLUDED.chunk_index,
                    chunk_strategy = EXCLUDED.chunk_strategy,
                    chunk_text = EXCLUDED.chunk_text,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )

    @staticmethod
    def _filter_clause(filters: dict[str, str] | None) -> tuple[list[str], list[str]]:
        if not filters:
            return [], []
        allowed = {"card_tier", "effective_year", "category"}
        unknown = set(filters) - allowed
        if unknown:
            raise ValueError(f"Unsupported filters: {sorted(unknown)}")
        clauses = [f"d.{key} = %s" for key in filters]
        values = [str(value) for value in filters.values()]
        return clauses, values

    def search_dense(
        self,
        query_vector: np.ndarray,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
    ) -> tuple[list[SearchHit], float]:
        """Return dense cosine candidates and elapsed milliseconds."""

        clauses, filter_values = self._filter_clause(filters)
        where = ["c.chunk_strategy = %s", *clauses]
        query_sql = f"""
            SELECT
                c.chunk_id,
                c.doc_id,
                d.title,
                c.chunk_text AS content,
                d.category,
                d.card_tier,
                d.effective_year,
                d.policy_code,
                d.metadata,
                1 - (c.embedding <=> %s) AS score
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE {' AND '.join(where)}
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """
        params: list[Any] = [query_vector, chunk_strategy, *filter_values, query_vector, int(limit)]
        started = time.perf_counter()
        with self.connect() as connection:
            connection.execute(f"SET LOCAL hnsw.ef_search = {self.hnsw_ef_search}")
            rows = connection.execute(query_sql, params).fetchall()
        elapsed = (time.perf_counter() - started) * 1000
        hits = [
            self._row_to_hit(row, source="dense", rank=index + 1)
            for index, row in enumerate(rows)
        ]
        return hits, elapsed

    def search_sparse(
        self,
        query: str,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
    ) -> tuple[list[SearchHit], float]:
        """Return PostgreSQL full-text candidates and elapsed milliseconds."""

        clauses, filter_values = self._filter_clause(filters)
        where = ["c.chunk_strategy = %s", "c.fts @@ q.tsq", *clauses]
        query_sql = f"""
            WITH q AS (SELECT websearch_to_tsquery('english', %s) AS tsq)
            SELECT
                c.chunk_id,
                c.doc_id,
                d.title,
                c.chunk_text AS content,
                d.category,
                d.card_tier,
                d.effective_year,
                d.policy_code,
                d.metadata,
                ts_rank_cd(c.fts, q.tsq) AS score
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            CROSS JOIN q
            WHERE {' AND '.join(where)}
            ORDER BY score DESC, c.chunk_id
            LIMIT %s
        """
        params: list[Any] = [query, chunk_strategy, *filter_values, int(limit)]
        started = time.perf_counter()
        with self.connect() as connection:
            rows = connection.execute(query_sql, params).fetchall()
        elapsed = (time.perf_counter() - started) * 1000
        hits = [
            self._row_to_hit(row, source="sparse", rank=index + 1)
            for index, row in enumerate(rows)
        ]
        return hits, elapsed

    @staticmethod
    def _row_to_hit(row: dict[str, Any], source: str, rank: int) -> SearchHit:
        return SearchHit(
            doc_id=str(row["doc_id"]),
            chunk_id=str(row["chunk_id"]),
            title=str(row["title"]),
            content=str(row["content"]),
            score=float(row["score"]),
            source=source,
            rank=rank,
            category=str(row["category"]),
            card_tier=str(row["card_tier"]),
            effective_year=str(row["effective_year"]),
            policy_code=str(row["policy_code"]),
            metadata=dict(row.get("metadata") or {}),
        )

    def stats(self) -> dict[str, Any]:
        """Return basic table counts and sizes."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM documents) AS documents,
                    (SELECT count(*) FROM chunks) AS chunks,
                    (SELECT count(DISTINCT chunk_strategy) FROM chunks) AS strategies,
                    pg_size_pretty(pg_total_relation_size('chunks')) AS chunks_size
                """
            ).fetchone()
        return dict(row or {})

    def clear(self) -> None:
        """Delete all rows while keeping the schema."""

        with self.connect() as connection:
            connection.execute("TRUNCATE TABLE chunks, documents CASCADE")

    def doc_ids(self) -> Iterable[str]:
        """Return all document identifiers."""

        with self.connect() as connection:
            rows = connection.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()
        return [str(row["doc_id"]) for row in rows]

