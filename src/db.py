"""PostgreSQL + pgvector storage and SQL retrieval primitives."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
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
        self._connection: psycopg.Connection[dict[str, Any]] | None = None

    def _get_connection(self) -> psycopg.Connection[dict[str, Any]]:
        """Return a reusable connection with vector adapters registered."""

        if self._connection is None or self._connection.closed:
            self._connection = psycopg.connect(self.dsn, row_factory=dict_row)
            register_vector(self._connection)
            self._connection.execute("SET jit = off")
        return self._connection

    @contextmanager
    def _session(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        """Commit a transaction without closing the reusable connection."""

        connection = self._get_connection()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def close(self) -> None:
        """Close the cached connection."""

        if self._connection is not None and not self._connection.closed:
            self._connection.close()
        self._connection = None

    def init_schema(self, reset: bool = False) -> None:
        """Create extension, tables and indexes."""

        self.close()
        with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
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
        with self._session() as connection, connection.cursor() as cursor:
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
        with self._session() as connection, connection.cursor() as cursor:
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
        force_index: bool = False,
    ) -> tuple[list[SearchHit], float]:
        """Return dense cosine candidates and elapsed milliseconds."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        started = time.perf_counter()
        with self._session() as connection:
            connection.execute(f"SET LOCAL hnsw.ef_search = {self.hnsw_ef_search}")
            if force_index:
                connection.execute("SET LOCAL enable_seqscan = off")
                connection.execute("SET LOCAL enable_bitmapscan = off")
            ids = self._candidate_ids(
                connection,
                query_vector,
                limit,
                chunk_strategy,
                filters,
            )
            rows = self._fetch_chunk_rows(connection, ids)
        elapsed = (time.perf_counter() - started) * 1000
        hits = self._ordered_hits(rows, ids, source="dense")
        return hits, elapsed

    def search_sparse(
        self,
        query: str,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
    ) -> tuple[list[SearchHit], float]:
        """Return PostgreSQL full-text candidates and elapsed milliseconds."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        terms = re.findall(r"[A-Za-z0-9%]+", query)
        sparse_query = " OR ".join(terms) if terms else query
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
            LIMIT {int(limit)}
        """
        params: list[Any] = [sparse_query, chunk_strategy, *filter_values]
        started = time.perf_counter()
        with self._session() as connection:
            rows = connection.execute(query_sql, params).fetchall()
        elapsed = (time.perf_counter() - started) * 1000
        hits = [
            self._row_to_hit(row, source="sparse", rank=index + 1)
            for index, row in enumerate(rows)
        ]
        return hits, elapsed

    def search_dense_index(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
    ) -> tuple[list[SearchHit], float]:
        """Benchmark an HNSW scan without the small-corpus strategy filter.

        The provided evaluation has only document-strategy chunks. A normal
        query includes ``chunk_strategy = 'document'``; at 150 rows the planner
        prefers a full scan plus sort. This method isolates the index scan so
        ``m`` / ``ef_construction`` / ``ef_search`` can be compared.
        """

        if limit <= 0:
            raise ValueError("limit must be positive")
        started = time.perf_counter()
        with self._session() as connection:
            connection.execute(f"SET LOCAL hnsw.ef_search = {self.hnsw_ef_search}")
            connection.execute("SET LOCAL enable_seqscan = off")
            connection.execute("SET LOCAL enable_bitmapscan = off")
            ids = self._candidate_ids(
                connection,
                query_vector,
                limit,
                chunk_strategy=None,
                filters=None,
            )
            rows = self._fetch_chunk_rows(connection, ids)
        elapsed = (time.perf_counter() - started) * 1000
        hits = self._ordered_hits(rows, ids, source="dense-index")
        return hits, elapsed

    def _candidate_ids(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        query_vector: np.ndarray,
        limit: int,
        chunk_strategy: str | None,
        filters: dict[str, str] | None,
    ) -> list[str]:
        """Return ordered chunk IDs from an HNSW candidate query."""

        where: list[str] = []
        params: list[Any] = []
        if chunk_strategy is not None:
            where.append("chunk_strategy = %s")
            params.append(chunk_strategy)

        if filters:
            allowed = {"card_tier", "effective_year", "category"}
            unknown = set(filters) - allowed
            if unknown:
                raise ValueError(f"Unsupported filters: {sorted(unknown)}")
            document_where = " AND ".join(f"{key} = %s" for key in filters)
            document_values = [str(value) for value in filters.values()]
            doc_rows = connection.execute(
                f"SELECT doc_id FROM documents WHERE {document_where}",
                document_values,
            ).fetchall()
            document_ids = [str(row["doc_id"]) for row in doc_rows]
            if not document_ids:
                return []
            where.append("doc_id = ANY(%s)")
            params.append(document_ids)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(query_vector)
        rows = connection.execute(
            f"""
            SELECT chunk_id
            FROM chunks
            {where_sql}
            ORDER BY embedding <=> %s
            LIMIT {int(limit)}
            """,
            params,
        ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    @staticmethod
    def _fetch_chunk_rows(
        connection: psycopg.Connection[dict[str, Any]],
        chunk_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        return connection.execute(
            """
            SELECT
                c.chunk_id,
                c.doc_id,
                d.title,
                c.chunk_text AS content,
                d.category,
                d.card_tier,
                d.effective_year,
                d.policy_code,
                d.metadata
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id = ANY(%s)
            """,
            (chunk_ids,),
        ).fetchall()

    @classmethod
    def _ordered_hits(
        cls,
        rows: list[dict[str, Any]],
        ordered_ids: list[str],
        source: str,
    ) -> list[SearchHit]:
        """Map detail rows back to candidate order and assign rank scores."""

        by_id = {str(row["chunk_id"]): row for row in rows}
        hits: list[SearchHit] = []
        denominator = max(1, len(ordered_ids))
        for index, chunk_id in enumerate(ordered_ids):
            row = by_id.get(chunk_id)
            if row is None:
                continue
            hits.append(
                cls._row_to_hit(
                    {**row, "score": 1.0 - index / denominator},
                    source=source,
                    rank=index + 1,
                )
            )
        return hits

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

        with self._session() as connection:
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

    def chunk_vectors(self, chunk_strategy: str = "document") -> tuple[list[str], np.ndarray]:
        """Load chunk identifiers and vectors for exact-search evaluation."""

        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, embedding
                FROM chunks
                WHERE chunk_strategy = %s
                ORDER BY chunk_id
                """,
                (chunk_strategy,),
            ).fetchall()
        ids = [str(row["chunk_id"]) for row in rows]
        vectors = np.asarray(
            [np.asarray(row["embedding"].to_numpy(), dtype=np.float32) for row in rows],
            dtype=np.float32,
        )
        return ids, vectors

    def rebuild_hnsw_index(self, m: int, ef_construction: int) -> float:
        """Rebuild the HNSW index and return index build time in milliseconds."""

        started = time.perf_counter()
        with self._session() as connection:
            connection.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
            connection.execute(
                sql.SQL(
                    """
                    CREATE INDEX idx_chunks_embedding_hnsw
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef})
                    """
                ).format(m=sql.Literal(int(m)), ef=sql.Literal(int(ef_construction)))
            )
            connection.execute("ANALYZE chunks")
        return (time.perf_counter() - started) * 1000

    def hnsw_index_size(self) -> str:
        """Return the human-readable HNSW index size."""

        with self._session() as connection:
            row = connection.execute(
                "SELECT pg_size_pretty(pg_relation_size('idx_chunks_embedding_hnsw')) AS size"
            ).fetchone()
        return str((row or {}).get("size", "unknown"))

    def clear(self) -> None:
        """Delete all rows while keeping the schema."""

        with self._session() as connection:
            connection.execute("TRUNCATE TABLE chunks, documents CASCADE")

    def doc_ids(self) -> Iterable[str]:
        """Return all document identifiers."""

        with self._session() as connection:
            rows = connection.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()
        return [str(row["doc_id"]) for row in rows]

