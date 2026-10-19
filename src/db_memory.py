"""In-memory retrieval backend for offline validation and CI."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.models import Chunk, Document, SearchHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")
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


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS
    ]


class MemoryDatabase:
    """Small BM25 + cosine store used when PostgreSQL/Docker is unavailable.

    It mirrors the PostgreSQL gateway API so the retrieval pipeline and
    evaluation code can be exercised without changing any business logic.
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = int(embedding_dim)
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self._tokenized: dict[str, list[str]] = {}
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0

    def init_schema(self, reset: bool = False) -> None:
        if reset:
            self.clear()

    def clone(self) -> MemoryDatabase:
        """Memory reads are safe to share, so return the same store."""

        return self

    def clear(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self.vectors.clear()
        self._tokenized.clear()
        self._df.clear()
        self._avgdl = 0.0

    def upsert_documents(self, documents: Sequence[Document]) -> None:
        for document in documents:
            self.documents[document.doc_id] = document

    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = np.asarray(embedding, dtype=np.float32)
            self._tokenized[chunk.chunk_id] = _tokenize(chunk.text)
        self._rebuild_statistics()

    def _rebuild_statistics(self) -> None:
        self._df = Counter()
        total_length = 0
        for tokens in self._tokenized.values():
            self._df.update(set(tokens))
            total_length += len(tokens)
        self._avgdl = total_length / max(1, len(self._tokenized))

    def _candidate_chunks(
        self,
        chunk_strategy: str,
        filters: dict[str, str] | None,
    ) -> list[Chunk]:
        candidates: list[Chunk] = []
        for chunk in self.chunks.values():
            document = self.documents.get(chunk.doc_id)
            if document is None or chunk.strategy != chunk_strategy:
                continue
            if filters and any(getattr(document, key) != value for key, value in filters.items()):
                continue
            candidates.append(chunk)
        return candidates

    def _to_hit(self, chunk: Chunk, score: float, source: str) -> SearchHit:
        document = self.documents[chunk.doc_id]
        return SearchHit(
            doc_id=document.doc_id,
            chunk_id=chunk.chunk_id,
            title=document.title,
            content=chunk.text,
            score=float(score),
            source=source,
            category=document.category,
            card_tier=document.card_tier,
            effective_year=document.effective_year,
            policy_code=document.policy_code,
            metadata=dict(document.metadata),
        )

    def search_dense(
        self,
        query_vector: np.ndarray,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
        force_index: bool = False,
    ) -> tuple[list[SearchHit], float]:
        del force_index
        started = time.perf_counter()
        query = np.asarray(query_vector, dtype=np.float32)
        candidates = self._candidate_chunks(chunk_strategy, filters)
        scored = [
            (float(np.dot(query, self.vectors[chunk.chunk_id])), chunk)
            for chunk in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = [
            self._to_hit(chunk, score, "dense")
            for score, chunk in scored[:limit]
        ]
        for index, hit in enumerate(hits, start=1):
            hit.rank = index
        return hits, (time.perf_counter() - started) * 1000

    def search_sparse(
        self,
        query: str,
        limit: int,
        chunk_strategy: str = "document",
        filters: dict[str, str] | None = None,
    ) -> tuple[list[SearchHit], float]:
        started = time.perf_counter()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return [], (time.perf_counter() - started) * 1000
        candidates = self._candidate_chunks(chunk_strategy, filters)
        total_docs = max(1, len(self._tokenized))
        k1, b = 1.5, 0.75
        scored: list[tuple[float, Chunk]] = []
        for chunk in candidates:
            tokens = self._tokenized[chunk.chunk_id]
            frequencies = Counter(tokens)
            length = max(1, len(tokens))
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if frequency == 0:
                    continue
                idf = math.log(1 + (total_docs - self._df[token] + 0.5) / (self._df[token] + 0.5))
                denominator = frequency + k1 * (1 - b + b * length / max(self._avgdl, 1.0))
                score += idf * frequency * (k1 + 1) / denominator
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = [
            self._to_hit(chunk, score, "sparse")
            for score, chunk in scored[:limit]
        ]
        for index, hit in enumerate(hits, start=1):
            hit.rank = index
        return hits, (time.perf_counter() - started) * 1000

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "strategies": len({chunk.strategy for chunk in self.chunks.values()}),
            "chunks_size": "in-memory",
        }

