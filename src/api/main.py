"""FastAPI retrieval service and OpenAPI documentation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.db import Database
from src.db_memory import MemoryDatabase
from src.db_protocol import DatabaseLike
from src.exceptions import ApexRagError, RetrievalError
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import build_embedder
from src.ingestion.loader import load_documents
from src.retrieval.pipeline import RetrievalPipeline

RetrievalModeName = Literal["dense_only", "sparse_only", "hybrid", "hybrid_rerank"]


class SearchRequest(BaseModel):
    """Search request accepted by the API."""

    query: str = Field(min_length=1, max_length=2_000, examples=["balance transfer fee"])
    top_k: int = Field(default=5, ge=1, le=20)
    mode: RetrievalModeName = "hybrid"
    use_metadata: bool = Field(
        default=False,
        description="Enable card-tier/year filtering and metadata heuristic reranking.",
    )


class SearchHitResponse(BaseModel):
    """One retrieved policy document."""

    doc_id: str
    title: str
    content: str
    score: float
    card_tier: str
    effective_year: str
    source: str


class ParsedQueryResponse(BaseModel):
    card_tier: str | None = None
    effective_year: str | None = None
    policy_code: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    parsed: ParsedQueryResponse
    hits: list[SearchHitResponse]
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    backend: str
    embedder: str
    reranker: str
    stats: dict[str, object]


def _build_memory_service(settings: Settings) -> tuple[MemoryDatabase, RetrievalPipeline]:
    """Load the supplied corpus into the offline backend at API startup."""

    documents = load_documents(settings.data_dir / "kb_documents.json")
    chunks = chunk_documents(documents, "document")
    texts = [chunk.text for chunk in chunks]
    embedder = build_embedder(settings, texts=texts)
    vectors = embedder.embed(texts)
    database = MemoryDatabase(embedder.dim)
    database.init_schema()
    database.upsert_documents(documents)
    database.upsert_chunks(chunks, vectors)
    pipeline = RetrievalPipeline(database=database, embedder=embedder, settings=settings)
    return database, pipeline


def _build_service(settings: Settings) -> tuple[DatabaseLike, RetrievalPipeline]:
    """Build the configured storage backend and retrieval pipeline."""

    if settings.backend == "memory":
        return _build_memory_service(settings)

    embedder = build_embedder(settings)
    database: DatabaseLike = Database(
        dsn=settings.postgres_dsn,
        embedding_dim=embedder.dim,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    database.init_schema()
    pipeline = RetrievalPipeline(database=database, embedder=embedder, settings=settings)
    return database, pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model/database once for the lifetime of the API process."""

    settings = get_settings()
    database, pipeline = _build_service(settings)
    app.state.settings = settings
    app.state.database = database
    app.state.pipeline = pipeline
    yield
    database.close()


app = FastAPI(
    title="Apex RAG Retrieval API",
    version="0.1.0",
    description="Hybrid retrieval API for the Apex Card Services policy knowledge base.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse, tags=["operations"])
def healthz() -> HealthResponse:
    """Return service and storage status."""

    database: DatabaseLike = app.state.database
    try:
        stats = database.stats()
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database unavailable",
        ) from exc
    return HealthResponse(
        status="ok",
        backend=app.state.settings.backend,
        embedder=app.state.pipeline.embedder.name,
        reranker=app.state.pipeline.reranker.name,
        stats={str(key): value for key, value in stats.items()},
    )


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["retrieval"],
    summary="Retrieve policy evidence for a support question",
)
def search(request: SearchRequest) -> SearchResponse:
    """Run dense/sparse retrieval, optional metadata rules and reranking."""

    try:
        result = app.state.pipeline.search(
            request.query,
            mode=request.mode,
            top_k=request.top_k,
            use_metadata=request.use_metadata,
        )
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (ApexRagError, psycopg.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="retrieval service unavailable",
        ) from exc

    return SearchResponse(
        query=request.query,
        mode=result.mode,
        parsed=ParsedQueryResponse(
            card_tier=result.parsed_query.card_tier,
            effective_year=result.parsed_query.effective_year,
            policy_code=result.parsed_query.policy_code,
        ),
        hits=[
            SearchHitResponse(
                doc_id=hit.doc_id,
                title=hit.title,
                content=hit.content,
                score=hit.score,
                card_tier=hit.card_tier,
                effective_year=hit.effective_year,
                source=hit.source,
            )
            for hit in result.hits
        ],
        timings_ms={str(key): round(value, 3) for key, value in result.timings_ms.items()},
    )

