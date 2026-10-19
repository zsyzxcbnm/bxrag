"""Command-line entry points for ingestion, retrieval and evaluation."""

from __future__ import annotations

import argparse
import json
import logging

from src.config import Settings, get_settings
from src.db import Database
from src.db_memory import MemoryDatabase
from src.db_protocol import DatabaseLike
from src.eval.report import generate_report
from src.eval.runner import EvalRunner, write_evaluation
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import Embedder, build_embedder
from src.ingestion.loader import load_documents, load_queries
from src.retrieval.pipeline import RetrievalPipeline

LOGGER = logging.getLogger("apex-rag")


def _database(settings: Settings, embedder: Embedder) -> DatabaseLike:
    return Database(
        dsn=settings.postgres_dsn,
        embedding_dim=embedder.dim,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )


def _ingest(
    settings: Settings,
    strategy: str,
    refit: bool,
    reset: bool,
    backend: str,
) -> DatabaseLike:
    documents = load_documents(settings.data_dir / "kb_documents.json")
    queries = load_queries(
        settings.data_dir / "eval_queries.json",
        known_doc_ids={document.doc_id for document in documents},
    )
    LOGGER.info("Loaded %d documents and %d queries", len(documents), len(queries))

    if strategy == "all":
        chunks = []
        for item in ("document", "fixed", "semantic"):
            chunks.extend(chunk_documents(documents, item))
    else:
        chunks = chunk_documents(documents, strategy)
    if not chunks:
        raise RuntimeError("No chunks were produced")
    LOGGER.info("Built %d chunks with strategy=%s", len(chunks), strategy)

    texts = [chunk.text for chunk in chunks]
    embedder = build_embedder(settings, texts=texts, force_fit=refit)
    LOGGER.info("Embedding with %s (%d dimensions)", embedder.name, embedder.dim)
    embeddings = embedder.embed(texts)

    if backend == "memory":
        database: DatabaseLike = MemoryDatabase(embedder.dim)
    else:
        database = _database(settings, embedder)
    database.init_schema(reset=reset)
    database.upsert_documents(documents)
    database.upsert_chunks(chunks, embeddings)
    LOGGER.info("Database stats: %s", database.stats())
    return database


def _load_embedder(settings: Settings) -> Embedder:
    return build_embedder(settings)


def _build_memory_database(
    settings: Settings,
    embedder: Embedder,
    strategy: str = "document",
) -> MemoryDatabase:
    documents = load_documents(settings.data_dir / "kb_documents.json")
    chunks = chunk_documents(documents, strategy)
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    database = MemoryDatabase(embedder.dim)
    database.init_schema()
    database.upsert_documents(documents)
    database.upsert_chunks(chunks, embeddings)
    return database


def _evaluate(
    settings: Settings,
    repeats: int,
    backend: str,
    database: DatabaseLike | None = None,
) -> None:
    documents = load_documents(settings.data_dir / "kb_documents.json")
    queries = load_queries(
        settings.data_dir / "eval_queries.json",
        known_doc_ids={document.doc_id for document in documents},
    )
    embedder = _load_embedder(settings)
    if database is None:
        database = (
            _build_memory_database(settings, embedder)
            if backend == "memory"
            else _database(settings, embedder)
        )
    pipeline = RetrievalPipeline(database=database, embedder=embedder, settings=settings)
    runner = EvalRunner(pipeline=pipeline, settings=settings, repeats=repeats)
    payload = runner.run(queries)
    write_evaluation(payload, settings.reports_dir / "results.json")
    report_path = generate_report(payload, settings.reports_dir)
    LOGGER.info("Wrote %s", report_path)


def _search(settings: Settings, query: str, mode: str, top_k: int, backend: str) -> None:
    embedder = _load_embedder(settings)
    database = (
        _build_memory_database(settings, embedder)
        if backend == "memory"
        else _database(settings, embedder)
    )
    pipeline = RetrievalPipeline(database=database, embedder=embedder, settings=settings)
    result = pipeline.search(query, mode=mode, top_k=top_k)  # type: ignore[arg-type]
    print(
        json.dumps(
            {
                "query": query,
                "mode": result.mode,
                "parsed": {
                    "card_tier": result.parsed_query.card_tier,
                    "effective_year": result.parsed_query.effective_year,
                    "policy_code": result.parsed_query.policy_code,
                },
                "timings_ms": result.timings_ms,
                "hits": [
                    {
                        "doc_id": hit.doc_id,
                        "title": hit.title,
                        "score": hit.score,
                        "card_tier": hit.card_tier,
                        "effective_year": hit.effective_year,
                        "content": hit.content,
                    }
                    for hit in result.hits
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _init_db(settings: Settings, reset: bool) -> None:
    try:
        embedder = _load_embedder(settings)
        dim = embedder.dim
    except (FileNotFoundError, ValueError):
        dim = settings.embedding_dim
    database = Database(
        dsn=settings.postgres_dsn,
        embedding_dim=dim,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    database.init_schema(reset=reset)
    LOGGER.info("Initialized schema with %d dimensions", dim)


def _add_backend_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("postgres", "memory"),
        default=None,
        help="Override the configured storage backend",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apex hybrid retrieval and RAG evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create tables and indexes")
    init_db.add_argument("--reset", action="store_true", help="Drop existing tables first")

    ingest = subparsers.add_parser("ingest", help="Load, embed and upsert the knowledge base")
    ingest.add_argument(
        "--strategy",
        choices=("document", "fixed", "semantic", "all"),
        default="document",
    )
    ingest.add_argument("--refit", action="store_true", help="Retrain the local embedder")
    ingest.add_argument("--reset", action="store_true", help="Drop existing tables first")
    _add_backend_argument(ingest)

    index = subparsers.add_parser("index", help="Create indexes for an existing database")
    index.add_argument("--reset", action="store_true", help="Drop existing tables first")

    eval_parser = subparsers.add_parser("eval", help="Run all retrieval ablations")
    eval_parser.add_argument("--repeats", type=int, default=3)
    _add_backend_argument(eval_parser)

    search = subparsers.add_parser("search", help="Run a single query")
    search.add_argument("query")
    search.add_argument(
        "--mode",
        choices=("dense_only", "sparse_only", "hybrid", "hybrid_rerank"),
        default="hybrid_rerank",
    )
    search.add_argument("--top-k", type=int, default=5)
    _add_backend_argument(search)

    stats = subparsers.add_parser("stats", help="Show document and chunk counts")
    _add_backend_argument(stats)

    reset = subparsers.add_parser("reset-db", help="Drop and recreate all tables")
    reset.add_argument("--yes", action="store_true", help="Confirm destructive reset")

    all_parser = subparsers.add_parser(
        "all",
        help="Ingest the default dataset and run evaluation",
    )
    _add_backend_argument(all_parser)
    return parser


def main() -> None:
    """Parse arguments and dispatch the requested command."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    settings = get_settings()

    if args.command == "init-db":
        _init_db(settings, reset=args.reset)
    elif args.command == "ingest":
        _ingest(
            settings,
            args.strategy,
            refit=args.refit,
            reset=args.reset,
            backend=args.backend or settings.backend,
        )
    elif args.command == "index":
        _init_db(settings, reset=args.reset)
    elif args.command == "eval":
        _evaluate(settings, repeats=args.repeats, backend=args.backend or settings.backend)
    elif args.command == "search":
        _search(
            settings,
            args.query,
            args.mode,
            args.top_k,
            backend=args.backend or settings.backend,
        )
    elif args.command == "stats":
        backend = args.backend or settings.backend
        embedder = _load_embedder(settings)
        database = (
            _build_memory_database(settings, embedder)
            if backend == "memory"
            else _database(settings, embedder)
        )
        print(json.dumps(database.stats(), indent=2))
    elif args.command == "reset-db":
        if not args.yes:
            raise SystemExit("Refusing to reset without --yes")
        _init_db(settings, reset=True)
    elif args.command == "all":
        backend = args.backend or settings.backend
        database = _ingest(
            settings,
            "document",
            refit=False,
            reset=False,
            backend=backend,
        )
        _evaluate(settings, repeats=3, backend=backend, database=database)


if __name__ == "__main__":
    main()
