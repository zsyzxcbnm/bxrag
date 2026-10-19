"""Centralized application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    postgres_dsn: str = "postgresql://apex:apex@localhost:5432/apex"
    backend: Literal["postgres", "memory"] = "postgres"

    embedding_provider: Literal["local", "openai", "sentence-transformers"] = "local"
    embedding_model: str = "local-tfidf-svd"
    embedding_dim: int = 128
    openai_api_key: str | None = None
    embedding_device: str | None = None
    embedding_query_prompt: str | None = None

    reranker_provider: Literal["heuristic", "cross-encoder"] = "heuristic"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str | None = None
    reranker_fp16: bool = False

    dense_limit: int = 50
    sparse_limit: int = 50
    rerank_candidates: int = 20
    final_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 0.4
    sparse_weight: float = 0.6

    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 80

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    artifacts_dir: Path = Path("data/artifacts")

    def ensure_directories(self) -> None:
        """Create output directories used by the pipeline."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "figures").mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""

    settings = Settings()
    settings.ensure_directories()
    return settings

