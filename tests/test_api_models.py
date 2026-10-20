"""API request/response contract tests."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.api.main import SearchRequest, _build_service
from src.config import Settings


def test_search_request_defaults() -> None:
    request = SearchRequest(query="balance transfer fee")
    assert request.top_k == 5
    assert request.mode == "hybrid"
    assert request.use_metadata is False


def test_search_request_rejects_invalid_top_k() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="x", top_k=50)


def test_search_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="")


def test_memory_backend_loads_service(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEmbedder:
        name = "fake"
        dim = 2

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), self.dim), dtype=np.float32)

        def embed_query(self, query: str) -> np.ndarray:
            del query
            return np.ones(self.dim, dtype=np.float32)

    def fake_build_embedder(settings: Settings, texts: list[str] | None = None) -> FakeEmbedder:
        del settings, texts
        return FakeEmbedder()

    monkeypatch.setattr("src.api.main.build_embedder", fake_build_embedder)
    settings = Settings(backend="memory", embedding_provider="local")
    database, pipeline = _build_service(settings)

    assert database.stats()["documents"] > 0
    assert database.stats()["chunks"] > 0
    result = pipeline.search("balance transfer fee", top_k=10)
    assert len(result.hits) == 10
    database.close()
