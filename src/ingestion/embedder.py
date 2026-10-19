"""Embedding providers with an offline, reproducible fallback."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import Settings
from src.exceptions import ConfigurationError, EmbeddingError


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used by ingestion and retrieval."""

    name: str
    dim: int

    def fit(self, texts: Sequence[str]) -> None:
        """Fit the embedding model when the provider requires training."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n_texts, dim)`` float32 matrix."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query with provider-specific instructions."""
        ...

    def save(self, path: Path) -> None:
        """Persist model artifacts when applicable."""
        ...


class LocalTfidfSvdEmbedder:
    """Offline dense embedding based on TF-IDF followed by LSA.

    This is intentionally lightweight: the evaluation contains only 150 short
    documents and may run without network access or an API key. Swap in
    ``OpenAIEmbedder`` or a sentence-transformer for production use.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = int(dim)
        self.name = "local-tfidf-svd"
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise EmbeddingError("Cannot fit embedder on an empty corpus")
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            min_df=1,
            max_features=30_000,
        )
        matrix = self._vectorizer.fit_transform(texts)
        max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1, self.dim)
        if max_components < 2:
            raise EmbeddingError("Corpus is too small for a dense LSA embedding")
        self.dim = max_components
        self._svd = TruncatedSVD(n_components=self.dim, random_state=42)
        self._svd.fit(matrix)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise EmbeddingError("Embedder has not been fitted or loaded")
        matrix = self._vectorizer.transform(texts)
        vectors = self._svd.transform(matrix)
        return cast(np.ndarray, np.asarray(normalize(vectors), dtype=np.float32))

    def embed_query(self, query: str) -> np.ndarray:
        return cast(np.ndarray, self.embed([query])[0])

    def save(self, path: Path) -> None:
        if self._vectorizer is None or self._svd is None:
            raise EmbeddingError("Cannot save an unfitted embedder")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "dim": self.dim,
                "vectorizer": self._vectorizer,
                "svd": self._svd,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LocalTfidfSvdEmbedder:
        payload = joblib.load(path)
        embedder = cls(dim=int(payload["dim"]))
        embedder._vectorizer = payload["vectorizer"]
        embedder._svd = payload["svd"]
        return embedder


class OpenAIEmbedder:
    """OpenAI embedding provider with batched requests."""

    def __init__(self, model: str, api_key: str | None, dim: int) -> None:
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for the openai embedding provider")
        from openai import OpenAI

        self.name = model
        self.dim = int(dim)
        self._client = OpenAI(api_key=api_key)

    def fit(self, texts: Sequence[str]) -> None:
        del texts

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = list(texts[start : start + 64])
            response = self._client.embeddings.create(model=self.name, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return cast(np.ndarray, np.asarray(normalize(np.asarray(vectors)), dtype=np.float32))

    def embed_query(self, query: str) -> np.ndarray:
        return cast(np.ndarray, self.embed([query])[0])

    def save(self, path: Path) -> None:
        del path


class SentenceTransformerEmbedder:
    """Local Sentence-Transformers embedding with Qwen3 query instructions."""

    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_seq_length: int = 512,
        query_prompt: str | None = None,
    ) -> None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = max_seq_length
        self.device = str(self._model.device)
        self.query_prompt = query_prompt
        if self.query_prompt is None and "bge" in model_name.lower():
            self.query_prompt = self._BGE_QUERY_PREFIX
        dimension_getter = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        dimension = dimension_getter()
        if dimension is None:
            raise EmbeddingError(f"Could not determine embedding dimension for {model_name}")
        self.dim = int(dimension)

    def fit(self, texts: Sequence[str]) -> None:
        del texts

    def _encode(self, texts: Sequence[str], prompt_name: str | None = None) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name=prompt_name,
        )
        return cast(np.ndarray, np.asarray(vectors, dtype=np.float32))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        if self.query_prompt:
            return cast(
                np.ndarray,
                self._encode([f"{self.query_prompt}{query}"])[0],
            )
        prompts = getattr(self._model, "prompts", {})
        if "query" in prompts:
            return cast(np.ndarray, self._encode([query], prompt_name="query")[0])
        return cast(np.ndarray, self._encode([query])[0])

    def save(self, path: Path) -> None:
        del path


def build_embedder(
    settings: Settings,
    texts: Sequence[str] | None = None,
    force_fit: bool = False,
) -> Embedder:
    """Build the configured embedder, loading or training local artifacts."""

    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            dim=settings.embedding_dim,
        )

    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device or None,
            query_prompt=settings.embedding_query_prompt or None,
        )

    artifact_path = settings.artifacts_dir / "local_embedder.joblib"
    if artifact_path.exists() and not force_fit:
        return LocalTfidfSvdEmbedder.load(artifact_path)
    if texts is None:
        raise ConfigurationError("Training texts are required to build a new local embedder")
    embedder = LocalTfidfSvdEmbedder(dim=settings.embedding_dim)
    embedder.fit(texts)
    embedder.save(artifact_path)
    return embedder
