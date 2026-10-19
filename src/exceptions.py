"""Application-specific exception hierarchy."""


class ApexRagError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(ApexRagError):
    """Invalid or missing application configuration."""


class IngestionError(ApexRagError):
    """Invalid source data or failed ingestion."""


class StorageError(ApexRagError):
    """Database/schema/query failure that is safe to report to the caller."""


class RetrievalError(ApexRagError):
    """Invalid retrieval request or pipeline state."""


class EmbeddingError(ApexRagError):
    """Embedding model failure."""


class RerankerError(ApexRagError):
    """Reranker model failure when no fallback is available."""
