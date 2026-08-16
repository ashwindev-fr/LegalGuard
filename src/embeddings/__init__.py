"""Embedding provider — interface and implementations (spec §79).

Supports swappable embedding backends. Stores model name and dimension
in metadata for reproducibility.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.config import get_settings
from src.graph.repository import GraphRepository

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...


class SentenceTransformerProvider:
    """Embedding provider using sentence-transformers (e.g. BAAI/bge-m3)."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading embedding model: %s on %s", self._model_name, self._device)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dimension=%d)", self._dimension)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        embeddings = self._model.encode(texts, show_progress_bar=len(texts) > 10)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        embedding = self._model.encode(text)
        return embedding.tolist()


def get_embedding_provider() -> SentenceTransformerProvider:
    """Factory for the configured embedding provider."""
    settings = get_settings()
    return SentenceTransformerProvider(model_name=settings.embedding_model)


def embed_all_chunks(provider: EmbeddingProvider | None = None, batch_size: int = 32) -> int:
    """Embed all chunks that don't have embeddings yet.

    Returns the number of chunks embedded.
    """
    if provider is None:
        provider = get_embedding_provider()

    repo = GraphRepository()
    chunks = repo.get_chunks_without_embeddings()

    if not chunks:
        logger.info("All chunks already have embeddings")
        return 0

    logger.info("Embedding %d chunks with %s", len(chunks), provider.model_name)
    count = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = provider.embed_texts(texts)

        for chunk, embedding in zip(batch, embeddings):
            repo.set_chunk_embedding(chunk["id"], embedding)
            # Also store embedding metadata
            repo.merge_chunk(chunk["id"], {
                "embedding_model": provider.model_name,
                "embedding_dimension": provider.dimension,
            })
            count += 1

        logger.info("Embedded batch %d-%d / %d", i, i + len(batch), len(chunks))

    logger.info("Finished embedding %d chunks", count)
    return count
