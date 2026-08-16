"""Reranker — interface and implementations (spec §80).

Provides a pluggable reranker with a cross-encoder default and a no-op fallback.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.config import get_settings
from src.retrieval.models import RetrievalCandidate

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Interface for rerankers (spec §80)."""

    def rerank(self, query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        ...


class CrossEncoderReranker:
    """Reranker using a cross-encoder model (e.g. ms-marco-MiniLM)."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder
        import torch

        settings = get_settings()
        self._model_name = model_name or settings.reranker_model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading reranker: %s on %s", self._model_name, device)
        self._model = CrossEncoder(self._model_name, device=device)
        logger.info("Reranker loaded")

    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        """Rerank candidates using cross-encoder scores."""
        if not candidates:
            return candidates

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.score_reranker = float(score)

        candidates.sort(key=lambda c: c.score_reranker or 0.0, reverse=True)
        logger.info("Reranked %d candidates", len(candidates))
        return candidates


class NoOpReranker:
    """Pass-through reranker — returns candidates unchanged."""

    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        return candidates


def get_reranker(use_reranker: bool = True) -> Reranker:
    """Factory for the configured reranker."""
    if not use_reranker:
        return NoOpReranker()
    try:
        return CrossEncoderReranker()
    except Exception as e:
        logger.warning("Failed to load reranker, using no-op: %s", e)
        return NoOpReranker()
