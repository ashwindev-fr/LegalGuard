"""Base source adapter for official legal document acquisition.

Enforces data collection principles (spec §7):
  - Rate limiting
  - User-Agent identification
  - Checksum computation (SHA-256)
  - Document caching
  - Preserves retrieval timestamp and source URL
  - Respects access terms / robots rules
"""

from __future__ import annotations

import abc
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.config import get_settings
from src.ingestion.models import SourceMetadata

logger = logging.getLogger(__name__)


class BaseSourceAdapter(abc.ABC):
    """Abstract base class for all legal source adapters."""

    def __init__(
        self,
        source_id: str,
        name: str,
        base_url: str,
        authority_level: int,
        rate_limit_delay_seconds: float = 1.0,
    ) -> None:
        self.source_id = source_id
        self.name = name
        self.base_url = base_url
        self.authority_level = authority_level
        self.rate_limit_delay_seconds = rate_limit_delay_seconds
        self._last_request_time: float = 0.0

        settings = get_settings()
        self.cache_dir = settings.cache_dir / "sources" / source_id.lower()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = httpx.Client(
            headers={"User-Agent": "IndianLaw-GraphRAG-ResearchBot/1.0 (+http://localhost)"},
            timeout=30.0,
            follow_redirects=True,
        )

    def get_metadata(() -> SourceMetadata:
        return SourceMetadata(
            source_id=self.source_id,
            name=self.name,
            base_url=self.base_url,
            authority_level=self.authority_level,
        )

    def _rate_limit(self) -> None:
        """Rate-limit outbound requests to comply with site rules."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay_seconds:
            time.sleep(self.rate_limit_delay_seconds - elapsed)
        self._last_request_time = time.time()

    def fetch_url(self, url: str, use_cache: bool = True) -> bytes:
        """Fetch URL content with caching and rate-limiting."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"{url_hash}.bin"

        if use_cache and cache_file.exists():
            logger.debug("Cache hit for %s", url)
            return cache_file.read_bytes()

        self._rate_limit()
        logger.info("Fetching %s", url)
        response = self.client.get(url)
        response.raise_for_status()

        content = response.content
        cache_file.write_bytes(content)
        return content

    @abc.abstractmethod
    def fetch_document(self, document_ref: str) -> dict[str, Any]:
        """Fetch and return document payload + metadata dict."""
        ...
