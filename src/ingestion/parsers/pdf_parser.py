"""PDF text extraction using PyMuPDF (fitz).

Primary extraction method.  OCR is NOT used unless normal text extraction
fails and is flagged for manual review.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_pages(path: str | Path) -> list[dict[str, Any]]:
    """Extract text from each page of a PDF.

    Returns a list of dicts with:
      - page_number (1-indexed)
      - text (raw extracted text)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))
    pages: list[dict[str, Any]] = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_number": i + 1,
            "text": text,
        })

    doc.close()
    logger.info("Extracted %d pages from %s", len(pages), path.name)
    return pages


def extract_full_text(path: str | Path) -> str:
    """Extract and concatenate all text from a PDF."""
    pages = extract_pages(path)
    return "\n\n".join(p["text"] for p in pages)


def compute_file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of the raw file bytes."""
    path = Path(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha256


def detect_file_type(path: str | Path) -> str:
    """Return the file extension/type."""
    return Path(path).suffix.lower()


def is_text_extractable(path: str | Path) -> bool:
    """Check if meaningful text can be extracted (not scanned-image-only)."""
    try:
        pages = extract_pages(path)
        total_chars = sum(len(p["text"].strip()) for p in pages)
        # Heuristic: if less than 100 chars across all pages, likely needs OCR
        return total_chars > 100
    except Exception:
        return False


def extract_metadata(path: str | Path) -> dict[str, Any]:
    """Extract PDF metadata (title, author, creation date, etc.)."""
    path = Path(path)
    doc = fitz.open(str(path))
    meta = doc.metadata or {}
    doc.close()
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "creation_date": meta.get("creationDate", ""),
        "modification_date": meta.get("modDate", ""),
        "page_count": len(extract_pages(path)),
        "file_hash": compute_file_hash(path),
    }
