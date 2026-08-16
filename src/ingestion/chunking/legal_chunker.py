"""Legal-aware chunking for Indian legal documents.

Follows the spec (Section 12):
  Acts:         Act → Chapter → Part → Section → Subsection → Clause
  Constitution: Constitution → Part → Chapter → Article → Clause
  Judgments:    Case → Page → Paragraph

Each chunk retains parent structure, neighboring chunk IDs, and metadata.
"""

from __future__ import annotations

import re
import logging
from typing import Any

from src.ingestion.models import TextChunk, DocumentType

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_MAX_CHUNK_SIZE = 1500   # chars — keeps chunks retrievable
DEFAULT_MIN_CHUNK_SIZE = 100    # chars — avoid micro-fragments
DEFAULT_OVERLAP = 100           # chars — context continuity


# ── Section/Article detection patterns ───────────────────────────────────

# Indian Acts: "Section 42.", "Sec. 103", "S. 12"
SECTION_PATTERN = re.compile(
    r"^(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?)\b[.\s]",
    re.IGNORECASE | re.MULTILINE,
)

# Constitution: "Article 21.", "Art. 19"
ARTICLE_PATTERN = re.compile(
    r"^(?:Article|Art\.)\s*(\d+[A-Z]?)\b[.\s]",
    re.IGNORECASE | re.MULTILINE,
)

# Chapter: "CHAPTER I", "Chapter XII"
CHAPTER_PATTERN = re.compile(
    r"^(?:CHAPTER|Chapter)\s+([IVXLCDM]+|\d+)",
    re.MULTILINE,
)

# Part: "PART I", "Part III"
PART_PATTERN = re.compile(
    r"^(?:PART|Part)\s+([IVXLCDM]+|\d+)",
    re.MULTILINE,
)

# Paragraph numbering in judgments: "42.", "para 42", "(42)"
PARAGRAPH_PATTERN = re.compile(
    r"^(?:\(?\d+\)?\.?\s)",
    re.MULTILINE,
)


# ── Core chunkers ───────────────────────────────────────────────────────


def chunk_act(
    document_id: str,
    text: str,
    source_id: str | None = None,
    source_url: str | None = None,
    authority_level: int = 0,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[TextChunk]:
    """Chunk an Act by sections, falling back to fixed-size if no sections found."""
    sections = _split_by_pattern(text, SECTION_PATTERN)

    if not sections:
        logger.info("No section markers found in Act %s — using paragraph chunking", document_id)
        return chunk_by_paragraphs(
            document_id, text, source_id, source_url, authority_level, max_chunk_size
        )

    chunks: list[TextChunk] = []
    for i, (section_num, section_text) in enumerate(sections):
        # Large sections get sub-chunked
        sub_chunks = _subdivide_if_needed(section_text, max_chunk_size)
        for j, sub_text in enumerate(sub_chunks):
            chunk_id = f"{document_id}_SEC{section_num}_{j}" if len(sub_chunks) > 1 else f"{document_id}_SEC{section_num}"
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=sub_text.strip(),
                section_number=section_num,
                source_id=source_id,
                source_url=source_url,
                authority_level=authority_level,
            ))

    _link_neighbors(chunks)
    logger.info("Created %d chunks from Act %s (%d sections)", len(chunks), document_id, len(sections))
    return chunks


def chunk_constitution(
    document_id: str,
    text: str,
    source_id: str | None = None,
    source_url: str | None = None,
    authority_level: int = 5,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[TextChunk]:
    """Chunk the Constitution by articles."""
    articles = _split_by_pattern(text, ARTICLE_PATTERN)

    if not articles:
        logger.info("No article markers found — using paragraph chunking")
        return chunk_by_paragraphs(
            document_id, text, source_id, source_url, authority_level, max_chunk_size
        )

    chunks: list[TextChunk] = []
    for i, (article_num, article_text) in enumerate(articles):
        sub_chunks = _subdivide_if_needed(article_text, max_chunk_size)
        for j, sub_text in enumerate(sub_chunks):
            chunk_id = f"{document_id}_ART{article_num}_{j}" if len(sub_chunks) > 1 else f"{document_id}_ART{article_num}"
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=sub_text.strip(),
                article_number=article_num,
                source_id=source_id,
                source_url=source_url,
                authority_level=authority_level,
            ))

    _link_neighbors(chunks)
    logger.info("Created %d chunks from Constitution (%d articles)", len(chunks), len(articles))
    return chunks


def chunk_judgment(
    document_id: str,
    pages: list[dict[str, Any]],
    source_id: str | None = None,
    source_url: str | None = None,
    authority_level: int = 5,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[TextChunk]:
    """Chunk a judgment by page then by paragraph boundaries."""
    chunks: list[TextChunk] = []

    for page_info in pages:
        page_num = page_info["page_number"]
        page_text = page_info["text"].strip()
        if not page_text:
            continue

        paragraphs = _split_paragraphs(page_text)

        current_chunk = ""
        para_start = 1
        para_idx = 0

        for para in paragraphs:
            para_idx += 1
            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                chunk_id = f"{document_id}_P{page_num}_R{para_start}"
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=current_chunk.strip(),
                    page_start=page_num,
                    page_end=page_num,
                    paragraph_start=para_start,
                    paragraph_end=para_idx - 1,
                    source_id=source_id,
                    source_url=source_url,
                    authority_level=authority_level,
                ))
                current_chunk = para
                para_start = para_idx
            else:
                current_chunk += "\n" + para if current_chunk else para

        if current_chunk.strip():
            chunk_id = f"{document_id}_P{page_num}_R{para_start}"
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=current_chunk.strip(),
                page_start=page_num,
                page_end=page_num,
                paragraph_start=para_start,
                paragraph_end=para_idx,
                source_id=source_id,
                source_url=source_url,
                authority_level=authority_level,
            ))

    _link_neighbors(chunks)
    logger.info("Created %d chunks from judgment %s (%d pages)", len(chunks), document_id, len(pages))
    return chunks


def chunk_by_paragraphs(
    document_id: str,
    text: str,
    source_id: str | None = None,
    source_url: str | None = None,
    authority_level: int = 0,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[TextChunk]:
    """General-purpose chunking by paragraph with size limits."""
    paragraphs = _split_paragraphs(text)
    chunks: list[TextChunk] = []
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            chunks.append(TextChunk(
                chunk_id=f"{document_id}_C{chunk_index}",
                document_id=document_id,
                text=current_chunk.strip(),
                source_id=source_id,
                source_url=source_url,
                authority_level=authority_level,
            ))
            chunk_index += 1
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(TextChunk(
            chunk_id=f"{document_id}_C{chunk_index}",
            document_id=document_id,
            text=current_chunk.strip(),
            source_id=source_id,
            source_url=source_url,
            authority_level=authority_level,
        ))

    _link_neighbors(chunks)
    return chunks


def chunk_document(
    document_id: str,
    document_type: DocumentType,
    text: str | None = None,
    pages: list[dict[str, Any]] | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
    authority_level: int = 0,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[TextChunk]:
    """Route to the appropriate chunker based on document type."""
    if document_type == DocumentType.CONSTITUTION:
        return chunk_constitution(
            document_id, text or "", source_id, source_url, authority_level, max_chunk_size
        )
    elif document_type == DocumentType.JUDGMENT:
        if pages:
            return chunk_judgment(
                document_id, pages, source_id, source_url, authority_level, max_chunk_size
            )
        elif text:
            return chunk_by_paragraphs(
                document_id, text, source_id, source_url, authority_level, max_chunk_size
            )
    elif document_type in (DocumentType.ACT, DocumentType.SECTION, DocumentType.RULE):
        return chunk_act(
            document_id, text or "", source_id, source_url, authority_level, max_chunk_size
        )

    # Fallback
    return chunk_by_paragraphs(
        document_id, text or "", source_id, source_url, authority_level, max_chunk_size
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Split text by a regex pattern, returning [(match_group, text), ...]."""
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    result: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_num = match.group(1)
        section_text = text[start:end]
        result.append((section_num, section_text))

    return result


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double newline or numbered paragraphs)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _subdivide_if_needed(text: str, max_size: int) -> list[str]:
    """Split a large block into sub-chunks at paragraph boundaries."""
    if len(text) <= max_size:
        return [text]

    paragraphs = _split_paragraphs(text)
    sub_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > max_size and current:
            sub_chunks.append(current)
            current = para
        else:
            current += "\n\n" + para if current else para

    if current.strip():
        sub_chunks.append(current)

    return sub_chunks if sub_chunks else [text]


def _link_neighbors(chunks: list[TextChunk]) -> None:
    """Set prev_chunk_id and next_chunk_id on each chunk."""
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk.prev_chunk_id = chunks[i - 1].chunk_id
        if i < len(chunks) - 1:
            chunk.next_chunk_id = chunks[i + 1].chunk_id
