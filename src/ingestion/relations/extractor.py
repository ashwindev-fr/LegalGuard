"""Relation extraction — deterministic (Layer A) and LLM-assisted (Layer B).

Layer A: regex/rule-based extraction for CONTAINS, HAS_SECTION, CITES, MENTIONS.
Layer B: model-assisted extraction for INTERPRETS, FOLLOWS, DISTINGUISHES,
         OVERRULES, ESTABLISHES, APPLIES (initially stubbed — uses LLM when available).

Every extracted relation gets a validation status per spec §23.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from src.ingestion.entities.extractor import extract_entities, ExtractedEntity
from src.ingestion.models import (
    ExtractedRelation,
    ExtractionMethod,
    RelationStatus,
    TextChunk,
)

logger = logging.getLogger(__name__)


# ── Layer A: Deterministic extraction ────────────────────────────────────


def extract_deterministic_relations(
    chunk: TextChunk,
    document_type: str | None = None,
) -> list[ExtractedRelation]:
    """Extract relations using regex/rules from a single chunk."""
    relations: list[ExtractedRelation] = []
    entities = extract_entities(chunk.text)

    for entity in entities:
        if entity.entity_type == "Section":
            rel_id = _make_relation_id(chunk.chunk_id, "MENTIONS", f"SEC_{entity.value}")
            relations.append(ExtractedRelation(
                relation_id=rel_id,
                source_chunk_id=chunk.chunk_id,
                subject_id=chunk.chunk_id,
                subject_label="Chunk",
                relation_type="MENTIONS",
                object_id=f"SEC_{entity.value}",
                object_label="Section",
                extraction_method=ExtractionMethod.DETERMINISTIC,
                confidence=1.0,
                status=RelationStatus.ACCEPTED,
                evidence_text=entity.context,
            ))

        elif entity.entity_type == "Article":
            rel_id = _make_relation_id(chunk.chunk_id, "MENTIONS", f"ART_{entity.value}")
            relations.append(ExtractedRelation(
                relation_id=rel_id,
                source_chunk_id=chunk.chunk_id,
                subject_id=chunk.chunk_id,
                subject_label="Chunk",
                relation_type="MENTIONS",
                object_id=f"ART_{entity.value}",
                object_label="Article",
                extraction_method=ExtractionMethod.DETERMINISTIC,
                confidence=1.0,
                status=RelationStatus.ACCEPTED,
                evidence_text=entity.context,
            ))

        elif entity.entity_type == "Citation":
            rel_id = _make_relation_id(chunk.chunk_id, "CITES", entity.value)
            relations.append(ExtractedRelation(
                relation_id=rel_id,
                source_chunk_id=chunk.chunk_id,
                subject_id=chunk.document_id,
                subject_label="Case",
                relation_type="CITES",
                object_id=_normalize_citation_id(entity.value),
                object_label="Case",
                extraction_method=ExtractionMethod.DETERMINISTIC,
                confidence=0.9,
                status=RelationStatus.ACCEPTED,
                evidence_text=entity.context,
            ))

        elif entity.entity_type == "Court":
            rel_id = _make_relation_id(chunk.document_id, "DECIDED_BY", entity.value)
            relations.append(ExtractedRelation(
                relation_id=rel_id,
                source_chunk_id=chunk.chunk_id,
                subject_id=chunk.document_id,
                subject_label="Case",
                relation_type="DECIDED_BY",
                object_id=_normalize_court_id(entity.value),
                object_label="Court",
                extraction_method=ExtractionMethod.DETERMINISTIC,
                confidence=0.85,
                status=RelationStatus.ACCEPTED,
                evidence_text=entity.value,
            ))

    return relations


# ── Layer B: LLM-assisted extraction (stub) ──────────────────────────────

# These semantic relations require context understanding:
LLM_RELATION_TYPES = [
    "INTERPRETS",
    "FOLLOWS",
    "DISTINGUISHES",
    "OVERRULES",
    "ESTABLISHES",
    "APPLIES",
]


def extract_llm_relations(
    chunk: TextChunk,
    llm_callable: Any | None = None,
) -> list[ExtractedRelation]:
    """Extract semantic legal relations using an LLM.

    Currently stubbed — returns empty list when no LLM is available.
    When implemented, all LLM-extracted relations will have:
      - status = REVIEW_REQUIRED
      - extraction_method = LLM
      - confidence from model output

    The system NEVER treats these as canonical until validated.
    """
    if llm_callable is None:
        return []

    # TODO: Implement LLM-based relation extraction
    # The prompt should ask the model to identify whether the text:
    # - INTERPRETS a section/article
    # - FOLLOWS a prior case
    # - DISTINGUISHES a prior case
    # - OVERRULES a prior case
    # - ESTABLISHES a legal principle
    # - APPLIES a statutory provision
    #
    # All results should be returned with REVIEW_REQUIRED status.
    logger.debug(
        "LLM relation extraction not yet implemented for chunk %s",
        chunk.chunk_id,
    )
    return []


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_relation_id(subject: str, rel_type: str, obj: str) -> str:
    """Create a deterministic relation ID."""
    key = f"{subject}:{rel_type}:{obj}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def _normalize_citation_id(citation: str) -> str:
    """Normalize a citation string to an ID-safe form."""
    # Remove special characters, collapse spaces
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", citation)
    clean = re.sub(r"\s+", "_", clean.strip())
    return f"CITE_{clean[:50]}"


def _normalize_court_id(court_name: str) -> str:
    """Normalize a court name to an ID."""
    clean = re.sub(r"[^a-zA-Z\s]", "", court_name)
    clean = re.sub(r"\s+", "_", clean.strip().upper())
    return f"COURT_{clean}"
