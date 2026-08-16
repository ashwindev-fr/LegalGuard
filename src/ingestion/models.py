"""Pydantic models for legal documents, chunks, and source metadata.

These are the canonical data models used throughout ingestion, storage,
retrieval, and generation.  Every field follows the spec (Sections 9-11).
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────


class DocumentType(str, Enum):
    CONSTITUTION = "CONSTITUTION"
    ACT = "ACT"
    SECTION = "SECTION"
    RULE = "RULE"
    REGULATION = "REGULATION"
    NOTIFICATION = "NOTIFICATION"
    ORDER = "ORDER"
    ORDINANCE = "ORDINANCE"
    AMENDMENT = "AMENDMENT"
    JUDGMENT = "JUDGMENT"
    BILL = "BILL"
    REPORT = "REPORT"


class DocumentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REPEALED = "REPEALED"
    AMENDED = "AMENDED"
    SUPERSEDED = "SUPERSEDED"
    UNKNOWN = "UNKNOWN"


class RelationStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ExtractionMethod(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm_relation_extraction"
    MANUAL = "manual"


# ── Source metadata ──────────────────────────────────────────────────────


class SourceMetadata(BaseModel):
    source_id: str
    name: str
    url: str | None = None
    base_url: str | None = None
    authority_level: int = Field(ge=0, le=5, default=0)
    license_note: str | None = None


# ── Standard document model (spec §9) ───────────────────────────────────


class LegalDocument(BaseModel):
    document_id: str
    document_type: DocumentType
    title: str
    short_title: str | None = None
    year: int | None = None
    language: str = "en"
    source: SourceMetadata
    retrieved_at: datetime | None = None
    effective_from: date | None = None
    effective_until: date | None = None
    status: DocumentStatus = DocumentStatus.UNKNOWN
    content_hash: str | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    file_path: str | None = None

    @staticmethod
    def generate_id(source_id: str, title: str, url: str | None = None) -> str:
        """Create a deterministic document ID from stable source info."""
        key = f"{source_id}:{url or ''}:{title}"
        return hashlib.sha256(key.encode()).hexdigest()[:24]


# ── Judgment document model (spec §10) ───────────────────────────────────


class JudgmentDocument(LegalDocument):
    document_type: DocumentType = DocumentType.JUDGMENT
    case_name: str | None = None
    court: str | None = None
    jurisdiction: str = "India"
    judgment_date: date | None = None
    citation: str | None = None
    neutral_citation: str | None = None
    bench: list[str] = Field(default_factory=list)


# ── Chunk model (spec §11) ──────────────────────────────────────────────


class TextChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    section_number: str | None = None
    chapter: str | None = None
    article_number: str | None = None
    source_url: str | None = None
    source_id: str | None = None
    authority_level: int = 0
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    # Neighbor tracking
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    @staticmethod
    def generate_id(document_id: str, page: int | None, paragraph: int | None, index: int) -> str:
        """Create a deterministic chunk ID."""
        suffix = f"P{page or 0}_R{paragraph or index}"
        return f"{document_id}_{suffix}"


# ── Extracted relation (spec §23-24) ─────────────────────────────────────


class ExtractedRelation(BaseModel):
    relation_id: str
    source_chunk_id: str
    subject_id: str
    subject_label: str
    relation_type: str
    object_id: str
    object_label: str
    extraction_method: ExtractionMethod
    model_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    status: RelationStatus = RelationStatus.PENDING
    reviewed: bool = False
    evidence_text: str | None = None


# ── Evidence item (spec §33) ────────────────────────────────────────────


class EvidenceItem(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    title: str | None = None
    document_type: str | None = None
    case_name: str | None = None
    court: str | None = None
    citation: str | None = None
    date: str | None = None
    section: str | None = None
    page: int | None = None
    paragraph: int | None = None
    text: str
    source_url: str | None = None
    authority_level: int = 0


# ── Evidence packet (spec §33) ──────────────────────────────────────────


class GraphFact(BaseModel):
    subject: str
    relation: str
    object: str
    evidence_id: str | None = None


class EvidencePacket(BaseModel):
    query: str
    query_analysis: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    graph_facts: list[GraphFact] = Field(default_factory=list)


# ── Claim / Answer models (spec §35, §73) ───────────────────────────────


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    claim_type: str | None = None
    support_status: str | None = None  # SUPPORTED / NOT_SUPPORTED / UNCERTAIN


class ConfidenceScore(BaseModel):
    label: str  # high / medium / low
    reasons: list[str] = Field(default_factory=list)


class LegalAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[EvidenceItem] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    temporal_notes: list[str] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(
        default_factory=lambda: ConfidenceScore(label="low", reasons=[])
    )
    disclaimer: str = (
        "This system provides AI-assisted legal information and research support "
        "based on retrieved sources. It is not a lawyer, does not provide guaranteed "
        "legal advice, and should not replace advice from a qualified legal professional. "
        "Always verify important legal claims against the current authoritative source."
    )
