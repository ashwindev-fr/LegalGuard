"""Manual document import — primary ingestion path for the demo corpus.

Supports importing PDFs and text files from the local filesystem into the
legal knowledge graph.  This is the recommended path when automated
scraping is not safe or appropriate.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.graph.repository import GraphRepository
from src.ingestion.chunking.legal_chunker import chunk_document
from src.ingestion.models import (
    DocumentStatus,
    DocumentType,
    LegalDocument,
    JudgmentDocument,
    SourceMetadata,
    TextChunk,
)
from src.ingestion.normalization.normalizer import normalize_legal_text
from src.ingestion.parsers.pdf_parser import (
    compute_file_hash,
    detect_file_type,
    extract_full_text,
    extract_metadata,
    extract_pages,
)
from src.ingestion.entities.extractor import extract_entities
from src.ingestion.relations.extractor import extract_deterministic_relations

logger = logging.getLogger(__name__)


class ManualImporter:
    """Import legal documents from local files into Neo4j."""

    def __init__(self) -> None:
        self.repo = GraphRepository()

    def ingest_file(
        self,
        file_path: str | Path,
        document_type: DocumentType,
        title: str,
        source_id: str = "MANUAL",
        source_name: str = "Manual Import",
        source_url: str | None = None,
        authority_level: int = 0,
        short_title: str | None = None,
        year: int | None = None,
        case_name: str | None = None,
        court: str | None = None,
        judgment_date: str | None = None,
        citation: str | None = None,
        bench: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Ingest a single file into the knowledge graph.

        Returns a summary dict with counts of created entities.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_type = detect_file_type(path)
        file_hash = compute_file_hash(path)

        # ── Extract text ─────────────────────────────────────────────
        if file_type == ".pdf":
            pages = extract_pages(path)
            raw_text = "\n\n".join(p["text"] for p in pages)
            pdf_meta = extract_metadata(path)
        elif file_type in (".txt", ".md"):
            raw_text = path.read_text(encoding="utf-8")
            pages = [{"page_number": 1, "text": raw_text}]
            pdf_meta = {}
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # ── Normalize ────────────────────────────────────────────────
        normalized_text = normalize_legal_text(raw_text)

        # ── Create document ID ───────────────────────────────────────
        document_id = LegalDocument.generate_id(source_id, title, source_url)

        source = SourceMetadata(
            source_id=source_id,
            name=source_name,
            url=source_url,
            authority_level=authority_level,
        )

        # ── Build document model ─────────────────────────────────────
        if document_type == DocumentType.JUDGMENT:
            doc = JudgmentDocument(
                document_id=document_id,
                document_type=document_type,
                title=title,
                short_title=short_title,
                year=year,
                source=source,
                retrieved_at=datetime.now(timezone.utc),
                status=DocumentStatus.ACTIVE,
                content_hash=file_hash,
                raw_text=raw_text,
                normalized_text=normalized_text,
                file_path=str(path),
                case_name=case_name,
                court=court,
                citation=citation,
                bench=bench or [],
            )
        else:
            doc = LegalDocument(
                document_id=document_id,
                document_type=document_type,
                title=title,
                short_title=short_title,
                year=year,
                source=source,
                retrieved_at=datetime.now(timezone.utc),
                status=DocumentStatus.ACTIVE,
                content_hash=file_hash,
                raw_text=raw_text,
                normalized_text=normalized_text,
                file_path=str(path),
            )

        # ── Chunk ────────────────────────────────────────────────────
        chunks = chunk_document(
            document_id=document_id,
            document_type=document_type,
            text=normalized_text,
            pages=pages if document_type == DocumentType.JUDGMENT else None,
            source_id=source_id,
            source_url=source_url,
            authority_level=authority_level,
        )

        # ── Extract entities and relations ───────────────────────────
        all_relations = []
        for chunk in chunks:
            rels = extract_deterministic_relations(chunk, document_type.value)
            all_relations.extend(rels)

        if dry_run:
            logger.info("[DRY RUN] Would ingest: %s → %d chunks, %d relations",
                       title, len(chunks), len(all_relations))
            return {
                "document_id": document_id,
                "title": title,
                "chunks": len(chunks),
                "relations": len(all_relations),
                "dry_run": True,
            }

        # ── Write to Neo4j ───────────────────────────────────────────
        # Source node
        self.repo.merge_source(source_id, source_name, source_url or "", authority_level)

        # Document node
        doc_props = {
            "title": doc.title,
            "short_title": doc.short_title or "",
            "document_type": doc.document_type.value,
            "year": doc.year or 0,
            "language": doc.language,
            "status": doc.status.value,
            "content_hash": file_hash,
            "retrieved_at": doc.retrieved_at.isoformat() if doc.retrieved_at else "",
            "file_path": str(path),
        }
        self.repo.merge_document(document_id, doc_props)

        # Type-specific node (Act, Case, etc.)
        if document_type in (DocumentType.ACT, DocumentType.CONSTITUTION):
            self.repo.merge_act(document_id, {
                "title": doc.title,
                "short_title": doc.short_title or doc.title,
                "year": doc.year or 0,
                "status": doc.status.value,
            })
        elif document_type == DocumentType.JUDGMENT and isinstance(doc, JudgmentDocument):
            self.repo.merge_case(document_id, {
                "case_name": doc.case_name or doc.title,
                "court": doc.court or "",
                "judgment_date": str(doc.judgment_date) if doc.judgment_date else "",
                "citation": doc.citation or "",
            })
            if doc.court:
                court_id = f"COURT_{doc.court.upper().replace(' ', '_')}"
                self.repo.merge_court(court_id, doc.court)
                self.repo.merge_relationship("Case", document_id, "DECIDED_BY", "Court", court_id)

        # Chunk nodes
        for chunk in chunks:
            chunk_props = {
                "text": chunk.text,
                "document_id": chunk.document_id,
                "page_start": chunk.page_start or 0,
                "page_end": chunk.page_end or 0,
                "paragraph_start": chunk.paragraph_start or 0,
                "paragraph_end": chunk.paragraph_end or 0,
                "section_number": chunk.section_number or "",
                "article_number": chunk.article_number or "",
                "source_id": chunk.source_id or "",
                "source_url": chunk.source_url or "",
                "authority_level": chunk.authority_level,
                "prev_chunk_id": chunk.prev_chunk_id or "",
                "next_chunk_id": chunk.next_chunk_id or "",
            }
            self.repo.merge_chunk(chunk.chunk_id, chunk_props)
            # Link chunk to document
            self.repo.merge_relationship("Document", document_id, "HAS_CHUNK", "Chunk", chunk.chunk_id)
            # Link chunk to source
            self.repo.merge_relationship("Chunk", chunk.chunk_id, "FROM_SOURCE", "Source", source_id)

        # Relations
        for rel in all_relations:
            # Only write accepted relations to the graph
            if rel.status.value in ("ACCEPTED", "PENDING"):
                try:
                    self.repo.merge_relationship(
                        rel.subject_label, rel.subject_id,
                        rel.relation_type,
                        rel.object_label, rel.object_id,
                        {
                            "evidence_chunk_id": rel.source_chunk_id,
                            "confidence": rel.confidence,
                            "method": rel.extraction_method.value,
                            "validated": rel.status == "ACCEPTED",
                        },
                    )
                except Exception as e:
                    # Don't fail the whole ingestion for a relation error
                    logger.warning("Failed to create relation %s: %s", rel.relation_id, e)

        logger.info(
            "Ingested: %s → %d chunks, %d relations",
            title, len(chunks), len(all_relations),
        )

        return {
            "document_id": document_id,
            "title": title,
            "chunks": len(chunks),
            "relations": len(all_relations),
            "file_hash": file_hash,
            "dry_run": False,
        }


def ingest_directory(
    directory: str | Path,
    document_type: DocumentType,
    source_id: str = "MANUAL",
    source_name: str = "Manual Import",
    authority_level: int = 0,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Batch-ingest all PDFs/text files in a directory.

    Never crashes the entire batch because one file fails (spec §98).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    results: list[dict[str, Any]] = []
    importer = ManualImporter()

    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() not in (".pdf", ".txt", ".md"):
            continue

        try:
            result = importer.ingest_file(
                file_path=file_path,
                document_type=document_type,
                title=file_path.stem.replace("_", " ").title(),
                source_id=source_id,
                source_name=source_name,
                authority_level=authority_level,
                dry_run=dry_run,
            )
            results.append(result)
        except Exception as e:
            logger.error("Failed to ingest %s: %s", file_path.name, e)
            results.append({
                "file": str(file_path),
                "error": str(e),
                "status": "failed",
            })

    logger.info(
        "Batch ingestion complete: %d/%d succeeded",
        sum(1 for r in results if "error" not in r),
        len(results),
    )
    return results
