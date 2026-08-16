"""Neo4j graph schema — constraints, indexes, and vector index creation.

All operations are idempotent (IF NOT EXISTS).
"""

from __future__ import annotations

import logging

from src.config import get_settings
from src.graph.driver import get_session

logger = logging.getLogger(__name__)

# ── Uniqueness constraints ───────────────────────────────────────────────

CONSTRAINTS = [
    ("act_id", "Act", "id"),
    ("article_id", "Article", "id"),
    ("part_id", "Part", "id"),
    ("chapter_id", "Chapter", "id"),
    ("section_id", "Section", "id"),
    ("subsection_id", "SubSection", "id"),
    ("rule_id", "Rule", "id"),
    ("regulation_id", "Regulation", "id"),
    ("notification_id", "Notification", "id"),
    ("order_id", "Order", "id"),
    ("ordinance_id", "Ordinance", "id"),
    ("amendment_id", "Amendment", "id"),
    ("case_id", "Case", "id"),
    ("judgment_id", "Judgment", "id"),
    ("court_id", "Court", "id"),
    ("judge_id", "Judge", "id"),
    ("party_id", "Party", "id"),
    ("legal_concept_id", "LegalConcept", "id"),
    ("legal_principle_id", "LegalPrinciple", "id"),
    ("document_id", "Document", "id"),
    ("chunk_id", "Chunk", "id"),
    ("source_id", "Source", "id"),
    ("bill_id", "Bill", "id"),
    ("committee_report_id", "CommitteeReport", "id"),
    ("law_commission_report_id", "LawCommissionReport", "id"),
]


def create_constraints() -> None:
    """Create all uniqueness constraints (idempotent)."""
    with get_session() as session:
        for constraint_name, label, prop in CONSTRAINTS:
            query = (
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )
            session.run(query)
            logger.debug("Constraint ensured: %s", constraint_name)
    logger.info("All %d uniqueness constraints created/verified", len(CONSTRAINTS))


# ── Vector index ─────────────────────────────────────────────────────────

def create_vector_index(
    index_name: str = "chunk_embedding",
    label: str = "Chunk",
    property_name: str = "embedding",
    dimensions: int | None = None,
    similarity: str = "cosine",
) -> None:
    """Create a vector index on Chunk.embedding (idempotent)."""
    settings = get_settings()
    dim = dimensions or settings.embedding_dimension

    query = (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
        f"FOR (c:{label}) ON c.{property_name} "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {dim}, "
        f"`vector.similarity_function`: '{similarity}'"
        f"}}}}"
    )
    with get_session() as session:
        session.run(query)
    logger.info(
        "Vector index '%s' created/verified (dim=%d, similarity=%s)",
        index_name, dim, similarity,
    )


# ── Full-text index ─────────────────────────────────────────────────────

FULLTEXT_INDEXES = [
    {
        "name": "chunk_fulltext",
        "labels": ["Chunk"],
        "properties": ["text"],
    },
    {
        "name": "case_fulltext",
        "labels": ["Case"],
        "properties": ["case_name", "citation", "neutral_citation"],
    },
    {
        "name": "section_fulltext",
        "labels": ["Section"],
        "properties": ["title", "section_number", "text"],
    },
    {
        "name": "act_fulltext",
        "labels": ["Act"],
        "properties": ["title", "short_title"],
    },
]


def create_fulltext_indexes() -> None:
    """Create all full-text indexes (idempotent)."""
    with get_session() as session:
        for idx in FULLTEXT_INDEXES:
            labels = ", ".join(idx["labels"])
            props = ", ".join(f"n.{p}" for p in idx["properties"])
            query = (
                f"CREATE FULLTEXT INDEX {idx['name']} IF NOT EXISTS "
                f"FOR (n:{labels}) ON EACH [{props}]"
            )
            session.run(query)
            logger.debug("Full-text index ensured: %s", idx["name"])
    logger.info("All %d full-text indexes created/verified", len(FULLTEXT_INDEXES))


# ── Composite helper ────────────────────────────────────────────────────

def init_schema() -> None:
    """Create all constraints and indexes — safe to call multiple times."""
    logger.info("Initializing Neo4j schema...")
    create_constraints()
    create_vector_index()
    create_fulltext_indexes()
    logger.info("Neo4j schema initialization complete")
