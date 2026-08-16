"""Neo4j graph repository — CRUD operations using MERGE for idempotency."""

from __future__ import annotations

import logging
from typing import Any

from src.graph.driver import get_session

logger = logging.getLogger(__name__)


class GraphRepository:
    """Provides idempotent MERGE-based write operations for the legal graph."""

    # ── Generic ──────────────────────────────────────────────────────

    @staticmethod
    def merge_node(label: str, node_id: str, properties: dict[str, Any]) -> None:
        """MERGE a node by id and SET additional properties."""
        safe_props = {k: v for k, v in properties.items() if k != "id"}
        set_clause = ", ".join(f"n.{k} = ${k}" for k in safe_props)
        query = f"MERGE (n:{label} {{id: $id}})"
        if set_clause:
            query += f" SET {set_clause}"
        params = {"id": node_id, **safe_props}
        with get_session() as session:
            session.run(query, params)

    @staticmethod
    def merge_relationship(
        src_label: str,
        src_id: str,
        rel_type: str,
        tgt_label: str,
        tgt_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """MERGE a relationship between two nodes identified by id."""
        props = properties or {}
        set_clause = ""
        if props:
            set_clause = " SET " + ", ".join(f"r.{k} = ${k}" for k in props)
        query = (
            f"MATCH (a:{src_label} {{id: $src_id}}) "
            f"MATCH (b:{tgt_label} {{id: $tgt_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b)"
            f"{set_clause}"
        )
        params = {"src_id": src_id, "tgt_id": tgt_id, **props}
        with get_session() as session:
            session.run(query, params)

    # ── Source ────────────────────────────────────────────────────────

    @staticmethod
    def merge_source(
        source_id: str,
        name: str,
        base_url: str,
        authority_level: int,
    ) -> None:
        """MERGE a Source node."""
        GraphRepository.merge_node("Source", source_id, {
            "name": name,
            "base_url": base_url,
            "authority_level": authority_level,
        })

    # ── Document ─────────────────────────────────────────────────────

    @staticmethod
    def merge_document(document_id: str, properties: dict[str, Any]) -> None:
        """MERGE a Document node."""
        GraphRepository.merge_node("Document", document_id, properties)

    # ── Act ───────────────────────────────────────────────────────────

    @staticmethod
    def merge_act(act_id: str, properties: dict[str, Any]) -> None:
        """MERGE an Act node."""
        GraphRepository.merge_node("Act", act_id, properties)

    # ── Section ──────────────────────────────────────────────────────

    @staticmethod
    def merge_section(section_id: str, properties: dict[str, Any]) -> None:
        """MERGE a Section node."""
        GraphRepository.merge_node("Section", section_id, properties)

    # ── Case ─────────────────────────────────────────────────────────

    @staticmethod
    def merge_case(case_id: str, properties: dict[str, Any]) -> None:
        """MERGE a Case node."""
        GraphRepository.merge_node("Case", case_id, properties)

    # ── Court ────────────────────────────────────────────────────────

    @staticmethod
    def merge_court(court_id: str, name: str) -> None:
        """MERGE a Court node."""
        GraphRepository.merge_node("Court", court_id, {"name": name})

    # ── Judge ────────────────────────────────────────────────────────

    @staticmethod
    def merge_judge(judge_id: str, name: str) -> None:
        """MERGE a Judge node."""
        GraphRepository.merge_node("Judge", judge_id, {"name": name})

    # ── Chunk ────────────────────────────────────────────────────────

    @staticmethod
    def merge_chunk(chunk_id: str, properties: dict[str, Any]) -> None:
        """MERGE a Chunk node (text, metadata, optionally embedding)."""
        GraphRepository.merge_node("Chunk", chunk_id, properties)

    @staticmethod
    def set_chunk_embedding(chunk_id: str, embedding: list[float]) -> None:
        """Set the embedding vector on an existing Chunk node."""
        query = "MATCH (c:Chunk {id: $id}) SET c.embedding = $embedding"
        with get_session() as session:
            session.run(query, {"id": chunk_id, "embedding": embedding})

    # ── Amendment ────────────────────────────────────────────────────

    @staticmethod
    def merge_amendment(amendment_id: str, properties: dict[str, Any]) -> None:
        """MERGE an Amendment node."""
        GraphRepository.merge_node("Amendment", amendment_id, properties)

    # ── LegalPrinciple ───────────────────────────────────────────────

    @staticmethod
    def merge_legal_principle(principle_id: str, properties: dict[str, Any]) -> None:
        """MERGE a LegalPrinciple node."""
        GraphRepository.merge_node("LegalPrinciple", principle_id, properties)

    # ── Article ──────────────────────────────────────────────────────

    @staticmethod
    def merge_article(article_id: str, properties: dict[str, Any]) -> None:
        """MERGE an Article node."""
        GraphRepository.merge_node("Article", article_id, properties)

    # ── Bulk helpers ─────────────────────────────────────────────────

    @staticmethod
    def get_all_chunk_ids() -> list[str]:
        """Return all Chunk IDs in the database."""
        query = "MATCH (c:Chunk) RETURN c.id AS id"
        with get_session() as session:
            result = session.run(query)
            return [record["id"] for record in result]

    @staticmethod
    def get_chunks_without_embeddings() -> list[dict[str, Any]]:
        """Return chunk IDs and text where embedding is null."""
        query = (
            "MATCH (c:Chunk) WHERE c.embedding IS NULL "
            "RETURN c.id AS id, c.text AS text"
        )
        with get_session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    @staticmethod
    def get_node_counts() -> dict[str, int]:
        """Return count of each node label in the database."""
        query = "CALL db.labels() YIELD label RETURN label"
        with get_session() as session:
            labels_result = session.run(query)
            labels = [record["label"] for record in labels_result]

        counts = {}
        for label in labels:
            count_query = f"MATCH (n:{label}) RETURN count(n) AS count"
            with get_session() as session:
                result = session.run(count_query)
                record = result.single()
                counts[label] = record["count"] if record else 0
        return counts
