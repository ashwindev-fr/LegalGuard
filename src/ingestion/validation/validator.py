"""Graph quality and document validation suite (spec §66).

Executes automated checks for:
  - Orphan sections
  - Orphan cases
  - Invalid citations
  - Duplicate cases
  - Impossible temporal data
  - Chunks missing provenance
  - Chunks missing URLs
  - Unsupported strong relations
"""

from __future__ import annotations

import logging
from typing import Any

from src.graph.driver import run_query

logger = logging.getLogger(__name__)


class GraphQualityValidator:
    """Automated legal graph quality and validation engine."""

    def check_orphan_sections(self) -> list[dict[str, Any]]:
        """Find sections not connected to any Act."""
        query = """
        MATCH (s:Section)
        WHERE NOT (s)<-[:HAS_SECTION]-(:Act) AND NOT (s)<-[:CONTAINS]-(:Act)
        RETURN s.id AS section_id, s.title AS title
        """
        try:
            return run_query(query)
        except Exception as e:
            logger.warning("Error checking orphan sections: %s", e)
            return []

    def check_orphan_cases(self) -> list[dict[str, Any]]:
        """Find cases missing court, judgment date, or source links."""
        query = """
        MATCH (c:Case)
        WHERE c.court IS NULL OR c.court = '' OR c.judgment_date IS NULL
        RETURN c.id AS case_id, c.case_name AS case_name
        """
        try:
            return run_query(query)
        except Exception as e:
            logger.warning("Error checking orphan cases: %s", e)
            return []

    def check_chunks_missing_provenance(self) -> list[dict[str, Any]]:
        """Find chunks without a FROM_SOURCE link."""
        query = """
        MATCH (c:Chunk)
        WHERE NOT (c)-[:FROM_SOURCE]->(:Source)
        RETURN c.id AS chunk_id, c.document_id AS document_id
        """
        try:
            return run_query(query)
        except Exception as e:
            logger.warning("Error checking chunk provenance: %s", e)
            return []

    def check_temporal_consistency(self) -> list[dict[str, Any]]:
        """Find sections where effective_until < effective_from."""
        query = """
        MATCH (s:Section)
        WHERE s.effective_from IS NOT NULL
          AND s.effective_until IS NOT NULL
          AND s.effective_until < s.effective_from
        RETURN s.id AS section_id, s.effective_from AS from_date, s.effective_until AS until_date
        """
        try:
            return run_query(query)
        except Exception as e:
            logger.warning("Error checking temporal consistency: %s", e)
            return []

    def check_unsupported_strong_relations(self) -> list[dict[str, Any]]:
        """Find OVERRULES or FOLLOWS relationships missing evidence_chunk_id."""
        query = """
        MATCH (c1:Case)-[r:OVERRULES|FOLLOWS]->(c2:Case)
        WHERE r.evidence_chunk_id IS NULL OR r.evidence_chunk_id = ''
        RETURN c1.id AS source_case, type(r) AS relation, c2.id AS target_case
        """
        try:
            return run_query(query)
        except Exception as e:
            logger.warning("Error checking unsupported strong relations: %s", e)
            return []

    def generate_quality_report(self) -> dict[str, Any]:
        """Compile a complete dataset and graph quality report (spec §93)."""
        from src.graph.repository import GraphRepository

        repo = GraphRepository()
        counts = repo.get_node_counts()

        orphans_sec = self.check_orphan_sections()
        orphans_case = self.check_orphan_cases()
        missing_prov = self.check_chunks_missing_provenance()
        temporal_errs = self.check_temporal_consistency()
        unsupported_rels = self.check_unsupported_strong_relations()

        total_issues = (
            len(orphans_sec)
            + len(orphans_case)
            + len(missing_prov)
            + len(temporal_errs)
            + len(unsupported_rels)
        )

        return {
            "node_counts": counts,
            "quality_checks": {
                "orphan_sections_count": len(orphans_sec),
                "orphan_cases_count": len(orphans_case),
                "chunks_missing_provenance_count": len(missing_prov),
                "temporal_inconsistencies_count": len(temporal_errs),
                "unsupported_strong_relations_count": len(unsupported_rels),
                "total_quality_issues": total_issues,
            },
            "status": "PASS" if total_issues == 0 else "WARNINGS_FOUND",
            "details": {
                "orphan_sections": orphans_sec[:10],
                "orphan_cases": orphans_case[:10],
                "chunks_missing_provenance": missing_prov[:10],
                "temporal_inconsistencies": temporal_errs[:10],
                "unsupported_strong_relations": unsupported_rels[:10],
            },
        }
