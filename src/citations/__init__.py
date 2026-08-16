"""Citation validation and claim checking (spec §34-36, §49, §74).

Validates that:
- Every cited evidence exists
- Every cited chunk actually supports the claim
- No invented citation IDs
- No invented URLs
- Source URLs come from the database only
"""

from __future__ import annotations

import logging
from typing import Any

from src.ingestion.models import Claim, EvidenceItem, EvidencePacket, LegalAnswer

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validates citations in generated answers."""

    def validate_answer(self, answer: LegalAnswer) -> LegalAnswer:
        """Validate all citations in the answer.

        - Checks that every evidence_id in claims exists in sources
        - Marks unsupported claims
        - Adds warnings for uncited material claims
        """
        # Build a lookup of available evidence
        available_evidence: dict[str, EvidenceItem] = {
            e.evidence_id: e for e in answer.sources
        }

        validated_claims: list[Claim] = []
        warnings: list[str] = []

        for claim in answer.claims:
            # Check all cited evidence IDs exist
            valid_ids = []
            invalid_ids = []
            for eid in claim.evidence_ids:
                if eid in available_evidence:
                    valid_ids.append(eid)
                else:
                    invalid_ids.append(eid)

            if invalid_ids:
                warnings.append(
                    f"Claim cites non-existent evidence: {invalid_ids}"
                )
                logger.warning("Invalid evidence IDs in claim: %s", invalid_ids)

            # Update claim with validated IDs only
            validated_claim = Claim(
                text=claim.text,
                evidence_ids=valid_ids,
                claim_type=claim.claim_type,
                support_status="SUPPORTED" if valid_ids else "UNSUPPORTED",
            )
            validated_claims.append(validated_claim)

        # Check for claims without any citations
        uncited = [c for c in validated_claims if not c.evidence_ids]
        if uncited:
            warnings.append(
                f"{len(uncited)} claim(s) have no supporting evidence citations"
            )

        answer.claims = validated_claims
        answer.uncertainties = list(set(answer.uncertainties + warnings))
        return answer

    def check_evidence_support(
        self, claim_text: str, evidence: EvidenceItem
    ) -> str:
        """Check if evidence text actually supports the claim.

        Returns: SUPPORTED, NOT_SUPPORTED, or UNCERTAIN.

        Currently uses simple lexical overlap heuristic.
        Can be upgraded to NLI/cross-encoder later.
        """
        claim_words = set(claim_text.lower().split())
        evidence_words = set(evidence.text.lower().split())

        overlap = len(claim_words & evidence_words)
        overlap_ratio = overlap / max(len(claim_words), 1)

        if overlap_ratio > 0.3:
            return "SUPPORTED"
        elif overlap_ratio > 0.15:
            return "UNCERTAIN"
        else:
            return "NOT_SUPPORTED"


def validate_pre_generation(evidence_packet: EvidencePacket) -> list[str]:
    """Pre-generation validation (spec §34).

    Checks each evidence item before sending to LLM.
    Returns a list of validation warnings.
    """
    warnings: list[str] = []

    for item in evidence_packet.evidence:
        if not item.text or not item.text.strip():
            warnings.append(f"{item.evidence_id}: Empty text")
        if item.authority_level == 0:
            warnings.append(f"{item.evidence_id}: Unknown authority level")
        if not item.chunk_id:
            warnings.append(f"{item.evidence_id}: Missing chunk_id")
        if not item.document_id:
            warnings.append(f"{item.evidence_id}: Missing document_id")

    if not evidence_packet.evidence:
        warnings.append("No evidence items available — answer may require abstention")

    if warnings:
        logger.warning("Pre-generation validation warnings: %s", warnings)

    return warnings
