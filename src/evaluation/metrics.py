"""Evaluation metrics engine (spec §48).

Calculates:
  - Retrieval: Recall@k, Precision@k, MRR, nDCG
  - Citation: Citation Precision, Citation Recall / Completeness
  - Hallucination: Unsupported claim rate
  - Abstention: Abstention precision, recall, false abstention rate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationMetrics:
    """Calculated metric scores for a single system or run."""
    system_name: str
    retrieval_recall_at_5: float = 0.0
    retrieval_recall_at_10: float = 0.0
    retrieval_precision_at_5: float = 0.0
    retrieval_mrr: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    unsupported_claim_rate: float = 0.0
    abstention_accuracy: float = 0.0
    total_evaluations: int = 0


def calculate_retrieval_recall(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    """Recall@k: proportion of gold relevant documents retrieved in top-k."""
    if not gold_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_found = sum(1 for g in gold_ids if g in top_k_retrieved)
    return relevant_found / len(gold_ids)


def calculate_retrieval_mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Mean Reciprocal Rank (MRR): 1 / rank of first relevant item."""
    if not gold_ids:
        return 1.0
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in gold_set:
            return 1.0 / rank
    return 0.0


def calculate_citation_precision(cited_ids: list[str], valid_ids: list[str]) -> float:
    """Citation Precision (spec §48): supported_citations / total_citations."""
    if not cited_ids:
        return 1.0
    valid_set = set(valid_ids)
    supported = sum(1 for cid in cited_ids if cid in valid_set)
    return supported / len(cited_ids)


def calculate_citation_recall(cited_ids: list[str], required_claim_ids: list[str]) -> float:
    """Citation Recall (spec §48): supported_required_claims / total_required_claims."""
    if not required_claim_ids:
        return 1.0
    cited_set = set(cited_ids)
    supported = sum(1 for req in required_claim_ids if req in cited_set)
    return supported / len(required_claim_ids)


def calculate_unsupported_claim_rate(claims: list[dict[str, Any]]) -> float:
    """Unsupported Claim Rate (spec §48): unsupported_claims / total_claims."""
    if not claims:
        return 0.0
    unsupported = sum(1 for c in claims if not c.get("evidence_ids") or c.get("support_status") == "UNSUPPORTED")
    return unsupported / len(claims)
