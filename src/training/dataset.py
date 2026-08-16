"""Fine-tuning dataset formatting and document-based splitting (spec §42-45).

Splits by document/case (NOT randomly by chunk) to prevent data leakage.
Encodes balanced dataset categories:
  20% legal QA
  15% citation grounding
  15% abstention
  10% temporal law
  10% conflicting authority
  10% source ranking
  10% explanation
  10% adversarial
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FineTuningExample:
    """Single fine-tuning dataset sample (spec §42)."""
    example_id: str
    category: str
    document_id: str
    question: str
    evidence: list[dict[str, Any]]
    target_answer: str
    claims: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "category": self.category,
            "document_id": self.document_id,
            "question": self.question,
            "evidence": self.evidence,
            "target_answer": self.target_answer,
            "claims": self.claims,
        }


def split_by_document(
    examples: list[FineTuningExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[FineTuningExample], list[FineTuningExample], list[FineTuningExample]]:
    """Group by document_id and split to avoid document leakage (spec §45)."""
    doc_to_examples: dict[str, list[FineTuningExample]] = {}
    for ex in examples:
        doc_to_examples.setdefault(ex.document_id, []).append(ex)

    doc_ids = sorted(list(doc_to_examples.keys()))
    total_docs = len(doc_ids)

    train_end = int(total_docs * train_ratio)
    val_end = train_end + int(total_docs * val_ratio)

    train_docs = set(doc_ids[:train_end])
    val_docs = set(doc_ids[train_end:val_end])

    train_examples = [ex for ex in examples if ex.document_id in train_docs]
    val_examples = [ex for ex in examples if ex.document_id in val_docs]
    test_examples = [ex for ex in examples if ex.document_id not in train_docs and ex.document_id not in val_docs]

    logger.info(
        "Split dataset: %d train (%d docs), %d val (%d docs), %d test (%d docs)",
        len(train_examples), len(train_docs),
        len(val_examples), len(val_docs),
        len(test_examples), total_docs - len(train_docs) - len(val_docs),
    )
    return train_examples, val_examples, test_examples
