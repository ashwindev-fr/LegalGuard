"""Benchmark dataset loader and structure (spec §46).

Loads test questions, gold answers, ground truth sections/cases/chunks,
and expected abstention flags.
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
class BenchmarkQuestion:
    """A benchmark question item for system evaluation."""
    id: str
    category: str
    question: str
    gold_answer: str
    gold_sections: list[str] = field(default_factory=list)
    gold_cases: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    should_abstain: bool = False


class EvaluationDataset:
    """Benchmark dataset manager."""

    def __init__(self, data_file: str | Path | None = None) -> None:
        settings = get_settings()
        self.data_file = Path(data_file or (settings.data_dir / "evaluation" / "questions.json"))
        self.questions: list[BenchmarkQuestion] = []
        self._load()

    def _load(self) -> None:
        if not self.data_file.exists():
            logger.info("Evaluation dataset file not found at %s. Initializing sample dataset.", self.data_file)
            self._create_sample_dataset()

        try:
            raw = json.loads(self.data_file.read_text(encoding="utf-8"))
            for item in raw:
                self.questions.append(BenchmarkQuestion(
                    id=item.get("id", ""),
                    category=item.get("category", "general"),
                    question=item.get("question", ""),
                    gold_answer=item.get("gold_answer", ""),
                    gold_sections=item.get("gold_sections", []),
                    gold_cases=item.get("gold_cases", []),
                    gold_chunk_ids=item.get("gold_chunk_ids", []),
                    should_abstain=item.get("should_abstain", False),
                ))
            logger.info("Loaded %d evaluation benchmark questions", len(self.questions))
        except Exception as e:
            logger.error("Failed to load evaluation dataset: %s", e)

    def _create_sample_dataset(self) -> None:
        """Create initial sample questions dataset file if missing."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        samples = [
            {
                "id": "Q001",
                "category": "constitutional",
                "question": "What protection does Article 21 of the Indian Constitution guarantee?",
                "gold_answer": "Article 21 guarantees protection of life and personal liberty, stating that no person shall be deprived of life or personal liberty except according to procedure established by law.",
                "gold_sections": ["CONSTITUTION_ART_21"],
                "should_abstain": False,
            },
            {
                "id": "Q002",
                "category": "criminal",
                "question": "What is the penalty for murder under Section 103 of Bharatiya Nyaya Sanhita, 2023?",
                "gold_answer": "Under Section 103 of BNS 2023, whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.",
                "gold_sections": ["BNS_2023_SEC_103"],
                "should_abstain": False,
            },
            {
                "id": "Q003",
                "category": "adversarial_no_answer",
                "question": "What did the Supreme Court hold in the imaginary case of Fake Corp v. Nonexistent State in 2099?",
                "gold_answer": "I could not verify this proposition from the retrieved authoritative sources, so I cannot provide a reliable legal conclusion.",
                "gold_sections": [],
                "should_abstain": True,
            },
        ]
        self.data_file.write_text(json.dumps(samples, indent=2), encoding="utf-8")
