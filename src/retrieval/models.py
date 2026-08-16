"""Retrieval models — RetrievalCandidate, QueryAnalysis (spec §26, §30)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryAnalysis:
    """Structured analysis of a user query (spec §26)."""
    raw_query: str
    intent: str = "general"  # case_law_interpretation, statutory_lookup, etc.
    sections: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    acts: list[str] = field(default_factory=list)
    courts: list[str] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)
    date_range: list[str] = field(default_factory=list)
    jurisdiction: str = "India"
    needs_citations: bool = True
    needs_explanation: bool = True
    legal_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "intent": self.intent,
            "sections": self.sections,
            "articles": self.articles,
            "acts": self.acts,
            "courts": self.courts,
            "cases": self.cases,
            "date_range": self.date_range,
            "jurisdiction": self.jurisdiction,
            "needs_citations": self.needs_citations,
            "needs_explanation": self.needs_explanation,
            "legal_concepts": self.legal_concepts,
        }


@dataclass
class RetrievalCandidate:
    """Standardized retrieval result (spec §30)."""
    evidence_id: str
    chunk_id: str
    document_id: str
    text: str
    score_vector: float | None = None
    score_lexical: float | None = None
    score_graph: float | None = None
    score_reranker: float | None = None
    score_final: float = 0.0
    authority_level: int = 0
    source_url: str = ""
    page_start: int | None = None
    page_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    section_number: str = ""
    article_number: str = ""
    case_name: str = ""
    court: str = ""
    citation: str = ""
    document_type: str = ""
    legal_entities: list[str] = field(default_factory=list)
