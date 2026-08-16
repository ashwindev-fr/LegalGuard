"""Query analyzer — extracts structured legal intent from user questions (spec §26)."""

from __future__ import annotations

import logging

from src.ingestion.entities.extractor import (
    extract_articles,
    extract_citations,
    extract_courts,
    extract_sections,
)
from src.retrieval.models import QueryAnalysis

logger = logging.getLogger(__name__)

# ── Intent keywords ──────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "case_law_interpretation": [
        "which cases", "court held", "interpreted", "judgment",
        "decision", "ruling", "case law", "precedent",
    ],
    "statutory_lookup": [
        "what does section", "section says", "provision",
        "statutory", "according to", "as per",
    ],
    "constitutional": [
        "article", "constitutional", "fundamental right",
        "directive principle", "constitution",
    ],
    "relationship": [
        "follow", "overrule", "distinguish", "cite",
        "which judgments", "related cases",
    ],
    "comparison": [
        "differ", "compare", "contrast", "difference between",
    ],
    "temporal": [
        "when", "what year", "effective date", "applied in",
        "before amendment", "after amendment", "at the time",
    ],
    "citation": [
        "official source", "cite", "reference", "source for",
    ],
    "adversarial": [
        "make up", "assume", "ignore", "fabricate", "invent",
    ],
}


def analyze_query(query: str) -> QueryAnalysis:
    """Parse a user question into structured legal analysis."""
    query_lower = query.lower()

    # Detect intent
    intent = "general"
    best_score = 0
    for intent_name, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            intent = intent_name

    # Extract legal identifiers
    sections = extract_sections(query)
    articles = extract_articles(query)
    courts = extract_courts(query)
    citations = extract_citations(query)

    analysis = QueryAnalysis(
        raw_query=query,
        intent=intent,
        sections=sections,
        articles=articles,
        courts=courts,
        cases=citations,
        needs_citations=True,
        needs_explanation=True,
    )

    logger.info(
        "Query analysis: intent=%s, sections=%s, articles=%s, courts=%s",
        intent, sections, articles, courts,
    )
    return analysis
