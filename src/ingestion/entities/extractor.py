"""Deterministic legal entity extraction (Layer A — spec §23).

Uses regex and rules to extract:
  - Section numbers
  - Article numbers
  - Act names
  - Dates
  - Citations
  - Court names
  - Paragraph numbers
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ExtractedEntity:
    """An entity extracted from legal text."""
    entity_type: str          # e.g. "Section", "Article", "Act", "Court", "Date"
    value: str                # e.g. "103", "21", "Bharatiya Nyaya Sanhita"
    normalized_id: str | None = None  # e.g. "BNS_2023_SEC_103"
    start: int = 0
    end: int = 0
    context: str = ""         # surrounding text for verification


# ── Patterns ─────────────────────────────────────────────────────────────

SECTION_RE = re.compile(
    r"(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\(\d+\))?)",
    re.IGNORECASE,
)

ARTICLE_RE = re.compile(
    r"(?:Article|Art\.)\s*(\d+[A-Z]?(?:\(\d+\))?)",
    re.IGNORECASE,
)

ACT_RE = re.compile(
    r"(?:the\s+)?((?:[A-Z][a-z]+\s+)+(?:Act|Code|Bill|Sanhita|Suraksha|Adhiniyam))"
    r"(?:,?\s*(\d{4}))?",
)

COURT_NAMES = [
    "Supreme Court of India",
    "Supreme Court",
    "High Court",
    "Delhi High Court",
    "Bombay High Court",
    "Madras High Court",
    "Calcutta High Court",
    "Karnataka High Court",
    "Allahabad High Court",
    "Kerala High Court",
    "Gujarat High Court",
    "Punjab and Haryana High Court",
    "Andhra Pradesh High Court",
    "Telangana High Court",
    "Rajasthan High Court",
    "Patna High Court",
    "Gauhati High Court",
    "Orissa High Court",
    "Jharkhand High Court",
    "Chhattisgarh High Court",
    "Uttarakhand High Court",
    "Tripura High Court",
    "Meghalaya High Court",
    "Manipur High Court",
    "Sikkim High Court",
    "Himachal Pradesh High Court",
    "Jammu and Kashmir High Court",
    "District Court",
    "Sessions Court",
]

DATE_RE = re.compile(
    r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})\b"
    r"|\b(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})\b"
    r"|\b(\d{1,2})\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)

CITATION_RE = re.compile(
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+"         # (2024) 1 SCC 123
    r"|AIR\s+\d{4}\s+SC\s+\d+"             # AIR 2024 SC 123
    r"|\d{4}\s+SCC\s+OnLine\s+SC\s+\d+"    # 2024 SCC OnLine SC 123
    r"|\[\d{4}\]\s+\d+\s+SCR\s+\d+",       # [2024] 1 SCR 123
    re.IGNORECASE,
)


# ── Extractor ────────────────────────────────────────────────────────────


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract all deterministic legal entities from text."""
    entities: list[ExtractedEntity] = []

    # Sections
    for match in SECTION_RE.finditer(text):
        entities.append(ExtractedEntity(
            entity_type="Section",
            value=match.group(1),
            start=match.start(),
            end=match.end(),
            context=text[max(0, match.start() - 30):match.end() + 30],
        ))

    # Articles
    for match in ARTICLE_RE.finditer(text):
        entities.append(ExtractedEntity(
            entity_type="Article",
            value=match.group(1),
            start=match.start(),
            end=match.end(),
            context=text[max(0, match.start() - 30):match.end() + 30],
        ))

    # Acts
    for match in ACT_RE.finditer(text):
        act_name = match.group(1).strip()
        year = match.group(2)
        entities.append(ExtractedEntity(
            entity_type="Act",
            value=f"{act_name}, {year}" if year else act_name,
            start=match.start(),
            end=match.end(),
            context=text[max(0, match.start() - 20):match.end() + 20],
        ))

    # Courts
    for court in COURT_NAMES:
        idx = 0
        lower_text = text.lower()
        lower_court = court.lower()
        while True:
            pos = lower_text.find(lower_court, idx)
            if pos == -1:
                break
            entities.append(ExtractedEntity(
                entity_type="Court",
                value=court,
                start=pos,
                end=pos + len(court),
            ))
            idx = pos + len(court)

    # Citations
    for match in CITATION_RE.finditer(text):
        entities.append(ExtractedEntity(
            entity_type="Citation",
            value=match.group(0),
            start=match.start(),
            end=match.end(),
        ))

    # Dates
    for match in DATE_RE.finditer(text):
        entities.append(ExtractedEntity(
            entity_type="Date",
            value=match.group(0),
            start=match.start(),
            end=match.end(),
        ))

    return entities


def extract_sections(text: str) -> list[str]:
    """Convenience: extract only section numbers."""
    return [e.value for e in extract_entities(text) if e.entity_type == "Section"]


def extract_articles(text: str) -> list[str]:
    """Convenience: extract only article numbers."""
    return [e.value for e in extract_entities(text) if e.entity_type == "Article"]


def extract_courts(text: str) -> list[str]:
    """Convenience: extract only court names (deduplicated)."""
    return list(set(e.value for e in extract_entities(text) if e.entity_type == "Court"))


def extract_citations(text: str) -> list[str]:
    """Convenience: extract only legal citations."""
    return [e.value for e in extract_entities(text) if e.entity_type == "Citation"]
