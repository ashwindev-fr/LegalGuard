"""Text normalization for legal documents.

Cleans PDF artifacts while preserving the original legal wording.
NEVER rewrites, paraphrases, or "corrects" legal text.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces; strip leading/trailing."""
    text = re.sub(r"[ \t]+", " ", text)
    # Preserve paragraph breaks (double newline) but normalize single newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_hyphenation(text: str) -> str:
    """Rejoin words split by PDF line-break hyphenation.

    E.g. "judi-\\nciary" → "judiciary"
    Only applies when a lowercase letter precedes the hyphen and follows the
    newline — avoids mangling actual hyphens in legal text.
    """
    return re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)


def remove_repeated_headers_footers(text: str, threshold: int = 3) -> str:
    """Remove lines that repeat across many page boundaries.

    Heuristic: if a short line (< 80 chars) appears more than `threshold`
    times, it is likely a header/footer.
    """
    lines = text.split("\n")
    from collections import Counter

    line_counts = Counter(line.strip() for line in lines if 0 < len(line.strip()) < 80)
    repeated = {line for line, count in line_counts.items() if count >= threshold}

    if not repeated:
        return text

    cleaned = [line for line in lines if line.strip() not in repeated]
    return "\n".join(cleaned)


def remove_page_numbers(text: str) -> str:
    """Remove standalone page number lines (e.g. "42", "Page 3 of 10")."""
    return re.sub(r"^(?:Page\s+)?\d+(?:\s+of\s+\d+)?\s*$", "", text, flags=re.MULTILINE)


def normalize_unicode(text: str) -> str:
    """Normalize to NFC form and fix common encoding issues."""
    text = unicodedata.normalize("NFC", text)
    # Fix common ligature/encoding artifacts
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "–",
        "\u2014": "—",
        "\u2026": "...",
        "\xa0": " ",  # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_legal_text(raw_text: str) -> str:
    """Apply the full normalization pipeline.

    Returns normalized text. Does NOT modify legal wording, punctuation,
    or substance — only removes PDF artifacts.
    """
    text = normalize_unicode(raw_text)
    text = fix_hyphenation(text)
    text = remove_page_numbers(text)
    text = remove_repeated_headers_footers(text)
    text = normalize_whitespace(text)
    return text
