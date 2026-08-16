import pytest
from src.ingestion.normalization.normalizer import (
    fix_hyphenation,
    normalize_legal_text,
    normalize_whitespace,
)

def test_normalize_whitespace():
    raw = "Section   103   \n\n  of BNS  "
    normalized = normalize_whitespace(raw)
    assert normalized == "Section 103 \n\n of BNS"

def test_fix_hyphenation():
    raw = "judi-\nciary"
    fixed = fix_hyphenation(raw)
    assert fixed == "judiciary"

def test_normalize_legal_text_preserves_wording():
    raw = "Article 21: Protection of life and personal liberty."
    result = normalize_legal_text(raw)
    assert "Article 21: Protection of life and personal liberty." in result
