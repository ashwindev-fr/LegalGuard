import pytest
from src.ingestion.entities.extractor import (
    extract_articles,
    extract_citations,
    extract_courts,
    extract_sections,
)

def test_extract_sections():
    text = "Under Section 103 of BNS and Section 302 of IPC."
    sections = extract_sections(text)
    assert "103" in sections
    assert "302" in sections

def test_extract_articles():
    text = "Article 21 and Article 14 guarantee fundamental rights."
    articles = extract_articles(text)
    assert "21" in articles
    assert "14" in articles

def test_extract_courts():
    text = "The Supreme Court of India upheld the Delhi High Court decision."
    courts = extract_courts(text)
    assert "Supreme Court of India" in courts
    assert "Delhi High Court" in courts
