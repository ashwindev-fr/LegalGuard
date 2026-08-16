import pytest
from src.ingestion.chunking.legal_chunker import chunk_act, chunk_constitution
from src.ingestion.models import DocumentType

def test_chunk_constitution_by_articles():
    text = """
    Article 14. Equality before law.
    The State shall not deny to any person equality before the law.

    Article 21. Protection of life and personal liberty.
    No person shall be deprived of his life or personal liberty except according to procedure established by law.
    """
    chunks = chunk_constitution("CONST_001", text)
    assert len(chunks) >= 2
    assert any(c.article_number == "14" for c in chunks)
    assert any(c.article_number == "21" for c in chunks)

def test_chunk_act_by_sections():
    text = """
    Section 103. Punishment for murder.
    Whoever commits murder shall be punished with death or imprisonment for life.
    """
    chunks = chunk_act("BNS_2023", text)
    assert len(chunks) >= 1
    assert chunks[0].section_number == "103"
