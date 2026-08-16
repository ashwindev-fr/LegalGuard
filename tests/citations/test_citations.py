import pytest
from src.citations import CitationValidator
from src.ingestion.models import Claim, EvidenceItem, LegalAnswer

def test_citation_validator_rejects_invented_evidence_ids():
    validator = CitationValidator()

    evidence = EvidenceItem(
        evidence_id="E001",
        chunk_id="CHUNK_1",
        document_id="DOC_1",
        text="Sample legal text.",
    )

    answer = LegalAnswer(
        answer="Claim supported by E001 and fake E999.",
        claims=[
            Claim(text="Real claim", evidence_ids=["E001"]),
            Claim(text="Fake claim", evidence_ids=["E999"]),
        ],
        sources=[evidence],
    )

    validated = validator.validate_answer(answer)
    assert validated.claims[0].support_status == "SUPPORTED"
    assert validated.claims[1].support_status == "UNSUPPORTED"
    assert validated.claims[1].evidence_ids == []
    assert any("E999" in u for u in validated.uncertainties)
