import pytest
from pydantic import ValidationError

from src.schemas import CitationCheck, CitationVerification, CriticAssessment


def test_critic_assessment_accepts_all_fields():
    ca = CriticAssessment(confidence=0.8, hallucination_rate=0.1, notes="looks fine")
    assert ca.confidence == 0.8
    assert ca.hallucination_rate == 0.1
    assert ca.notes == "looks fine"


def test_critic_assessment_rejects_missing_field():
    with pytest.raises(ValidationError):
        CriticAssessment(confidence=0.8, hallucination_rate=0.1)


def test_citation_verification_aggregates_checks():
    cv = CitationVerification(
        checks=[
            CitationCheck(chunk_id=1, page="2", found_in_chunks=True, text_match=True),
            CitationCheck(chunk_id=2, page=None, found_in_chunks=False, text_match=False),
        ],
        verified_ratio=0.5,
    )
    assert len(cv.checks) == 2
    assert cv.verified_ratio == 0.5


def test_citation_check_accepts_a_page_range_string():
    check = CitationCheck(chunk_id=1, page="3-4", found_in_chunks=True, text_match=True)
    assert check.page == "3-4"
