from typing import List, Optional

from pydantic import BaseModel


class Citation(BaseModel):
    page: Optional[int]
    chunk_id: Optional[int]
    excerpt: Optional[str]


class SummaryResponse(BaseModel):
    summary: str
    citations: List[Citation]


class CriticAssessment(BaseModel):
    confidence: float
    hallucination_rate: float
    notes: str


class CitationCheck(BaseModel):
    chunk_id: Optional[int]
    page: Optional[int]
    found_in_chunks: bool
    text_match: bool


class CitationVerification(BaseModel):
    checks: List[CitationCheck]
    verified_ratio: float
