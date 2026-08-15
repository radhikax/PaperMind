from typing import List, Optional
from pydantic import BaseModel


class Citation(BaseModel):
    page: Optional[int]
    chunk_id: Optional[int]
    excerpt: Optional[str]


class SummaryResponse(BaseModel):
    summary: str
    citations: List[Citation]
