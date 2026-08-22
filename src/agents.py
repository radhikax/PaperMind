from typing import Dict, List, Optional

from src.llm_client import get_openai_client
from src.retry import call_with_retries_validate
from src.schemas import CriticAssessment, SummaryResponse

_UNSET = object()


class RetrieverAgent:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embedder.embed_texts([query])
        results = self.store.search(q_emb, top_k=top_k)
        return results


def _format_passages(chunks: List[dict]) -> str:
    passages = []
    for i, c in enumerate(chunks):
        if isinstance(c, dict):
            text = c.get('text', '')
            src = c.get('source', {}) or {}
            page = src.get('page') or c.get('page')
            chunk_id = c.get('id') if c.get('id') is not None else i
            header = f"[chunk_id:{chunk_id} page:{page}]" if page is not None else f"[chunk_id:{chunk_id}]"
            passages.append(f"{header}\n{text}")
        else:
            passages.append(str(c))
    return "\n\n".join(passages)


class SummarizerAgent:
    def __init__(self, llm_model: str = "gpt-4o-mini", max_attempts: int = 3, client=_UNSET):
        self.model = llm_model
        self.max_attempts = max_attempts
        self.client = get_openai_client() if client is _UNSET else client

    @property
    def llm_available(self) -> bool:
        return self.client is not None

    def _heuristic_summary(self, joined: str) -> dict:
        text = joined[:1500].strip() + ("..." if len(joined) > 1500 else "")
        return {"summary": text, "citations": [], "valid": False}

    def summarize(
        self,
        chunks: List[dict],
        critique_feedback: Optional[str] = None,
        previous_summary: Optional[str] = None,
    ) -> dict:
        joined = _format_passages(chunks)
        if not self.llm_available:
            return self._heuristic_summary(joined)

        system = (
            "You are a helpful assistant that summarizes scientific paper snippets. "
            "Each passage is prefixed with metadata like [chunk_id:X page:Y]. "
            "Every citation you return must reference a chunk_id that actually appears "
            "in the passages, and its excerpt must be copied verbatim from that chunk's text."
        )
        user = f"Summarize the following passages, preserving factual statements:\n\n{joined}"
        if critique_feedback and previous_summary:
            user += (
                f"\n\nYour previous summary was:\n{previous_summary}\n\n"
                f"A reviewer found these problems: {critique_feedback}\n"
                "Revise the summary to fix these specific issues while staying grounded "
                "in the passages above."
            )

        def call():
            resp = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=SummaryResponse,
                temperature=0.0,
            )
            return resp.choices[0].message

        def is_valid(message) -> bool:
            return message is not None and getattr(message, "parsed", None) is not None

        try:
            message = call_with_retries_validate(
                call, validator=is_valid, max_attempts=self.max_attempts, initial_delay=0.5,
            )
        except Exception:
            message = None

        if message is None or message.parsed is None:
            return self._heuristic_summary(joined)

        parsed: SummaryResponse = message.parsed
        citation_strs = [
            f"[p.{c.page}#id:{c.chunk_id}]" for c in parsed.citations if c.page and c.chunk_id is not None
        ]
        summary_text = parsed.summary
        if citation_strs:
            summary_text = summary_text + "\n\nCitations: " + ", ".join(citation_strs)

        return {
            "summary": summary_text,
            "citations": [c.model_dump() for c in parsed.citations],
            "valid": True,
        }


class CriticAgent:
    def __init__(self, llm_model: str = "gpt-4o-mini", max_attempts: int = 2, client=_UNSET):
        self.model = llm_model
        self.max_attempts = max_attempts
        self.client = get_openai_client() if client is _UNSET else client

    @property
    def llm_available(self) -> bool:
        return self.client is not None

    def _heuristic_assessment(self, summary: str) -> dict:
        length = len(summary)
        confidence = min(0.95, max(0.1, length / 2000.0))
        return {
            "confidence": round(confidence, 2),
            "hallucination_rate": round(1 - confidence, 2),
            "notes": "Heuristic fallback: confidence based on summary length.",
        }

    def assess(self, summary: str) -> Dict:
        if not self.llm_available:
            return self._heuristic_assessment(summary)

        system = (
            "You are an expert reviewer who detects hallucinations in summaries of "
            "scientific text. Assign a confidence (0-1) that the summary is fully "
            "supported by its source, a hallucination_rate (0-1), and short notes."
        )
        user = f"Assess the following summary for hallucinations:\n\n{summary}"

        def call():
            resp = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=CriticAssessment,
                temperature=0.0,
            )
            return resp.choices[0].message

        def is_valid(message) -> bool:
            return message is not None and getattr(message, "parsed", None) is not None

        try:
            message = call_with_retries_validate(
                call, validator=is_valid, max_attempts=self.max_attempts, initial_delay=0.5,
            )
        except Exception:
            message = None

        if message is None or message.parsed is None:
            return self._heuristic_assessment(summary)

        return message.parsed.model_dump()


class CitationVerifierAgent:
    """Checks summarizer citations against the retrieved chunks; no LLM call required."""

    def verify(self, citations: List[dict], retrieved_chunks: List[dict]) -> dict:
        chunk_by_id = {c.get('id'): c for c in retrieved_chunks if c.get('id') is not None}
        checks = []
        for cit in citations:
            chunk_id = cit.get('chunk_id')
            page = cit.get('page')
            excerpt = (cit.get('excerpt') or '').strip().lower()
            chunk = chunk_by_id.get(chunk_id)
            found = chunk is not None
            if found and excerpt:
                text_match = excerpt in (chunk.get('text', '') or '').lower()
            else:
                text_match = found
            checks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page,
                    "found_in_chunks": found,
                    "text_match": text_match,
                }
            )

        if not checks:
            return {"checks": [], "verified_ratio": 0.0}

        verified = sum(1 for c in checks if c["found_in_chunks"] and c["text_match"])
        return {"checks": checks, "verified_ratio": verified / len(checks)}
