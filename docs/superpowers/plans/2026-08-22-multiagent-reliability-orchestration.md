# Multi-Agent Reliability Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the non-functional LangGraph facade in `src/langgraph_agents.py` with a real LangGraph `StateGraph` that runs a reliability-gated revision loop — summaries are scored by a dedicated `ReliabilityEvaluator` agent and sent back for a targeted rewrite when untrustworthy, instead of being shown as-is.

**Architecture:** Five single-purpose agent nodes (Retriever, Summarizer, Critic, CitationVerifier, ReliabilityEvaluator) wired into a `StateGraph`. Critic and CitationVerifier run as parallel branches after each Summarizer attempt and join at ReliabilityEvaluator, which either accepts, loops back to Summarizer with concrete critique text, or (after `max_attempts`) returns the best attempt flagged low-confidence. `src/ingest.py`, `src/embeddings.py`, `src/vectorstore.py`, and the Streamlit UI shell/feedback loop are unchanged.

**Tech Stack:** Python 3.10, LangGraph `StateGraph` (real package, `pip show` confirms `langgraph==1.2.11` already installed in `.venv`), OpenAI Python SDK v1+ client with structured outputs (`openai==3.1.0` installed), Pydantic v2, pytest, Streamlit.

**Spec:** `docs/superpowers/specs/2026-08-22-multiagent-reliability-design.md`

## Global Constraints

- Python 3.10 (matches CI's `actions/setup-python` config in `.github/workflows/ci.yml`).
- `requirements.txt` pins: `langgraph>=1.2.0,<2.0.0`, `openai>=3.0.0,<4.0.0` (both already installed in `.venv` at 1.2.11 / 3.1.0 — pin the range, don't add a new package manager).
- flake8 max line length 120 (`flake8 --max-line-length=120 src tests app.py`, as run in CI).
- isort default profile, no config file exists — match existing import grouping: stdlib, blank line, third-party, blank line, `src.*` local imports (see `app.py` for the existing pattern).
- No comments explaining *what* code does; only *why*, and only when non-obvious.
- Structured LLM outputs require a model that supports OpenAI structured outputs (e.g. `gpt-4o-mini`) — `gpt-3.5-turbo` (the old default) does not support `response_format=<pydantic model>`. This plan changes every agent's default `llm_model` to `"gpt-4o-mini"` as a direct consequence of the already-approved "modernize OpenAI SDK usage" decision, not a new scope item.
- Every task must pass `PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q` and `isort --check-only src tests app.py` + `flake8 --max-line-length=120 src tests app.py` before committing (mirrors CI).
- Run all commands in this plan through `.venv/Scripts/python.exe` / `.venv/Scripts/pip.exe` (or an activated `.venv`) — the system `python` on this machine is a different, unrelated 3.14 install with none of this project's dependencies.

---

### Task 1: Retry-with-backoff helper

**Files:**
- Create: `src/retry.py`
- Test: `tests/test_retry.py`

**Interfaces:**
- Produces: `call_with_retries_validate(fn, *args, validator=None, max_attempts=3, initial_delay=1.0, multiplier=2.0, on_attempt=None, **kwargs) -> Any` — calls `fn(*args, **kwargs)` until `validator(result)` is `True` or attempts are exhausted; re-raises the last exception if every attempt raised.

This is the retry logic currently inlined in `app.py:17-62`. Pulling it into its own module makes it reusable by the agent classes (Task 4) and testable without Streamlit.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retry.py
from src.retry import call_with_retries_validate


def test_returns_immediately_when_no_validator():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = call_with_retries_validate(fn, max_attempts=3, initial_delay=0)
    assert result == "ok"
    assert len(calls) == 1


def test_retries_until_validator_passes():
    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    result = call_with_retries_validate(fn, validator=lambda r: r >= 3, max_attempts=5, initial_delay=0)
    assert result == 3
    assert len(calls) == 3


def test_gives_up_after_max_attempts_and_returns_last_result():
    calls = []

    def fn():
        calls.append(1)
        return "bad"

    result = call_with_retries_validate(fn, validator=lambda r: r == "good", max_attempts=2, initial_delay=0)
    assert result == "bad"
    assert len(calls) == 2


def test_on_attempt_callback_invoked_per_try():
    seen = []

    def fn():
        return "bad"

    call_with_retries_validate(
        fn,
        validator=lambda r: r == "good",
        max_attempts=3,
        initial_delay=0,
        on_attempt=lambda a: seen.append(a),
    )
    assert seen == [1, 2, 3]


def test_reraises_after_exhausting_attempts_on_exception():
    calls = []

    def fn():
        calls.append(1)
        raise RuntimeError("boom")

    try:
        call_with_retries_validate(fn, max_attempts=2, initial_delay=0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert len(calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_retry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retry'`

- [ ] **Step 3: Implement `src/retry.py`**

```python
import time
from typing import Any, Callable, Optional


def call_with_retries_validate(
    fn: Callable[..., Any],
    *args: Any,
    validator: Optional[Callable[[Any], bool]] = None,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    on_attempt: Optional[Callable[[int], None]] = None,
    **kwargs: Any,
) -> Any:
    """Call fn(*args, **kwargs) repeatedly until validator(result) is True or attempts exhausted."""
    delay = initial_delay
    result = None
    for attempt in range(1, max_attempts + 1):
        if on_attempt:
            try:
                on_attempt(attempt)
            except Exception:
                pass
        try:
            result = fn(*args, **kwargs)
        except Exception:
            if attempt < max_attempts:
                time.sleep(delay)
                delay *= multiplier
                continue
            raise

        if validator is None:
            return result

        try:
            ok = validator(result)
        except Exception:
            ok = False

        if ok:
            return result
        if attempt < max_attempts:
            time.sleep(delay)
            delay *= multiplier
            continue
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_retry.py -v`
Expected: 5 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m isort src/retry.py tests/test_retry.py
.venv/Scripts/python.exe -m flake8 --max-line-length=120 src/retry.py tests/test_retry.py
git add src/retry.py tests/test_retry.py
git commit -m "Add reusable retry-with-backoff helper"
```

---

### Task 2: Reliability-related schemas

**Files:**
- Modify: `src/schemas.py`
- Test: `tests/test_schemas.py` (new)

**Interfaces:**
- Consumes: nothing new (extends the existing `Citation`/`SummaryResponse` models already in `src/schemas.py`).
- Produces: `CriticAssessment(confidence: float, hallucination_rate: float, notes: str)`, `CitationCheck(chunk_id: Optional[int], page: Optional[int], found_in_chunks: bool, text_match: bool)`, `CitationVerification(checks: List[CitationCheck], verified_ratio: float)` — used by Task 4 (agents) and Task 6 (graph nodes).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schemas.py
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
            CitationCheck(chunk_id=1, page=2, found_in_chunks=True, text_match=True),
            CitationCheck(chunk_id=2, page=None, found_in_chunks=False, text_match=False),
        ],
        verified_ratio=0.5,
    )
    assert len(cv.checks) == 2
    assert cv.verified_ratio == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_schemas.py -v`
Expected: FAIL with `ImportError: cannot import name 'CriticAssessment' from 'src.schemas'`

- [ ] **Step 3: Extend `src/schemas.py`**

Add to the end of the existing file (keep `Citation` and `SummaryResponse` as-is):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_schemas.py -v`
Expected: 3 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m isort src/schemas.py tests/test_schemas.py
.venv/Scripts/python.exe -m flake8 --max-line-length=120 src/schemas.py tests/test_schemas.py
git add src/schemas.py tests/test_schemas.py
git commit -m "Add CriticAssessment and citation-verification schemas"
```

---

### Task 3: OpenAI client factory

**Files:**
- Create: `src/llm_client.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Produces: `get_openai_client() -> Optional[OpenAI]` — returns a configured `openai.OpenAI` client if the SDK is importable and `OPENAI_API_KEY` is set, else `None`. Loads `.env` via `python-dotenv` (moved here from `src/agents.py`, single place for this concern now).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_client.py
import src.llm_client as llm_client


def test_get_openai_client_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_client.get_openai_client() is None


def test_get_openai_client_returns_none_when_sdk_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm_client, "OpenAI", None)
    assert llm_client.get_openai_client() is None


def test_get_openai_client_returns_client_when_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = llm_client.get_openai_client()
    assert client is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm_client'`

- [ ] **Step 3: Implement `src/llm_client.py`**

```python
import os
from typing import Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_openai_client() -> Optional["OpenAI"]:
    """Return a configured OpenAI client, or None if the SDK or API key is unavailable."""
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_llm_client.py -v`
Expected: 3 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m isort src/llm_client.py tests/test_llm_client.py
.venv/Scripts/python.exe -m flake8 --max-line-length=120 src/llm_client.py tests/test_llm_client.py
git add src/llm_client.py tests/test_llm_client.py
git commit -m "Add OpenAI client factory used by agent classes"
```

---

### Task 4: Modernize agents — Summarizer, Critic, new CitationVerifier

**Files:**
- Modify: `src/agents.py` (full rewrite of `SummarizerAgent`/`CriticAgent`, removal of `VerifierAgent`, addition of `CitationVerifierAgent`; `RetrieverAgent` unchanged)
- Modify: `requirements.txt` (pin `openai>=3.0.0,<4.0.0`)
- Modify: `tests/test_agents.py` (full rewrite)

`VerifierAgent` (currently `src/agents.py:28-40`) is removed: its overlap-ratio logic is a duplicate of `verifier_overlap_ratio` in `src/evaluator.py`, which `ReliabilityEvaluator` (Task 5) calls directly. Its only caller, `app.py`, is rewritten in Task 7 to read the reliability score from the graph instead of calling `VerifierAgent` separately — after that, `VerifierAgent` has no callers, so it's deleted rather than left as dead code.

**Interfaces:**
- Consumes: `call_with_retries_validate` from `src/retry.py` (Task 1); `get_openai_client` from `src/llm_client.py` (Task 3); `SummaryResponse`, `CriticAssessment` from `src/schemas.py` (Task 2).
- Produces:
  - `RetrieverAgent(store, embedder).retrieve(query: str, top_k: int = 5) -> List[Tuple[float, dict]]` (unchanged signature).
  - `SummarizerAgent(llm_model="gpt-4o-mini", max_attempts=3, client=<unset>).summarize(chunks: List[dict], critique_feedback: Optional[str] = None, previous_summary: Optional[str] = None) -> dict` returning `{"summary": str, "citations": List[dict], "valid": bool}`. `.llm_available: bool` property.
  - `CriticAgent(llm_model="gpt-4o-mini", max_attempts=2, client=<unset>).assess(summary: str) -> dict` returning `{"confidence": float, "hallucination_rate": float, "notes": str}`. `.llm_available: bool` property.
  - `CitationVerifierAgent().verify(citations: List[dict], retrieved_chunks: List[dict]) -> dict` returning `{"checks": List[dict], "verified_ratio": float}`. Pure Python, no LLM call.
  - Passing `client=None` explicitly to `SummarizerAgent`/`CriticAgent` forces the no-LLM heuristic fallback (used by tests); omitting `client` auto-detects via `get_openai_client()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agents.py
from src.agents import CitationVerifierAgent, CriticAgent, SummarizerAgent
from src.schemas import Citation, CriticAssessment, SummaryResponse


class FakeMessage:
    def __init__(self, parsed=None, content="", refusal=""):
        self.parsed = parsed
        self.content = content
        self.refusal = refusal


class FakeResponse:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class FakeCompletions:
    def __init__(self, message):
        self._message = message

    def parse(self, **kwargs):
        return FakeResponse(self._message)


class FakeClient:
    def __init__(self, message):
        self.chat = type("Chat", (), {"completions": FakeCompletions(message)})()


class RaisingClient:
    class _RaisingCompletions:
        def parse(self, **kwargs):
            raise RuntimeError("network error")

    def __init__(self):
        self.chat = type("Chat", (), {"completions": self._RaisingCompletions()})()


def test_summarizer_without_client_falls_back_to_heuristic():
    agent = SummarizerAgent(client=None)
    chunks = [{"text": "This is a test passage.", "source": {"page": 3}, "id": 7}]
    res = agent.summarize(chunks)
    assert res["valid"] is False
    assert "[chunk_id:7" in res["summary"]


def test_summarizer_uses_structured_output_when_client_available():
    parsed = SummaryResponse(
        summary="Short summary.",
        citations=[Citation(page=3, chunk_id=7, excerpt="test passage")],
    )
    agent = SummarizerAgent(client=FakeClient(FakeMessage(parsed=parsed)))
    chunks = [{"text": "This is a test passage.", "source": {"page": 3}, "id": 7}]
    res = agent.summarize(chunks)
    assert res["valid"] is True
    assert "Short summary." in res["summary"]
    assert res["citations"][0]["chunk_id"] == 7


def test_summarizer_includes_critique_feedback_in_revision_prompt():
    captured = {}

    class CapturingCompletions:
        def parse(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return FakeResponse(FakeMessage(parsed=SummaryResponse(summary="Revised.", citations=[])))

    client = type("C", (), {"chat": type("Chat", (), {"completions": CapturingCompletions()})()})()
    agent = SummarizerAgent(client=client)
    chunks = [{"text": "Passage.", "source": {"page": 1}, "id": 1}]
    agent.summarize(chunks, critique_feedback="citation missing", previous_summary="Old summary")
    user_msg = captured["messages"][1]["content"]
    assert "citation missing" in user_msg
    assert "Old summary" in user_msg


def test_summarizer_falls_back_when_client_raises():
    agent = SummarizerAgent(client=RaisingClient(), max_attempts=1)
    chunks = [{"text": "Passage text.", "source": {"page": 1}, "id": 1}]
    res = agent.summarize(chunks)
    assert res["valid"] is False


def test_summarizer_max_attempts_attr():
    a = SummarizerAgent(max_attempts=5, client=None)
    assert a.max_attempts == 5


def test_critic_without_client_falls_back_to_heuristic():
    agent = CriticAgent(client=None)
    result = agent.assess("short")
    assert 0.0 <= result["confidence"] <= 1.0
    assert "hallucination_rate" in result


def test_critic_uses_structured_output_when_client_available():
    parsed = CriticAssessment(confidence=0.9, hallucination_rate=0.1, notes="solid")
    agent = CriticAgent(client=FakeClient(FakeMessage(parsed=parsed)))
    result = agent.assess("A well supported summary.")
    assert result == {"confidence": 0.9, "hallucination_rate": 0.1, "notes": "solid"}


def test_critic_falls_back_when_client_raises():
    agent = CriticAgent(client=RaisingClient(), max_attempts=1)
    result = agent.assess("some summary text")
    assert "confidence" in result


def test_citation_verifier_flags_missing_chunk():
    result = CitationVerifierAgent().verify(
        [{"chunk_id": 99, "page": 1, "excerpt": "nope"}],
        [{"id": 1, "text": "Some real text.", "source": {"page": 1}}],
    )
    assert result["verified_ratio"] == 0.0
    assert result["checks"][0]["found_in_chunks"] is False


def test_citation_verifier_matches_excerpt_in_source_text():
    result = CitationVerifierAgent().verify(
        [{"chunk_id": 1, "page": 1, "excerpt": "real text"}],
        [{"id": 1, "text": "Some real text right here.", "source": {"page": 1}}],
    )
    assert result["verified_ratio"] == 1.0
    assert result["checks"][0]["text_match"] is True


def test_citation_verifier_no_citations_returns_zero_ratio():
    result = CitationVerifierAgent().verify([], [{"id": 1, "text": "x", "source": {}}])
    assert result == {"checks": [], "verified_ratio": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_agents.py -v`
Expected: FAIL — `CitationVerifierAgent` doesn't exist yet, `client=` kwarg not accepted yet.

- [ ] **Step 3: Rewrite `src/agents.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_agents.py -v`
Expected: 11 passed

- [ ] **Step 5: Pin openai in requirements.txt**

Edit `requirements.txt`, change the `openai` line to:

```
openai>=3.0.0,<4.0.0
```

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m isort src/agents.py tests/test_agents.py
.venv/Scripts/python.exe -m flake8 --max-line-length=120 src/agents.py tests/test_agents.py
PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q
git add src/agents.py tests/test_agents.py requirements.txt
git commit -m "Modernize Summarizer/Critic agents, add CitationVerifierAgent"
```

---

### Task 5: Reliability evaluator

**Files:**
- Modify: `src/evaluator.py` (remove `citation_present`, remove `compute_numeric_confidence`, add `ReliabilityEvaluator`; keep `semantic_similarity_score` and `verifier_overlap_ratio` unchanged)
- Modify: `tests/test_evaluator.py` (full rewrite)

`citation_present`'s regex heuristic is fully superseded by `CitationVerifierAgent` (Task 4). `compute_numeric_confidence`'s only caller was `app.py`, which is rewritten in Task 7 to read `reliability_score` from the graph state instead — so both are deleted rather than kept as dead code.

**Interfaces:**
- Consumes: nothing new (uses `store.search`/`embedder.embed_texts` duck-typed interfaces already used elsewhere in the repo).
- Produces:
  - `semantic_similarity_score(summary, store, embedder, top_k=10) -> float` (unchanged).
  - `verifier_overlap_ratio(summary, store, embedder, original_ids, top_k=10) -> float` (unchanged).
  - `ReliabilityEvaluator(accept_threshold=70).evaluate(summary_text, retrieved_chunks, store, embedder, critic_assessment, citation_verified_ratio, attempt, max_attempts) -> dict` returning `{"score": int, "decision": "accept" | "revise" | "exhausted", "critique_feedback": Optional[str]}`. Used by Task 6's `reliability_evaluator_node`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluator.py
from src.evaluator import ReliabilityEvaluator, semantic_similarity_score, verifier_overlap_ratio


class DummyStore:
    def __init__(self, score=0.95, ids=(1,)):
        self.score = score
        self.ids = ids

    def search(self, q_emb, top_k=10):
        return [(self.score, {"id": i, "source": {"page": 1}}) for i in self.ids]


class EmptyStore:
    def search(self, q_emb, top_k=10):
        return []


class DummyEmbedder:
    def embed_texts(self, texts, batch_size=64, normalize=True):
        import numpy as np

        return np.ones((len(texts), 8), dtype="float32")


def test_semantic_similarity_score_uses_max_result_score():
    assert semantic_similarity_score("a summary", DummyStore(score=0.8), DummyEmbedder()) == 0.8


def test_semantic_similarity_score_empty_results_is_zero():
    assert semantic_similarity_score("x", EmptyStore(), DummyEmbedder()) == 0.0


def test_verifier_overlap_ratio_full_overlap():
    assert verifier_overlap_ratio("a summary", DummyStore(ids=(1, 2)), DummyEmbedder(), original_ids=[1, 2]) == 1.0


def test_verifier_overlap_ratio_no_original_ids_is_zero():
    assert verifier_overlap_ratio("x", DummyStore(ids=(1,)), DummyEmbedder(), original_ids=[]) == 0.0


def test_reliability_evaluator_accepts_high_score():
    evaluator = ReliabilityEvaluator(accept_threshold=70)
    result = evaluator.evaluate(
        "a summary",
        retrieved_chunks=[{"id": 1}],
        store=DummyStore(score=0.95, ids=(1,)),
        embedder=DummyEmbedder(),
        critic_assessment={"confidence": 0.9, "hallucination_rate": 0.05},
        citation_verified_ratio=1.0,
        attempt=1,
        max_attempts=3,
    )
    assert result["decision"] == "accept"
    assert result["score"] >= 70
    assert result["critique_feedback"] is None


def test_reliability_evaluator_revises_low_score_with_retries_left():
    evaluator = ReliabilityEvaluator(accept_threshold=70)
    result = evaluator.evaluate(
        "a summary",
        retrieved_chunks=[{"id": 99}],
        store=DummyStore(score=0.0, ids=(1,)),
        embedder=DummyEmbedder(),
        critic_assessment={"confidence": 0.1, "hallucination_rate": 0.8},
        citation_verified_ratio=0.0,
        attempt=1,
        max_attempts=3,
    )
    assert result["decision"] == "revise"
    assert result["critique_feedback"] is not None
    assert "hallucination_rate" in result["critique_feedback"]


def test_reliability_evaluator_exhausted_after_max_attempts():
    evaluator = ReliabilityEvaluator(accept_threshold=70)
    result = evaluator.evaluate(
        "a summary",
        retrieved_chunks=[{"id": 99}],
        store=DummyStore(score=0.0, ids=(1,)),
        embedder=DummyEmbedder(),
        critic_assessment={"confidence": 0.1, "hallucination_rate": 0.8},
        citation_verified_ratio=0.0,
        attempt=3,
        max_attempts=3,
    )
    assert result["decision"] == "exhausted"
    assert result["critique_feedback"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_evaluator.py -v`
Expected: FAIL — `ReliabilityEvaluator` doesn't exist yet.

- [ ] **Step 3: Rewrite `src/evaluator.py`**

```python
from typing import List, Optional


def semantic_similarity_score(summary: str, store, embedder, top_k: int = 10) -> float:
    """Compute semantic similarity score between summary and retrieved chunks.

    Returns a float in [0,1] (higher is more similar).
    """
    q_emb = embedder.embed_texts([summary])
    results = store.search(q_emb, top_k=top_k)
    if not results:
        return 0.0
    scores = [s for s, _ in results]
    norm_scores = [max(0.0, min(1.0, float(sc))) for sc in scores]
    return float(max(norm_scores))


def verifier_overlap_ratio(summary: str, store, embedder, original_ids: List[int], top_k: int = 10) -> float:
    """Re-query the vector DB with the summary and compute overlap ratio with original ids.

    Returns ratio in [0,1].
    """
    q_emb = embedder.embed_texts([summary])
    results = store.search(q_emb, top_k=top_k)
    retrieved_ids = [m.get('id') for _, m in results if m.get('id') is not None]
    if not original_ids:
        return 0.0
    common = set(original_ids) & set(retrieved_ids)
    return len(common) / len(set(original_ids))


class ReliabilityEvaluator:
    """Aggregates multiple signals into a 0-100 reliability score and a routing decision."""

    SEMANTIC_WEIGHT = 0.40
    VERIFIER_WEIGHT = 0.25
    CRITIC_WEIGHT = 0.20
    CITATION_WEIGHT = 0.15

    def __init__(self, accept_threshold: int = 70):
        self.accept_threshold = accept_threshold

    def evaluate(
        self,
        summary_text: str,
        retrieved_chunks: List[dict],
        store,
        embedder,
        critic_assessment: Optional[dict],
        citation_verified_ratio: float,
        attempt: int,
        max_attempts: int,
    ) -> dict:
        original_ids = [c.get('id') for c in retrieved_chunks if c.get('id') is not None]
        semantic = semantic_similarity_score(summary_text, store, embedder, top_k=10)
        verifier = verifier_overlap_ratio(summary_text, store, embedder, original_ids, top_k=10)

        critic_conf = 0.5
        if critic_assessment and isinstance(critic_assessment.get('confidence'), (int, float)):
            critic_conf = float(critic_assessment['confidence'])

        base = (
            self.SEMANTIC_WEIGHT * semantic
            + self.VERIFIER_WEIGHT * verifier
            + self.CRITIC_WEIGHT * critic_conf
            + self.CITATION_WEIGHT * citation_verified_ratio
        )
        score = int(round(max(0.0, min(1.0, base)) * 100))

        if score >= self.accept_threshold:
            return {"score": score, "decision": "accept", "critique_feedback": None}

        if attempt >= max_attempts:
            return {"score": score, "decision": "exhausted", "critique_feedback": None}

        critique_feedback = self._build_critique(semantic, verifier, critic_assessment, citation_verified_ratio)
        return {"score": score, "decision": "revise", "critique_feedback": critique_feedback}

    def _build_critique(
        self,
        semantic: float,
        verifier: float,
        critic_assessment: Optional[dict],
        citation_verified_ratio: float,
    ) -> str:
        issues = []
        if citation_verified_ratio < 0.5:
            issues.append(
                "citations could not be verified against the source text — cite exact "
                "page/chunk ids that appear in the retrieved passages, with excerpts copied verbatim"
            )
        hallucination_rate = (critic_assessment or {}).get('hallucination_rate')
        if isinstance(hallucination_rate, (int, float)) and hallucination_rate > 0.3:
            issues.append(f"hallucination_rate {hallucination_rate:.2f} exceeds 0.3 — remove unsupported claims")
        if verifier < 0.3:
            issues.append("the summary drifts from the retrieved passages — stay closer to the source wording")
        if semantic < 0.3:
            issues.append(
                "the summary is not semantically close to the retrieved passages — reground it in the provided text"
            )
        if not issues:
            issues.append("overall reliability score is below threshold — tighten factual grounding and citations")
        return "; ".join(issues)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_evaluator.py -v`
Expected: 7 passed

- [ ] **Step 5: Lint and commit**

```bash
.venv/Scripts/python.exe -m isort src/evaluator.py tests/test_evaluator.py
.venv/Scripts/python.exe -m flake8 --max-line-length=120 src/evaluator.py tests/test_evaluator.py
PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q
git add src/evaluator.py tests/test_evaluator.py
git commit -m "Replace regex citation check + flat score with ReliabilityEvaluator"
```

---

### Task 6: Real LangGraph StateGraph orchestration

**Files:**
- Modify: `src/langgraph_agents.py` (full rewrite)
- Modify: `requirements.txt` (pin `langgraph>=1.2.0,<2.0.0`)
- Modify: `tests/test_orchestrator.py` (full rewrite)

**Interfaces:**
- Consumes: `RetrieverAgent`, `SummarizerAgent`, `CriticAgent`, `CitationVerifierAgent` from `src/agents.py` (Task 4); `ReliabilityEvaluator` from `src/evaluator.py` (Task 5).
- Produces:
  - `GraphState` (TypedDict): `query: str`, `retrieved_chunks: List[dict]`, `summary: Optional[dict]`, `critic_assessment: Optional[dict]`, `citation_verification: Optional[dict]`, `reliability_score: Optional[int]`, `reliability_decision: Optional[str]`, `critique_feedback: Optional[str]`, `attempt: int`, `max_attempts: int`, `history: List[dict]`, `degraded_mode: bool`.
  - `build_initial_state(query: str, max_attempts: int = 3) -> GraphState`.
  - `make_orchestrator(store, embedder, llm=None, accept_threshold=70, top_k=5, summarizer=None, critic=None, citation_verifier=None, evaluator=None)` returns a compiled LangGraph graph exposing `.invoke(state) -> dict` and `.stream(state, stream_mode="values")`. `summarizer`/`critic`/`citation_verifier`/`evaluator` are optional dependency-injection overrides for tests. Used by Task 7's `app.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_orchestrator.py
import numpy as np

from src.langgraph_agents import build_initial_state, make_orchestrator


class DummyStore:
    def __init__(self, score=0.9, ids=(1,)):
        self.score = score
        self.ids = ids

    def search(self, q_emb, top_k=5):
        return [(self.score, {"id": i, "source": {"page": 1}, "text": "source text"}) for i in self.ids]


class DummyEmbedder:
    def embed_texts(self, texts, batch_size=64, normalize=True):
        return np.ones((len(texts), 8), dtype="float32")


class ScriptedSummarizer:
    """Returns each entry in `responses` in order, one per call to summarize()."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.llm_available = True

    def summarize(self, chunks, critique_feedback=None, previous_summary=None):
        self.calls.append({"critique_feedback": critique_feedback, "previous_summary": previous_summary})
        return self.responses.pop(0)


class FixedCritic:
    def __init__(self, assessment):
        self.assessment = assessment
        self.llm_available = True

    def assess(self, summary_text):
        return self.assessment


def test_graph_accepts_a_high_quality_first_attempt():
    store = DummyStore(score=0.95, ids=(1,))
    embedder = DummyEmbedder()
    summarizer = ScriptedSummarizer(
        [{"summary": "Good summary.", "citations": [{"chunk_id": 1, "page": 1, "excerpt": "source text"}],
          "valid": True}]
    )
    critic = FixedCritic({"confidence": 0.9, "hallucination_rate": 0.05, "notes": "fine"})

    graph = make_orchestrator(store, embedder, summarizer=summarizer, critic=critic)
    final_state = graph.invoke(build_initial_state("what is this paper about?", max_attempts=3))

    assert final_state["reliability_decision"] == "accept"
    assert final_state["attempt"] == 1
    assert len(summarizer.calls) == 1


def test_graph_revises_a_low_quality_summary_then_accepts():
    store = DummyStore(score=0.5, ids=(1,))
    embedder = DummyEmbedder()
    bad = {"summary": "Bad summary.", "citations": [{"chunk_id": 999, "page": 9, "excerpt": "nope"}],
           "valid": True}
    good = {"summary": "Good revised summary.", "citations": [{"chunk_id": 1, "page": 1, "excerpt": "source text"}],
            "valid": True}
    summarizer = ScriptedSummarizer([bad, good])
    critic = FixedCritic({"confidence": 0.9, "hallucination_rate": 0.05, "notes": "fine"})

    graph = make_orchestrator(store, embedder, summarizer=summarizer, critic=critic)
    final_state = graph.invoke(build_initial_state("what is this paper about?", max_attempts=3))

    assert final_state["attempt"] == 2
    assert final_state["reliability_decision"] == "accept"
    assert summarizer.calls[1]["critique_feedback"] is not None
    assert "citation" in summarizer.calls[1]["critique_feedback"]


def test_graph_returns_best_effort_with_low_confidence_after_exhausting_retries():
    store = DummyStore(score=0.0, ids=(1,))
    embedder = DummyEmbedder()
    always_bad = {"summary": "Always bad.", "citations": [], "valid": True}
    summarizer = ScriptedSummarizer([always_bad, always_bad, always_bad])
    critic = FixedCritic({"confidence": 0.1, "hallucination_rate": 0.9, "notes": "bad"})

    graph = make_orchestrator(store, embedder, summarizer=summarizer, critic=critic)
    final_state = graph.invoke(build_initial_state("what is this paper about?", max_attempts=3))

    assert final_state["reliability_decision"] == "exhausted"
    assert final_state["attempt"] == 3
    assert len(final_state["history"]) == 3
    assert len(summarizer.calls) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_orchestrator.py -v`
Expected: FAIL — `build_initial_state` doesn't exist, `make_orchestrator` doesn't accept `summarizer=`/`critic=` yet.

- [ ] **Step 3: Rewrite `src/langgraph_agents.py`**

```python
"""LangGraph orchestrator: a real StateGraph wiring the research-assistant
agents into a reliability-gated revision loop.
"""
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents import CitationVerifierAgent, CriticAgent, RetrieverAgent, SummarizerAgent
from src.evaluator import ReliabilityEvaluator


class GraphState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]
    critic_assessment: Optional[Dict[str, Any]]
    citation_verification: Optional[Dict[str, Any]]
    reliability_score: Optional[int]
    reliability_decision: Optional[str]
    critique_feedback: Optional[str]
    attempt: int
    max_attempts: int
    history: List[Dict[str, Any]]
    degraded_mode: bool


def build_initial_state(query: str, max_attempts: int = 3) -> GraphState:
    """Build a fresh GraphState dict to pass into `compiled_graph.invoke()`/`.stream()`."""
    return {
        "query": query,
        "retrieved_chunks": [],
        "summary": None,
        "critic_assessment": None,
        "citation_verification": None,
        "reliability_score": None,
        "reliability_decision": None,
        "critique_feedback": None,
        "attempt": 0,
        "max_attempts": max_attempts,
        "history": [],
        "degraded_mode": False,
    }


def make_orchestrator(
    store,
    embedder,
    llm: Optional[str] = None,
    accept_threshold: int = 70,
    top_k: int = 5,
    summarizer=None,
    critic=None,
    citation_verifier=None,
    evaluator=None,
):
    """Build and compile the reliability-gated multi-agent graph.

    `summarizer`/`critic`/`citation_verifier`/`evaluator` are optional
    dependency-injection overrides (used by tests); omit them for real use.
    """
    retriever = RetrieverAgent(store, embedder)
    summarizer = summarizer or SummarizerAgent(llm_model=(llm or "gpt-4o-mini"))
    critic = critic or CriticAgent(llm_model=(llm or "gpt-4o-mini"))
    citation_verifier = citation_verifier or CitationVerifierAgent()
    evaluator = evaluator or ReliabilityEvaluator(accept_threshold=accept_threshold)

    def retriever_node(state: GraphState) -> dict:
        results = retriever.retrieve(state["query"], top_k=top_k)
        retrieved = [
            {"text": meta.get("text", ""), "source": meta.get("source", {}) or {}, "id": meta.get("id")}
            for _, meta in results
        ]
        return {"retrieved_chunks": retrieved, "degraded_mode": not summarizer.llm_available}

    def summarizer_node(state: GraphState) -> dict:
        previous = state.get("summary")
        previous_summary = previous.get("summary") if previous else None
        result = summarizer.summarize(
            state["retrieved_chunks"],
            critique_feedback=state.get("critique_feedback"),
            previous_summary=previous_summary,
        )
        return {"summary": result, "attempt": state["attempt"] + 1}

    def critic_node(state: GraphState) -> dict:
        summary_text = (state.get("summary") or {}).get("summary", "")
        return {"critic_assessment": critic.assess(summary_text)}

    def citation_verifier_node(state: GraphState) -> dict:
        citations = (state.get("summary") or {}).get("citations", [])
        return {"citation_verification": citation_verifier.verify(citations, state["retrieved_chunks"])}

    def reliability_evaluator_node(state: GraphState) -> dict:
        summary_text = (state.get("summary") or {}).get("summary", "")
        citation_verification = state.get("citation_verification") or {}
        verdict = evaluator.evaluate(
            summary_text,
            state["retrieved_chunks"],
            store,
            embedder,
            critic_assessment=state.get("critic_assessment"),
            citation_verified_ratio=citation_verification.get("verified_ratio", 0.0),
            attempt=state["attempt"],
            max_attempts=state["max_attempts"],
        )
        history_entry = {
            "attempt": state["attempt"],
            "summary": state.get("summary"),
            "score": verdict["score"],
        }
        return {
            "reliability_score": verdict["score"],
            "reliability_decision": verdict["decision"],
            "critique_feedback": verdict["critique_feedback"],
            "history": state["history"] + [history_entry],
        }

    def route_on_reliability(state: GraphState) -> str:
        return state["reliability_decision"]

    graph = StateGraph(GraphState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("citation_verifier", citation_verifier_node)
    graph.add_node("reliability_evaluator", reliability_evaluator_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("summarizer", "citation_verifier")
    graph.add_edge(["critic", "citation_verifier"], "reliability_evaluator")
    graph.add_conditional_edges(
        "reliability_evaluator",
        route_on_reliability,
        {"accept": END, "exhausted": END, "revise": "summarizer"},
    )

    return graph.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_orchestrator.py -v`
Expected: 3 passed

- [ ] **Step 5: Pin langgraph in requirements.txt**

Edit `requirements.txt`, change the `langgraph` line to:

```
langgraph>=1.2.0,<2.0.0
```

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m isort src/langgraph_agents.py tests/test_orchestrator.py
.venv/Scripts/python.exe -m flake8 --max-line-length=120 src/langgraph_agents.py tests/test_orchestrator.py
PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q
git add src/langgraph_agents.py tests/test_orchestrator.py requirements.txt
git commit -m "Replace non-functional LangGraph facade with a real StateGraph revision loop"
```

---

### Task 7: Wire the Streamlit UI to the compiled graph

**Files:**
- Modify: `app.py` (remove `call_with_retries_validate`, lines 1-62; rewrite the query-handling block, lines ~164-336)
- Modify: `README.md` (replace the stale "placeholder agents" / "crash-course scaffold" description)

The manual "Regenerate structured summary" button (old `app.py:327-335`) is removed: the graph's own revision loop now retries automatically, so a manual regenerate button duplicates that behavior. The separate "Verifier overlap" panel (old `app.py:246-268`, backed by the now-deleted `VerifierAgent`) is removed since `ReliabilityEvaluator` already folds that overlap signal into `reliability_score`.

**Interfaces:**
- Consumes: `build_initial_state`, `make_orchestrator` from `src/langgraph_agents.py` (Task 6).
- Produces: no new importable interface — this is the UI leaf of the graph.

- [ ] **Step 1: Rewrite `app.py`**

Replace lines 1-10 (imports) with:

```python
import tempfile
from pathlib import Path

import streamlit as st

from src.embeddings import EmbeddingModel
from src.ingest import load_and_chunk
from src.langgraph_agents import build_initial_state, make_orchestrator
from src.vectorstore import FaissStore
```

Delete the `call_with_retries_validate` function (old lines 17-62) entirely — it now lives in `src/retry.py` and is used inside `src/agents.py`, not in `app.py`.

Keep the mode selector, Feedback Dashboard section, and PDF upload/embedding/index-building section (old lines 65-163) unchanged.

Replace the query-handling block (old lines 164-336, everything from `if 'store' in st.session_state:` through the end of the `if query:` block) with:

```python
    if 'store' in st.session_state:
        query = st.text_input("Ask a question about the uploaded paper")
        if query:
            store = st.session_state['store']
            embedder = st.session_state['embedder']

            graph = make_orchestrator(store, embedder)
            initial_state = build_initial_state(query, max_attempts=3)

            status_placeholder = st.empty()
            final_state = initial_state
            for state_update in graph.stream(initial_state, stream_mode="values"):
                final_state = state_update
                status_placeholder.info(f"Running multi-agent pipeline... (attempt {final_state.get('attempt', 0)})")
            status_placeholder.empty()

            st.header("Retrieved chunks")
            for chunk in final_state['retrieved_chunks']:
                src = chunk.get('source', {})
                st.write(f"page: {src.get('page', 'N/A')}")
                st.write(chunk.get('text', '')[:1000])

            if final_state.get('degraded_mode'):
                st.warning(
                    "Running in degraded mode: no OpenAI API key/SDK detected, so summarization "
                    "and scoring are using heuristic fallbacks instead of an LLM."
                )

            summary_res = final_state.get('summary') or {}
            summary_text = summary_res.get('summary', '')
            st.header("Summary")
            st.write(summary_text)
            if not summary_res.get('valid', True):
                st.error("Summarizer did not return valid structured citations. Summary marked as invalid.")

            st.header("Critic")
            st.json(final_state.get('critic_assessment') or {})

            st.header("Citation verification")
            st.json(final_state.get('citation_verification') or {})

            reliability_score = final_state.get('reliability_score')
            st.metric(label="Reliability (0-100)", value=reliability_score)

            decision = final_state.get('reliability_decision')
            if decision == "exhausted":
                st.warning(
                    f"Reliability score stayed below threshold after {final_state.get('attempt')} attempts "
                    "— showing the best attempt. Treat this summary as low-confidence."
                )
            elif decision == "accept":
                st.success(f"Reliability check passed after {final_state.get('attempt')} attempt(s).")

            st.session_state['last_query'] = query
            st.session_state['last_summary'] = summary_text
            st.session_state['last_assessment'] = final_state.get('critic_assessment')
            st.session_state['last_confidence'] = reliability_score

            allow_accept = summary_res.get('valid', False) or decision == "accept"

            st.write("Was this answer accurate?")
            col1, col2 = st.columns(2)
            if allow_accept and col1.button("Accurate"):
                import datetime
                fb = {
                    "query": query,
                    "summary": summary_text,
                    "confidence": reliability_score,
                    "label": "accurate",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "paper_id": st.session_state.get('paper_id'),
                }
                st.session_state.setdefault('feedback', []).append(fb)
                import json

                with open("feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.success("Feedback saved: accurate")

            if allow_accept and col2.button("Hallucinated"):
                import datetime
                fb = {
                    "query": query,
                    "summary": summary_text,
                    "confidence": reliability_score,
                    "label": "hallucinated",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "paper_id": st.session_state.get('paper_id'),
                }
                st.session_state.setdefault('feedback', []).append(fb)
                import json

                with open("feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.error("Feedback saved: hallucinated")
            if not allow_accept:
                st.info("Summary was not structurally valid — ask the question again to retry.")

        st.sidebar.header("Index storage")
        if st.sidebar.button("Load saved index"):
            try:
                store = FaissStore.load("paper_index")
                st.session_state['store'] = store
                st.success("Loaded index from 'paper_index'")
            except Exception as e:
                st.error(f"Failed to load index: {e}")

        if st.sidebar.button("Save current index"):
            try:
                st.session_state['store'].save("paper_index")
                st.success("Saved index to 'paper_index'")
            except Exception as e:
                st.error(f"Failed to save index: {e}")
```

- [ ] **Step 2: Manual smoke test**

This repo has no existing `app.py` test harness (Streamlit's top-level script style isn't pytest-importable), so verify by hand:

Run: `.venv/Scripts/python.exe -m streamlit run app.py`

1. Upload `examples/sample.pdf`, click "Build embeddings & index" — confirm it reaches "Index ready" with no traceback.
2. Type a question, submit — confirm the "Running multi-agent pipeline... (attempt N)" status updates at least once, then a Summary, Critic JSON, Citation verification JSON, and a "Reliability (0-100)" metric all render with no traceback.
3. Confirm either a green "Reliability check passed..." message or (if `OPENAI_API_KEY` isn't set) a "Running in degraded mode..." warning appears — both are valid depending on whether a key is configured.
4. Click "Accurate" — confirm `feedback.jsonl` gets a new line and a success toast appears.
5. Switch the sidebar Mode to "Feedback Dashboard" — confirm it renders without error using the row just added.

- [ ] **Step 3: Update README.md**

Replace the line:

```
This repo is a crash-course scaffold. Replace the placeholder agents with LangGraph orchestrations and LLM calls as needed.
```

with:

```
The retriever, summarizer, critic, citation-verifier, and reliability-evaluator agents run as a real LangGraph `StateGraph` (see `src/langgraph_agents.py`). If the reliability evaluator scores a summary below its threshold, it sends the summary back to the summarizer with specific critique feedback for revision, up to `max_attempts` times; after that it returns the best attempt flagged low-confidence rather than failing outright.
```

Update the `Files` list's `src/agents.py` line from:

```
- `src/agents.py`: simple Retriever / Summarizer / Critic placeholders
```

to:

```
- `src/agents.py`: Retriever, Summarizer, Critic, and CitationVerifier agents
- `src/evaluator.py`: ReliabilityEvaluator — aggregates agent signals into a score and a revise/accept/exhausted decision
- `src/langgraph_agents.py`: the LangGraph StateGraph wiring the agents into a reliability-gated revision loop
```

- [ ] **Step 4: Full-suite verification, lint, and commit**

```bash
.venv/Scripts/python.exe -m isort src tests app.py
.venv/Scripts/python.exe -m flake8 --max-line-length=120 src tests app.py
PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q
git add app.py README.md
git commit -m "Drive the Streamlit UI from the compiled reliability-gated graph"
```

---

## Self-Review Notes

- **Spec coverage:** every spec section maps to a task — Architecture/State/Components → Task 6 (+ Tasks 1-5 for the pieces it wires together); Error handling & degraded mode → Task 1 (retry), Task 4 (heuristic fallbacks + `_UNSET` client override), Task 6 (`degraded_mode` flag), Task 7 (UI warning); Testing plan → each task's own test file plus Task 6's golden end-to-end routing tests.
- **Placeholder scan:** no TBD/TODO markers; every step has runnable code.
- **Type consistency:** `SummarizerAgent.summarize()` return shape (`{"summary", "citations", "valid"}`) matches what `summarizer_node`, `citation_verifier_node`, and `app.py` all read; `CitationVerifierAgent.verify()` / `ReliabilityEvaluator.evaluate()` return shapes match what `reliability_evaluator_node` and `app.py` read; `GraphState` keys match every node's return dict keys across Task 6 and the tests in Task 6/Task 7.
