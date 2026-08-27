# Cross-Page Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop losing information at two points in ingestion: sentences split across a PDF page break, and pages whose text extraction silently returns nothing.

**Architecture:** Rewrite `chunk_pages_to_chunks` to chunk over the whole document as one text stream (via a concatenate + offset→page lookup table) instead of resetting at every page boundary, so a chunk can carry a `start_page`/`end_page` range. Alongside that, add a small pure-function check that flags pages whose extracted text is empty or suspiciously short, surfaced as a UI warning. Both land in the same pass since they touch the same ingestion entry point (`load_and_chunk`), but are functionally independent.

**Tech Stack:** Python 3.10, `pdfplumber`, `pydantic`, `pytest` — no new dependencies (the offset lookup uses the stdlib `bisect` module).

**Spec:** `docs/superpowers/specs/2026-08-27-cross-page-chunking-design.md` (including its "Addendum: extraction-gap detection" section).

## Global Constraints

- Don't break the existing 51 passing tests or the CI lint gate (`isort --check-only src tests app.py` / `python -m isort --check-only src tests app.py`, `flake8 --max-line-length=120 src tests app.py` / `python -m flake8 --max-line-length=120 src tests app.py`).
- No new hard dependencies (the design uses only stdlib `bisect`).
- No backward-compatibility shims for the `source` dict shape or `load_and_chunk`'s return type — every caller is inside this repo and gets updated directly.
- Windows-first repo (paths, line endings) — this plan doesn't add filesystem path logic, so no new risk here.

---

## Task 1: Extraction-gap detection and page-label formatting helpers

**Files:**
- Modify: `src/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Produces: `MIN_PAGE_TEXT_CHARS: int` (constant, value 20); `flagged_low_text_pages(pages: List[Dict], min_chars: int = MIN_PAGE_TEXT_CHARS) -> List[int]`; `format_page_label(source: dict) -> Optional[str]`.

These two helpers are independent of the chunking rewrite in Task 2 — `flagged_low_text_pages` only inspects the raw per-page text before chunking starts, and `format_page_label` only formats a chunk's `source` dict for display. Landing them first keeps Task 2's diff focused on the chunking algorithm itself.

- [ ] **Step 1: Write the failing tests**

First, update the import line at the top of `tests/test_ingest.py`.

Find:
```python
import tempfile

from reportlab.pdfgen import canvas

from src.ingest import load_and_chunk
```

Replace with:
```python
import tempfile

from reportlab.pdfgen import canvas

from src.ingest import flagged_low_text_pages, format_page_label, load_and_chunk
```

Then add the following tests to `tests/test_ingest.py`, above the existing `test_load_and_chunk`:

```python
def test_flagged_low_text_pages_flags_empty_and_near_empty():
    pages = [
        {"page": 1, "text": "Plenty of real content on this page to pass the threshold easily."},
        {"page": 2, "text": ""},
        {"page": 3, "text": "x"},
    ]
    assert flagged_low_text_pages(pages) == [2, 3]


def test_flagged_low_text_pages_empty_when_all_pages_have_content():
    pages = [{"page": 1, "text": "This page has more than twenty characters of real text."}]
    assert flagged_low_text_pages(pages) == []


def test_format_page_label_single_page():
    assert format_page_label({"start_page": 3, "end_page": 3}) == "3"


def test_format_page_label_spanning_pages():
    assert format_page_label({"start_page": 3, "end_page": 4}) == "3-4"


def test_format_page_label_missing_start_page_is_none():
    assert format_page_label({}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ImportError: cannot import name 'flagged_low_text_pages' from 'src.ingest'`

- [ ] **Step 3a: Update the top-of-file import in `src/ingest.py`**

Find:
```python
import re
from typing import Dict, List

import pdfplumber
```

Replace with:
```python
import re
from typing import Dict, List, Optional

import pdfplumber
```

- [ ] **Step 3b: Add `MIN_PAGE_TEXT_CHARS` and `flagged_low_text_pages` after `load_pdf_pages`**

Find:
```python
def load_pdf_pages(path: str) -> List[Dict]:
    """Load PDF and return a list of pages with text and metadata.

    Returns a list of dicts: {"page": int, "text": str}
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
    return pages


_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
```

Replace with:
```python
def load_pdf_pages(path: str) -> List[Dict]:
    """Load PDF and return a list of pages with text and metadata.

    Returns a list of dicts: {"page": int, "text": str}
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
    return pages


MIN_PAGE_TEXT_CHARS = 20


def flagged_low_text_pages(pages: List[Dict], min_chars: int = MIN_PAGE_TEXT_CHARS) -> List[int]:
    """Page numbers whose extracted text is empty or suspiciously short.

    Likely scanned/image-only pages, or a layout pdfplumber couldn't parse.
    """
    return [p["page"] for p in pages if len(p.get("text", "").strip()) < min_chars]


_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
```

- [ ] **Step 3c: Add `format_page_label` after `split_sentences`**

Find:
```python
def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # naive sentence split by punctuation + whitespace
    sentences = _SENTENCE_SPLIT.split(text)
    # strip and filter
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages_to_chunks(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
```

Replace with:
```python
def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # naive sentence split by punctuation + whitespace
    sentences = _SENTENCE_SPLIT.split(text)
    # strip and filter
    return [s.strip() for s in sentences if s.strip()]


def format_page_label(source: dict) -> Optional[str]:
    """Render a chunk's page(s) as a display label: "3", or "3-4" when it spans pages."""
    start_page = source.get('start_page')
    if start_page is None:
        return None
    end_page = source.get('end_page')
    if end_page is not None and end_page != start_page:
        return f"{start_page}-{end_page}"
    return str(start_page)


def chunk_pages_to_chunks(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: PASS (6 tests: the original `test_load_and_chunk` plus the 5 new ones)

- [ ] **Step 5: Lint**

Run: `python -m isort --check-only src/ingest.py tests/test_ingest.py && python -m flake8 --max-line-length=120 src/ingest.py tests/test_ingest.py`
Expected: no output (clean). If isort complains, run `python -m isort src/ingest.py tests/test_ingest.py` and re-check.

- [ ] **Step 6: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "Add extraction-gap detection and page-range label formatting"
```

---

## Task 2: Cross-page chunking rewrite

**Files:**
- Modify: `src/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `flagged_low_text_pages` from Task 1.
- Produces: `_build_document(pages: List[Dict]) -> Tuple[str, List[Tuple[int, int]]]`; `_page_at(offset: int, page_offsets: List[Tuple[int, int]]) -> int`; `chunk_pages_to_chunks(pages, chunk_size=1000, overlap=200) -> List[Dict]` (unchanged signature, changed `source` shape — see below); `load_and_chunk(path, chunk_size=1000, overlap=200) -> Tuple[List[Dict], List[int]]` (changed from `-> List[Dict]`).

This is the core rewrite from the spec: `chunk_pages_to_chunks` no longer resets at page boundaries. Each chunk's `source` becomes `{"start_page": int, "end_page": int, "start_char": int, "end_char": int}` instead of `{"page": int, "start_char": int, "end_char": int}`. `start_char`/`end_char` now index into the whole concatenated document, not a single page's text (confirmed in the spec that nothing outside `ingest.py` reads these).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ingest.py`, after the existing tests:

```python
def make_two_page_pdf(path: str, page1_text: str, page2_text: str):
    c = canvas.Canvas(path)
    c.drawString(72, 720, page1_text)
    c.showPage()
    c.drawString(72, 720, page2_text)
    c.showPage()
    c.save()


def test_sentence_spanning_a_page_break_stays_in_one_chunk():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    make_two_page_pdf(
        tmp.name,
        "Intro sentence here. The argument continues across the page break without",
        "stopping until it finally ends here. A short new sentence follows.",
    )
    chunks, _flagged = load_and_chunk(tmp.name, chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    assert "continues across the page break without" in chunks[0]["text"]
    assert "until it finally ends here" in chunks[0]["text"]
    src = chunks[0]["source"]
    assert src["start_page"] == 1
    assert src["end_page"] == 2


def test_chunk_within_a_single_page_has_equal_start_and_end_page():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    make_sample_pdf(tmp.name, "A short single-page document with one plain sentence in it.")
    chunks, _flagged = load_and_chunk(tmp.name, chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    assert chunks[0]["source"]["start_page"] == chunks[0]["source"]["end_page"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ingest.py -v -k "spanning_a_page_break or single_page"`
Expected: FAIL — `test_sentence_spanning_a_page_break_stays_in_one_chunk` fails on `assert len(chunks) == 1` (today's code produces 2 chunks, one cut off mid-sentence at the page break) or on `assert src["end_page"] == 2` (today's `source` dict has no `end_page` key at all, so this raises `KeyError`).

- [ ] **Step 3: Replace the import line, add `_build_document`/`_page_at`, and rewrite `chunk_pages_to_chunks` + `load_and_chunk`**

Find (top of `src/ingest.py`, as left by Task 1):
```python
import re
from typing import Dict, List, Optional

import pdfplumber
```

Replace with:
```python
import bisect
import re
from typing import Dict, List, Optional, Tuple

import pdfplumber
```

Find the entire `chunk_pages_to_chunks` and `load_and_chunk` functions (everything from `def chunk_pages_to_chunks` to the end of the file):

```python
def chunk_pages_to_chunks(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Create overlapping chunks across pages, preserving source metadata.

    Each chunk is a dict: {"text": str, "source": {"page": int, "start_char": int, "end_char": int}}
    """
    chunks: List[Dict] = []
    for page in pages:
        page_num = page.get("page")
        text = page.get("text", "")
        if not text:
            continue
        sentences = split_sentences(text)
        cur = ""
        for sent in sentences:
            if cur:
                next_text = cur + " " + sent
            else:
                next_text = sent

            if len(next_text) >= chunk_size:
                # finalize current chunk
                start_idx = text.find(cur) if cur else text.find(sent)
                end_idx = start_idx + len(cur)
                chunks.append(
                    {
                        "text": cur.strip(),
                        "source": {
                            "page": page_num,
                            "start_char": start_idx,
                            "end_char": end_idx,
                        },
                    }
                )
                # prepare next chunk with overlap
                overlap_text = next_text[-overlap:]
                cur = overlap_text
            else:
                cur = next_text

        if cur:
            start_idx = text.find(cur)
            end_idx = start_idx + len(cur)
            chunks.append(
                {
                    "text": cur.strip(),
                    "source": {
                        "page": page_num,
                        "start_char": start_idx,
                        "end_char": end_idx,
                    },
                }
            )

    return chunks


def load_and_chunk(path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    pages = load_pdf_pages(path)
    return chunk_pages_to_chunks(pages, chunk_size=chunk_size, overlap=overlap)
```

Replace with:
```python
def _build_document(pages: List[Dict]) -> Tuple[str, List[Tuple[int, int]]]:
    """Concatenate page texts into one document string, plus an offset->page lookup.

    Returns (document_text, page_offsets), where page_offsets is a list of
    (start_offset, page_num) pairs -- one per non-empty page, sorted by offset.
    """
    parts = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for page in pages:
        text = page.get("text", "")
        if not text:
            continue
        offsets.append((cursor, page.get("page")))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the "\n" join separator
    return "\n".join(parts), offsets


def _page_at(offset: int, page_offsets: List[Tuple[int, int]]) -> int:
    """Binary-search the page owning a character offset in the concatenated document."""
    starts = [o for o, _ in page_offsets]
    idx = max(bisect.bisect_right(starts, offset) - 1, 0)
    return page_offsets[idx][1]


def chunk_pages_to_chunks(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Create overlapping chunks across the whole document, preserving page attribution.

    Chunks are cut on chunk_size/overlap alone -- page boundaries no longer force a
    cut, so a sentence that spans two pages stays in one chunk. Each chunk is a dict:
    {"text": str, "source": {"start_page": int, "end_page": int, "start_char": int, "end_char": int}}.
    start_char/end_char index into the concatenated document, not a single page's text.
    """
    document, page_offsets = _build_document(pages)
    if not document:
        return []

    sentences = split_sentences(document)
    chunks: List[Dict] = []
    cur = ""
    cur_start = 0
    pos = 0

    for sent in sentences:
        sent_start = pos if not cur else pos + 1
        if cur:
            next_text = cur + " " + sent
        else:
            next_text = sent
            cur_start = sent_start

        if len(next_text) >= chunk_size:
            start_idx = cur_start
            end_idx = start_idx + len(cur)
            chunks.append(
                {
                    "text": cur.strip(),
                    "source": {
                        "start_page": _page_at(start_idx, page_offsets),
                        "end_page": _page_at(max(end_idx - 1, start_idx), page_offsets),
                        "start_char": start_idx,
                        "end_char": end_idx,
                    },
                }
            )
            # prepare next chunk with overlap
            overlap_text = next_text[-overlap:]
            cur = overlap_text
            sent_end = sent_start + len(sent)
            cur_start = sent_end - len(overlap_text)
            pos = sent_end
        else:
            cur = next_text
            pos = sent_start + len(sent)

    if cur:
        end_idx = cur_start + len(cur)
        chunks.append(
            {
                "text": cur.strip(),
                "source": {
                    "start_page": _page_at(cur_start, page_offsets),
                    "end_page": _page_at(max(end_idx - 1, cur_start), page_offsets),
                    "start_char": cur_start,
                    "end_char": end_idx,
                },
            }
        )

    return chunks


def load_and_chunk(path: str, chunk_size: int = 1000, overlap: int = 200) -> Tuple[List[Dict], List[int]]:
    """Load a PDF, chunk it, and report which pages produced little or no text.

    Returns (chunks, flagged_pages) -- flagged_pages are page numbers likely to be
    scanned images or layouts pdfplumber couldn't parse (see flagged_low_text_pages).
    """
    pages = load_pdf_pages(path)
    chunks = chunk_pages_to_chunks(pages, chunk_size=chunk_size, overlap=overlap)
    return chunks, flagged_low_text_pages(pages)
```

- [ ] **Step 4: Update the existing `test_load_and_chunk` to unpack the new tuple**

Find (in `tests/test_ingest.py`):
```python
def test_load_and_chunk():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    make_sample_pdf(tmp.name, "This is a simple PDF for testing ingestion. It has a few sentences.")
    chunks = load_and_chunk(tmp.name, chunk_size=200, overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for c in chunks:
        assert "text" in c and "source" in c
```

Replace with:
```python
def test_load_and_chunk():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    make_sample_pdf(tmp.name, "This is a simple PDF for testing ingestion. It has a few sentences.")
    chunks, flagged_pages = load_and_chunk(tmp.name, chunk_size=200, overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for c in chunks:
        assert "text" in c and "source" in c
    assert isinstance(flagged_pages, list)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: PASS (8 tests: 6 from Task 1 plus the 2 new ones from this task)

- [ ] **Step 6: Lint**

Run: `python -m isort --check-only src/ingest.py tests/test_ingest.py && python -m flake8 --max-line-length=120 src/ingest.py tests/test_ingest.py`
Expected: no output (clean)

- [ ] **Step 7: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "Chunk across the whole document instead of resetting at page breaks"
```

---

## Task 3: Citation schema accepts a page-range string

**Files:**
- Modify: `src/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `Citation.page: Optional[str]` (was `Optional[int]`); `CitationCheck.page: Optional[str]` (was `Optional[int]`).

The LLM echoes back whatever page label it was shown in a passage header (`format_page_label`'s output — `"3"` or `"3-4"`). A free-form string accommodates both without a schema-level range type. `CitationVerifierAgent` (Task 4 doesn't touch this — confirmed in the spec's Investigation findings) never parses `page`, only passes it through, so this is a pure type widening.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas.py`, after `test_citation_verification_aggregates_checks`:

```python
def test_citation_check_accepts_a_page_range_string():
    check = CitationCheck(chunk_id=1, page="3-4", found_in_chunks=True, text_match=True)
    assert check.page == "3-4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schemas.py -v -k page_range`
Expected: FAIL with `pydantic_core._pydantic_core.ValidationError: ... Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='3-4', ...]`

- [ ] **Step 3: Widen the schema**

Find (in `src/schemas.py`):
```python
class Citation(BaseModel):
    page: Optional[int]
    chunk_id: Optional[int]
    excerpt: Optional[str]
```

Replace with:
```python
class Citation(BaseModel):
    page: Optional[str]
    chunk_id: Optional[int]
    excerpt: Optional[str]
```

Find:
```python
class CitationCheck(BaseModel):
    chunk_id: Optional[int]
    page: Optional[int]
    found_in_chunks: bool
    text_match: bool
```

Replace with:
```python
class CitationCheck(BaseModel):
    chunk_id: Optional[int]
    page: Optional[str]
    found_in_chunks: bool
    text_match: bool
```

- [ ] **Step 4: Fix the now-stale existing test**

Find (in `tests/test_schemas.py`):
```python
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

Replace with:
```python
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
```

(This test would otherwise fail after Step 3's schema change — `page=2` on an `Optional[str]` field raises `ValidationError`, confirmed by hand before writing this plan.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Lint**

Run: `python -m isort --check-only src/schemas.py tests/test_schemas.py && python -m flake8 --max-line-length=120 src/schemas.py tests/test_schemas.py`
Expected: no output (clean)

- [ ] **Step 7: Commit**

```bash
git add src/schemas.py tests/test_schemas.py
git commit -m "Allow Citation/CitationCheck.page to carry a page-range string"
```

---

## Task 4: Page-range-aware passage headers in the Summarizer

**Files:**
- Modify: `src/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `format_page_label` from Task 1; `Citation.page: Optional[str]` from Task 3.
- Produces: `_format_passages` unchanged signature, changed header rendering (`page:3` or `page:3-4` instead of `page:3` derived from the old single `page` key).

This task also fixes `tests/test_agents.py`'s existing fixtures, which construct chunks with the old `{"source": {"page": N}}` shape and `Citation(page=N)` with an int — both now stale after Tasks 2 and 3.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agents.py`, after `test_summarizer_max_attempts_attr`:

```python
def test_summarizer_header_shows_a_page_range_for_a_spanning_chunk():
    agent = SummarizerAgent(client=None)
    chunks = [{"text": "Spans two pages.", "source": {"start_page": 3, "end_page": 4}, "id": 2}]
    res = agent.summarize(chunks)
    assert "[chunk_id:2 page:3-4]" in res["summary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agents.py -v -k page_range`
Expected: FAIL — today's `_format_passages` reads `src.get('page')`, which is absent from `{"start_page": 3, "end_page": 4}`, so the header renders as `[chunk_id:2]` (no `page:` at all), not `[chunk_id:2 page:3-4]`.

- [ ] **Step 3: Update `_format_passages` and the import line**

Find (top of `src/agents.py`):
```python
from typing import Dict, List, Optional

from src.llm_client import get_openai_client
from src.retry import call_with_retries_validate
from src.schemas import CriticAssessment, SummaryResponse
```

Replace with:
```python
from typing import Dict, List, Optional

from src.ingest import format_page_label
from src.llm_client import get_openai_client
from src.retry import call_with_retries_validate
from src.schemas import CriticAssessment, SummaryResponse
```

Find:
```python
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
```

Replace with:
```python
def _format_passages(chunks: List[dict]) -> str:
    passages = []
    for i, c in enumerate(chunks):
        if isinstance(c, dict):
            text = c.get('text', '')
            src = c.get('source', {}) or {}
            page_label = format_page_label(src)
            chunk_id = c.get('id') if c.get('id') is not None else i
            if page_label is not None:
                header = f"[chunk_id:{chunk_id} page:{page_label}]"
            else:
                header = f"[chunk_id:{chunk_id}]"
            passages.append(f"{header}\n{text}")
        else:
            passages.append(str(c))
    return "\n\n".join(passages)
```

- [ ] **Step 4: Update the system prompt's header-format description**

Find (in `SummarizerAgent.summarize`):
```python
        system = (
            "You are a helpful assistant that summarizes scientific paper snippets. "
            "Each passage is prefixed with metadata like [chunk_id:X page:Y]. "
            "Every citation you return must reference a chunk_id that actually appears "
            "in the passages, and its excerpt must be copied verbatim from that chunk's text."
        )
```

Replace with:
```python
        system = (
            "You are a helpful assistant that summarizes scientific paper snippets. "
            "Each passage is prefixed with metadata like [chunk_id:X page:Y], where Y may be a "
            "single page number or a range like 3-4 when a passage spans two pages. "
            "Every citation you return must reference a chunk_id that actually appears "
            "in the passages, and its excerpt must be copied verbatim from that chunk's text."
        )
```

- [ ] **Step 5: Fix the stale fixtures in `tests/test_agents.py`**

Find each of these four occurrences (there are two pairs with identical text — use each surrounding function to disambiguate, per the line numbers below):

In `test_summarizer_without_client_falls_back_to_heuristic`:
```python
    chunks = [{"text": "This is a test passage.", "source": {"page": 3}, "id": 7}]
```
Replace with:
```python
    chunks = [{"text": "This is a test passage.", "source": {"start_page": 3, "end_page": 3}, "id": 7}]
```

In `test_summarizer_uses_structured_output_when_client_available`:
```python
    parsed = SummaryResponse(
        summary="Short summary.",
        citations=[Citation(page=3, chunk_id=7, excerpt="test passage")],
    )
    agent = SummarizerAgent(client=FakeClient(FakeMessage(parsed=parsed)))
    chunks = [{"text": "This is a test passage.", "source": {"page": 3}, "id": 7}]
```
Replace with:
```python
    parsed = SummaryResponse(
        summary="Short summary.",
        citations=[Citation(page="3", chunk_id=7, excerpt="test passage")],
    )
    agent = SummarizerAgent(client=FakeClient(FakeMessage(parsed=parsed)))
    chunks = [{"text": "This is a test passage.", "source": {"start_page": 3, "end_page": 3}, "id": 7}]
```

In `test_summarizer_includes_critique_feedback_in_revision_prompt`:
```python
    chunks = [{"text": "Passage.", "source": {"page": 1}, "id": 1}]
```
Replace with:
```python
    chunks = [{"text": "Passage.", "source": {"start_page": 1, "end_page": 1}, "id": 1}]
```

In `test_summarizer_falls_back_when_client_raises`:
```python
    chunks = [{"text": "Passage text.", "source": {"page": 1}, "id": 1}]
```
Replace with:
```python
    chunks = [{"text": "Passage text.", "source": {"start_page": 1, "end_page": 1}, "id": 1}]
```

In `test_citation_verifier_flags_missing_chunk`:
```python
def test_citation_verifier_flags_missing_chunk():
    result = CitationVerifierAgent().verify(
        [{"chunk_id": 99, "page": 1, "excerpt": "nope"}],
        [{"id": 1, "text": "Some real text.", "source": {"page": 1}}],
    )
```
Replace with:
```python
def test_citation_verifier_flags_missing_chunk():
    result = CitationVerifierAgent().verify(
        [{"chunk_id": 99, "page": "1", "excerpt": "nope"}],
        [{"id": 1, "text": "Some real text.", "source": {"start_page": 1, "end_page": 1}}],
    )
```

In `test_citation_verifier_matches_excerpt_in_source_text`:
```python
def test_citation_verifier_matches_excerpt_in_source_text():
    result = CitationVerifierAgent().verify(
        [{"chunk_id": 1, "page": 1, "excerpt": "real text"}],
        [{"id": 1, "text": "Some real text right here.", "source": {"page": 1}}],
    )
```
Replace with:
```python
def test_citation_verifier_matches_excerpt_in_source_text():
    result = CitationVerifierAgent().verify(
        [{"chunk_id": 1, "page": "1", "excerpt": "real text"}],
        [{"id": 1, "text": "Some real text right here.", "source": {"start_page": 1, "end_page": 1}}],
    )
```

(`test_citation_verifier_no_citations_returns_zero_ratio`'s `"source": {}` needs no change — it has no `page` key either way.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_agents.py -v`
Expected: PASS (12 tests: the 11 existing plus the new range-header test)

- [ ] **Step 7: Lint**

Run: `python -m isort --check-only src/agents.py tests/test_agents.py && python -m flake8 --max-line-length=120 src/agents.py tests/test_agents.py`
Expected: no output (clean)

- [ ] **Step 8: Commit**

```bash
git add src/agents.py tests/test_agents.py
git commit -m "Render page ranges in the Summarizer's passage headers"
```

---

## Task 5: Wire extraction-gap warning and page-range display into `app.py`

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `load_and_chunk` (now returns `Tuple[List[Dict], List[int]]`) and `format_page_label` from `src.ingest`.

No existing automated test covers `app.py`'s Streamlit flow (the repo has no such tests today — this is unrelated to this plan). Verification for this task is a script that exercises the same functions `app.py` calls, plus lint/compile checks, following the same approach used to verify Phase 1's `app.py` wiring.

- [ ] **Step 1: Update the import**

Find (in `app.py`):
```python
from src.ingest import load_and_chunk
```

Replace with:
```python
from src.ingest import format_page_label, load_and_chunk
```

- [ ] **Step 2: Unpack the new tuple and show the extraction-gap warning**

Find:
```python
        st.info("Loading PDF and chunking...")
        chunks = load_and_chunk(path)
        st.success(f"Loaded {len(chunks)} chunks")
```

Replace with:
```python
        st.info("Loading PDF and chunking...")
        chunks, flagged_pages = load_and_chunk(path)
        st.success(f"Loaded {len(chunks)} chunks")
        if flagged_pages:
            page_list = ", ".join(str(p) for p in flagged_pages)
            st.warning(
                f"Pages {page_list} produced little or no extractable text — they may be scanned "
                "images or a layout pdfplumber can't parse. Content from these pages may be missing "
                "from answers."
            )
```

- [ ] **Step 3: Show a page range instead of a single page number for retrieved chunks**

Find:
```python
            st.header("Retrieved chunks")
            for chunk in final_state['retrieved_chunks']:
                src = chunk.get('source', {})
                st.write(f"page: {src.get('page', 'N/A')}")
                st.write(chunk.get('text', '')[:1000])
```

Replace with:
```python
            st.header("Retrieved chunks")
            for chunk in final_state['retrieved_chunks']:
                src = chunk.get('source', {})
                page_label = format_page_label(src)
                st.write(f"page: {page_label if page_label is not None else 'N/A'}")
                st.write(chunk.get('text', '')[:1000])
```

- [ ] **Step 4: Verify `app.py` compiles and imports cleanly**

Run: `python -m py_compile app.py`
Expected: no output (clean)

- [ ] **Step 5: Scripted verification of the two new behaviors**

Run this from the repo root (adjust the script path to your scratch directory):

```python
"""Verifies app.py's new load_and_chunk tuple-unpack and warning trigger,
against a real two-page PDF with a spanning sentence and a blank page."""
import sys

sys.path.insert(0, ".")

from reportlab.pdfgen import canvas  # noqa: E402

from src.ingest import format_page_label, load_and_chunk  # noqa: E402

path = "verify_phase2_tmp.pdf"
c = canvas.Canvas(path)
c.drawString(72, 720, "Intro sentence here. The argument continues across the page break without")
c.showPage()
c.drawString(72, 720, "stopping until it finally ends here.")
c.showPage()
c.showPage()  # a third, blank page -- should be flagged
c.save()

chunks, flagged_pages = load_and_chunk(path)
assert len(chunks) == 1, chunks
assert chunks[0]["source"]["start_page"] == 1
assert chunks[0]["source"]["end_page"] == 2
assert flagged_pages == [3], flagged_pages
assert format_page_label(chunks[0]["source"]) == "1-2"

import os  # noqa: E402
os.remove(path)
print("PHASE 2 VERIFICATION: PASS")
```

Expected output: `PHASE 2 VERIFICATION: PASS`

(As with Phase 1, driving the actual Streamlit browser UI wasn't possible in this environment — no `chromium-cli`, and the Chrome extension wasn't connected. If it becomes available, a manual pass — upload a PDF with a known blank/scanned page, confirm the warning banner appears with the right page numbers, and confirm a spanning chunk's page display reads e.g. "page: 3-4" — is worth doing once.)

- [ ] **Step 6: Run the full test suite and lint**

Run: `python -m pytest -q && python -m isort --check-only src tests app.py && python -m flake8 --max-line-length=120 src tests app.py`
Expected: all tests pass, lint clean. (`tests/test_evaluator.py` and `tests/test_orchestrator.py` still pass at this point even though their fixtures use the old `source` shape — see Task 6 for why.)

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "Surface extraction-gap warnings and page ranges in the UI"
```

---

## Task 6: Migrate remaining stale test fixtures

**Files:**
- Modify: `tests/test_evaluator.py`
- Modify: `tests/test_orchestrator.py`
- Modify: `examples/run_langgraph_demo.py`

**Interfaces:** none — this task changes no production code, only fixture data, for consistency with the `source` shape Task 2 introduced.

**Why this task exists even though nothing currently fails without it:** `ReliabilityEvaluator` (`src/evaluator.py`) and the LangGraph orchestrator (`src/langgraph_agents.py`) never read `chunk['source']['page']` — they only use `id` and pass chunk text through opaquely. So `tests/test_evaluator.py` and `tests/test_orchestrator.py` keep passing today even with the old `{"source": {"page": 1}}` shape in their fixtures. This task is pure consistency cleanup — leaving the old shape here would be a misleading example for a future reader who doesn't know the real chunk shape changed in Task 2. `examples/run_langgraph_demo.py`'s edit is different in kind: it calls `load_and_chunk`, whose *return type* actually changed (a tuple now), so its one line needs updating or the script raises `TypeError` the moment it runs.

**Known pre-existing issue, not fixed here:** `examples/run_langgraph_demo.py` calls `make_orchestrator(store, embedder)` expecting a 3-tuple `(retriever, summarizer, critic)` back; the current `make_orchestrator` (since the LangGraph rebuild in commit `79d4cd1`) returns a single compiled graph object instead. This script has been broken independent of this plan's changes. Only the `load_and_chunk` line is touched here — fixing the rest is out of scope.

- [ ] **Step 1: Update `tests/test_evaluator.py`'s fixture**

Find:
```python
    def search(self, q_emb, top_k=10):
        return [(self.score, {"id": i, "source": {"page": 1}}) for i in self.ids]
```

Replace with:
```python
    def search(self, q_emb, top_k=10):
        return [(self.score, {"id": i, "source": {"start_page": 1, "end_page": 1}}) for i in self.ids]
```

- [ ] **Step 2: Update `tests/test_orchestrator.py`'s fixtures**

Find:
```python
    def search(self, q_emb, top_k=5):
        return [(self.score, {"id": i, "source": {"page": 1}, "text": "source text"}) for i in self.ids]
```

Replace with:
```python
    def search(self, q_emb, top_k=5):
        return [
            (self.score, {"id": i, "source": {"start_page": 1, "end_page": 1}, "text": "source text"})
            for i in self.ids
        ]
```

Find:
```python
    summarizer = ScriptedSummarizer(
        [{"summary": "Good summary.", "citations": [{"chunk_id": 1, "page": 1, "excerpt": "source text"}],
          "valid": True}]
    )
```

Replace with:
```python
    summarizer = ScriptedSummarizer(
        [{"summary": "Good summary.", "citations": [{"chunk_id": 1, "page": "1", "excerpt": "source text"}],
          "valid": True}]
    )
```

Find:
```python
    bad = {"summary": "Bad summary.", "citations": [{"chunk_id": 999, "page": 9, "excerpt": "nope"}],
           "valid": True}
    good = {"summary": "Good revised summary.", "citations": [{"chunk_id": 1, "page": 1, "excerpt": "source text"}],
            "valid": True}
```

Replace with:
```python
    bad = {"summary": "Bad summary.", "citations": [{"chunk_id": 999, "page": "9", "excerpt": "nope"}],
           "valid": True}
    good = {"summary": "Good revised summary.",
            "citations": [{"chunk_id": 1, "page": "1", "excerpt": "source text"}],
            "valid": True}
```

- [ ] **Step 3: Update `examples/run_langgraph_demo.py`'s `load_and_chunk` call**

Find:
```python
def main(pdf_path: str, query: str):
    chunks = load_and_chunk(pdf_path)
    texts = [c.get('text', '') for c in chunks]
```

Replace with:
```python
def main(pdf_path: str, query: str):
    chunks, _flagged_pages = load_and_chunk(pdf_path)
    texts = [c.get('text', '') for c in chunks]
```

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass (same count as before this task — this task changes no test behavior, only fixture realism)

- [ ] **Step 5: Verify the example script still parses**

Run: `python -m py_compile examples/run_langgraph_demo.py`
Expected: no output (clean). Not run end-to-end — see the "Known pre-existing issue" note above.

- [ ] **Step 6: Lint**

Run: `python -m isort --check-only tests/test_evaluator.py tests/test_orchestrator.py examples/run_langgraph_demo.py && python -m flake8 --max-line-length=120 tests/test_evaluator.py tests/test_orchestrator.py examples/run_langgraph_demo.py`
Expected: no output (clean)

- [ ] **Step 7: Commit**

```bash
git add tests/test_evaluator.py tests/test_orchestrator.py examples/run_langgraph_demo.py
git commit -m "Migrate remaining test fixtures to the start_page/end_page chunk shape"
```

---

## Task 7: Full verification

**Files:** none modified — this task only runs checks.

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q`
Expected: PASS, all tests (51 existing + 5 new from Task 1 + 2 new from Task 2 + 1 new from Task 3 + 1 new from Task 4 = 60 total)

- [ ] **Step 2: Full lint gate**

Run: `python -m isort --check-only src tests app.py && python -m flake8 --max-line-length=120 src tests app.py`
Expected: no output (clean)

- [ ] **Step 3: Re-run the Task 5 scripted verification once more against the final state of the code**

Run the same script from Task 5 Step 5.
Expected: `PHASE 2 VERIFICATION: PASS`

- [ ] **Step 4: Confirm no stale references to the old chunk shape remain**

Run: `grep -rn "src.get('page')" src app.py 2>/dev/null; grep -rn '"source": {"page"' src app.py tests examples 2>/dev/null; grep -rn 'page=[0-9]' tests 2>/dev/null`
Expected: no output for all three. (The first was the old page-lookup fallback in production code; the second was the old nested `source` shape; the third was an unquoted integer `page=` literal in a `Citation`/`CitationCheck` construction. Note this deliberately does *not* grep for a bare `"page": N` — `load_pdf_pages`'s own per-page dicts, e.g. `{"page": 1, "text": "..."}` in `tests/test_ingest.py`, legitimately keep that shape; only a chunk's `source` dict and `Citation`/`CitationCheck.page` changed.)

---

## Self-review

- **Spec coverage:** every section of `docs/superpowers/specs/2026-08-27-cross-page-chunking-design.md` (including the extraction-gap addendum) maps to a task: main chunking rewrite → Task 2; `Citation`/`CitationCheck` schema → Task 3; `_format_passages`/system prompt → Task 4; `app.py` display → Task 5; test migration → Task 6; extraction-gap helpers → Task 1; extraction-gap `app.py` warning → Task 5. ✓
- **Placeholders:** no "TBD"/"handle appropriately" — the one intentionally-deferred item (fixing `examples/run_langgraph_demo.py`'s unrelated `make_orchestrator` breakage) is explicitly called out as out of scope with a reason, not a gap. ✓
- **Type/name consistency:** `format_page_label`, `flagged_low_text_pages`, `_build_document`, `_page_at` are named identically everywhere they're consumed (Task 1 → Tasks 2, 4, 5). `load_and_chunk`'s new `Tuple[List[Dict], List[int]]` return is consistently unpacked as `chunks, flagged_pages` (or `_flagged_pages` where unused) in every caller: `app.py` (Task 5), `examples/run_langgraph_demo.py` (Task 6), `tests/test_ingest.py` (Task 2). ✓
