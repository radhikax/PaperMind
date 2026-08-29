# Cross-Page Chunking — Design Spec

**Status:** Approved design, ready for implementation planning.

**Origin:** Phase 2 of `docs/superpowers/plans/2026-08-23-improvement-roadmap.md`. That roadmap flagged this phase as needing its own `superpowers:brainstorming` pass before planning, because of an open product decision on how citations should read for a chunk that spans two pages. This document is that pass.

## Problem

`chunk_pages_to_chunks()` (`src/ingest.py:32-85`) loops `for page in pages` and force-finalizes the in-progress chunk (`cur`) at the end of every page, regardless of where the chunk-size/overlap logic would otherwise cut. A sentence — or an argument spanning several sentences — that straddles a page break is always split at the page boundary, so the tail of page *N* and the head of page *N+1* never appear together in one chunk even when they are one continuous thought. This can cause retrieval to surface only half of a relevant passage.

## Investigation findings (why the scope is smaller than the roadmap assumed)

- `CitationVerifierAgent.verify()` (`src/agents.py:176-202`) matches citations against retrieved chunks using only `chunk_id` and a text-excerpt substring check. It never reads `page`. The roadmap's "risk to watch" — that the verifier assumes one page per chunk — does not apply; no verification logic changes as part of this work.
- `page` is purely display/prompt metadata: it appears in the `[chunk_id:X page:Y]` header shown to the LLM (`src/agents.py:_format_passages`), in the LLM's own echoed citation (`Citation.page`, `src/schemas.py`), and in the Streamlit sidebar chunk display (`app.py:149`).
- `start_char`/`end_char` on a chunk's `source` dict are not consumed anywhere outside `src/ingest.py` itself (confirmed via repo-wide grep) — free to redefine without breaking a consumer.
- `src/vectorstore.py` (`FaissStore`) is agnostic to chunk shape — no changes needed there.

## Product decision: citation display for a spanning chunk

**Chosen: page range.** A chunk that spans pages 3 and 4 is cited as "page 3-4" (or "p. 3-4" in prose); a chunk within a single page is cited as before, e.g. "page 3". This was chosen over (a) picking the page contributing the most characters — misleading when the split is close to even — and (b) avoiding spanning chunks altogether via sentence-level carry-forward — meaningfully more algorithmic complexity for a smaller behavior change, and still an indirect way of avoiding the same information.

## Approach: concatenate + offset→page lookup table

Two approaches were considered:

- **Chosen — concatenate all page texts into one document string, plus a sorted `(char_offset, page_num)` table.** Run the existing sentence-accumulation/chunk-size/overlap loop once over the whole document instead of once per page (no per-page reset). For each finished chunk, look up its start/end character offsets in the table (binary search) to get `start_page`/`end_page`. This handles the existing overlap-tail edge case correctly by construction: if the carried-over overlap text itself crosses a page boundary, the offset lookup gets the right page because it's derived from real character positions, not inferred.
- **Rejected — carry the accumulator across the page loop without a lookup table.** Track `start_page`/`end_page` inline as the loop crosses page boundaries. Smaller diff at first glance, but the overlap-tail case (the last `overlap` characters carried into the next chunk) can itself span a page boundary, and without an offset table there's no clean way to attribute those characters to a page — this approach ends up reinventing a piece of the chosen approach's lookup table just for that one case, so it doesn't actually save complexity.

## Components and file changes

### `src/ingest.py`

- New internal helper `_build_document(pages: List[Dict]) -> Tuple[str, List[Tuple[int, int]]]`: concatenates each page's text (joined by `"\n"`) into one document string; returns the string plus a list of `(start_offset, page_num)` pairs, one per non-empty page, sorted by offset (they already are, since pages are processed in order).
- New internal helper `_page_at(offset: int, page_offsets: List[Tuple[int, int]]) -> int`: binary-search (`bisect`) for the page owning a given character offset in the concatenated document.
- `chunk_pages_to_chunks` rewritten: build the document + offset table via `_build_document`; run `split_sentences` over the whole document text (unchanged function, just called once instead of per page); run the same accumulate-until-`chunk_size`/overlap loop as today, but as a single pass with no page-boundary reset. Track each chunk's start/end offset within the document via running position (cumulative length), not the current `text.find(cur)` approach — `.find()` on the full document risks matching an earlier, unrelated occurrence of repeated text now that the search space spans the whole document rather than one page.
- `source` dict shape changes from `{"page": int, "start_char": int, "end_char": int}` to `{"start_page": int, "end_page": int, "start_char": int, "end_char": int}`. `start_char`/`end_char` now index into the concatenated document rather than a single page's text (still consumed by nothing outside `ingest.py`; kept because they're cheap to produce and already documented in the function's docstring).
- Update the function's docstring to describe the new `source` shape and the whole-document approach.

### `src/schemas.py`

- `Citation.page: Optional[int]` → `Optional[str]`. The LLM echoes back whatever page label it was shown in the passage header (`"3"` or `"3-4"`); a free-form string accommodates both without a schema-level range type.
- `CitationCheck.page: Optional[int]` → `Optional[str]`, matching `Citation.page` since `CitationVerifierAgent` just passes this value through without parsing it.

### `src/agents.py`

- `_format_passages`: replace `src.get('page') or c.get('page')` with reading `start_page`/`end_page` from `source`, rendering `page:{start_page}` when they're equal and `page:{start_page}-{end_page}` when they differ, in the `[chunk_id:X page:Y]` header.
- Update the one-line system-prompt description of the header format (`SummarizerAgent.summarize`, currently "Each passage is prefixed with metadata like `[chunk_id:X page:Y]`") to note that `Y` may be a range like `3-4` when a passage spans two pages.
- `SummarizerAgent.summarize`'s own citation-string formatting (`f"[p.{c.page}#id:{c.chunk_id}]"`) needs no structural change — `c.page` is now a string but is only ever interpolated, not compared or arithmetic'd on.

### `app.py`

- Line 149 (`st.write(f"page: {src.get('page', 'N/A')}")`) becomes range-aware using the same `start_page`/`end_page` → single-or-range rendering as `_format_passages`, to avoid duplicating the format logic, factor it into a small shared helper (e.g. `src/ingest.py` or `src/agents.py` — implementation plan decides placement) rather than writing it twice.

## Test changes

- `tests/test_agents.py`, `tests/test_evaluator.py`, `tests/test_orchestrator.py`, `tests/test_schemas.py` currently construct fixtures with the old singular shape (`{"source": {"page": N}}`, `Citation(page=N)` with an int). These are updated in place to the new `start_page`/`end_page` shape and string `page` values — a mechanical migration, not a design change, and no backward-compatibility shim is introduced since every call site is inside this repo and under this change's control.
- `tests/test_ingest.py`: existing `test_load_and_chunk` continues to pass unchanged (its assertions are loose — only checks `"text"` and `"source"` keys exist). New tests:
  - A two-page fixture PDF constructed so one sentence's text is split across the page boundary (page 1 ends mid-sentence, page 2 continues it) — asserts that sentence lands inside a single chunk, and that chunk's `start_page != end_page`.
  - A same-page case — asserts `start_page == end_page` for a chunk that doesn't cross a boundary.
  - Optionally, a direct unit test of `_page_at` against a small hand-built offset table, if it's exposed for testing (implementation plan decides whether to test it directly or only through `chunk_pages_to_chunks`).

## Error handling

No new failure modes are introduced. `_page_at` always has at least one table entry for any document built from non-empty pages (empty pages are already skipped, matching current behavior), and every chunk's start/end offsets are within `[0, len(document))` by construction of the single-pass loop, so lookups never fall outside the table's range.

## Out of scope

- Any change to `CitationVerifierAgent`'s matching logic (confirmed unnecessary — see Investigation findings).
- Fixing the pre-existing fragility of `text.find(cur)`-style offset computation beyond what's needed to make it correct at document scale (the rewrite already requires replacing this with running-position tracking, so it's fixed as a byproduct, not a separate goal).
- Any change to how retrieval ranks or selects chunks — this phase only changes chunk boundaries and their page attribution.

## Addendum: extraction-gap detection

**Origin:** raised alongside the cross-page chunking question — while cross-page chunking fixes information lost at a chunk *boundary*, it does nothing for information lost because a page's text was never extracted in the first place. This addendum is a bounded design (per `superpowers:brainstorming`'s bounded path — a small, contained addition, not requiring its own spec document), folded into this same implementation pass because it touches the same ingestion entry point.

**Problem:** `load_pdf_pages()` (`src/ingest.py:7-17`) does `page.extract_text() or ""`. When `pdfplumber` can't extract text from a page — a scanned/image-only page, a page that's mostly a table or figure — that page silently contributes an empty string. `chunk_pages_to_chunks` skips pages with empty text (`if not text: continue`) with no warning anywhere in the pipeline. A page with a little text (e.g. a stray caption on an otherwise-scanned page) isn't empty, so today it passes through unflagged even though the real content is lost.

**Design:**
- `src/ingest.py`: add `MIN_PAGE_TEXT_CHARS = 20` and `flagged_low_text_pages(pages: List[Dict], min_chars: int = MIN_PAGE_TEXT_CHARS) -> List[int]`, returning the page numbers whose extracted text is empty or under `min_chars`. `load_and_chunk()` calls this once against the pages it already loads and returns `(chunks, flagged_pages)` instead of just `chunks`.
- `app.py`: unpack the new tuple; when `flagged_pages` is non-empty, show `st.warning(...)` naming the affected page numbers and explaining they may be scanned images or a layout `pdfplumber` can't parse, so content from them may be missing from answers.
- `examples/run_langgraph_demo.py` and `tests/test_ingest.py`: both call `load_and_chunk()` and need updating to unpack the new two-value return — mechanical, no behavior change for them beyond the unpacking.

**Detection threshold:** empty-or-under-20-characters, not strict-empty-only. Catches near-empty pages (e.g. a scanned page with just a stray caption) at the cost of occasionally flagging a genuinely short real page (e.g. a title page) — accepted tradeoff, chosen over strict-empty-only which would miss the near-empty case entirely.

**Out of scope:** detecting scrambled reading order on multi-column layouts (text present but interleaved out of order). `pdfplumber` gives no reliable per-page confidence signal for this short of comparing against word-level bounding boxes, which is a materially harder problem than "did this page produce text at all" — not attempted here.

**Testing:** a new `tests/test_ingest.py` case with one normal-text page and one blank page, asserting the blank page's number comes back in `flagged_pages`; the existing `test_load_and_chunk` updated to unpack the tuple.

**Independence from the main design:** this only inspects `pages` before chunking starts, so it doesn't interact with the whole-document-concatenation rewrite above — both land in the same implementation pass without conflicting.
