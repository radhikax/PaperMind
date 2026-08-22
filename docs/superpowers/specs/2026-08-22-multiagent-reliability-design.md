# Multi-Agent Reliability Orchestration — Design Spec

Date: 2026-08-22
Status: Approved, moving to implementation

## Problem

The repo's stated goal is a LangGraph-based multi-agent research assistant
that both summarizes papers and evaluates the reliability of its own
summaries. In practice, `src/langgraph_agents.py`'s `make_orchestrator`
never builds a real graph: it probes for `langgraph.Graph`/`Node` or
`langgraph.build_graph`, neither of which is a real LangGraph API (the real
API is `StateGraph`), and `langgraph` is not even installed. It silently
falls back to returning plain callables, which `app.py` then chains
manually and imperatively. There is no shared state object, no branching,
and no revision loop: the "critic" and the numeric "evaluator" score are
computed but never gate or improve the output. Citation checking is a
regex heuristic that doesn't verify the cited text exists in the source.

## Scope

**In scope:** rebuild the orchestration and agent-responsibility layer —
`src/langgraph_agents.py`, `src/agents.py`, `src/evaluator.py` — as a real
LangGraph `StateGraph` with a genuine reliability-gated revision loop, plus
a new citation-verification agent, plus modernizing the OpenAI SDK usage.
Update `app.py` to drive the compiled graph instead of chaining functions
by hand, and update/extend tests accordingly.

**Out of scope (kept as-is):** `src/ingest.py`, `src/embeddings.py`,
`src/vectorstore.py`, `src/schemas.py` (extended, not replaced), the
Streamlit UI shell and feedback-dashboard tab, `feedback.jsonl` persistence
format.

## Architecture

Five single-purpose agent nodes wired into a LangGraph `StateGraph`:

```
Retriever → Summarizer ──┬──→ Critic ─────────┐
                          └──→ CitationVerifier ┴──→ ReliabilityEvaluator ─┬─(score ≥ threshold)──────────→ END (accept)
                                                                            ├─(score < threshold, retries left)→ back to Summarizer (with critique_feedback)
                                                                            └─(retries exhausted)──────────→ END (best-effort + low-confidence warning)
```

- Retriever runs once per query, outside the revision loop.
- Summarizer, Critic, CitationVerifier, and ReliabilityEvaluator can all
  re-run on a revision cycle.
- Critic and CitationVerifier depend only on the current summary +
  retrieved chunks, not on each other — they run as parallel branches that
  join at ReliabilityEvaluator.
- ReliabilityEvaluator's revise decision writes a concrete
  `critique_feedback` string (not just a score) that is injected into the
  next Summarizer prompt, so a retry is a targeted revision, not a blind
  resend.

## Shared state

```python
class GraphState(TypedDict):
    query: str
    paper_id: str
    retrieved_chunks: list[dict]          # set once by Retriever
    summary: SummaryResponse | None       # current attempt
    critic_assessment: dict | None        # confidence, hallucination_rate, notes
    citation_verification: dict | None    # per-citation match results + verified_ratio
    reliability_score: int | None         # 0-100
    critique_feedback: str | None         # evaluator's notes fed back to Summarizer on revise
    attempt: int
    max_attempts: int
    history: list[dict]                   # every attempt's summary+score, for best-effort fallback
    degraded_mode: bool                   # true if OpenAI unavailable, heuristics used instead
```

## Components

- **RetrieverAgent** — unchanged retrieval logic, becomes the Retriever node.
- **SummarizerAgent** — accepts optional `critique_feedback` and the prior
  `summary` to produce a targeted revision rather than resummarizing from
  scratch each attempt. Uses the OpenAI v1 client with structured outputs
  (Pydantic schema via `SummaryResponse`) instead of manual `json.loads` +
  retry-on-parse-failure.
- **CriticAgent** — same hallucination/confidence judgment responsibility,
  modernized SDK call.
- **CitationVerifierAgent** (new) — for each claimed citation, checks the
  `chunk_id`/`page` exists among `retrieved_chunks` and that the cited
  excerpt text actually overlaps the source chunk text (substring/fuzzy
  match), producing a `verified_ratio` in `citation_verification`.
- **ReliabilityEvaluator** (rebuilt from `evaluator.py`) — aggregates
  semantic similarity + verifier overlap + critic confidence +
  `citation_verified_ratio` into the existing 0–100 weighted score, decides
  accept / revise / exhausted, and on revise writes `critique_feedback`
  naming the specific deficiency (e.g. "citation on page 3 not found in
  source chunk; hallucination_rate 0.4 exceeds threshold").

## Error handling & degraded mode

- LLM call failures (network/rate-limit) are retried with backoff at the
  node level — the existing `call_with_retries_validate` logic moves out of
  `app.py` into the orchestration layer so it applies uniformly to every
  node and is testable without Streamlit.
- If OpenAI is unavailable (missing key/package), each agent falls back to
  its existing heuristic (length-based confidence, regex citation check),
  and the node sets `degraded_mode: True` on state so the UI can visibly
  flag that scoring is heuristic-based, not model-judged.
- Malformed structured output: OpenAI v1 structured outputs should mostly
  prevent this; one repair retry remains as a safety net.
- Retries exhausted: return the best-scoring attempt from `history` with a
  visible low-confidence warning, rather than failing closed.

## Testing plan

- Unit-test each node with a fake/injected LLM client — no real API calls,
  mirroring the existing `DummyStore`/`DummyEmbedder` pattern in
  `test_orchestrator.py`.
- Test graph routing explicitly: low score → assert loop-back occurs and
  `critique_feedback` is non-empty; high score → assert immediate accept;
  repeated low scores → assert exhaustion path returns the best `history`
  entry with `degraded_mode`/low-confidence flag set.
- Keep and extend `test_ingest.py` / `test_evaluator.py`; add
  `test_citation_verifier.py`; replace the thin `test_orchestrator.py` with
  a golden end-to-end `graph.invoke()` test using fakes.

## Decisions log (from brainstorming)

- Rebuild orchestration/agent layer in place; keep ingest/embeddings/
  vectorstore/UI/feedback loop as-is.
- Reliability = gatekeeping revision loop, not just transparent scoring or
  multi-agent debate voting.
- Real LangGraph `StateGraph` dependency, not a hand-rolled state machine.
- On exhausted retries: show best-effort with a low-confidence warning
  (fail open, not closed).
- Modernize OpenAI SDK usage as part of this rebuild.
- Add real citation verification against source text (not just regex).
- Modular one-job-per-node graph structure (Critic and CitationVerifier as
  parallel branches), not a single aggregator "god node".
