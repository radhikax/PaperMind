# Golden-Question Eval Harness — Design Spec

**Status:** Approved design, ready for implementation planning.

**Origin:** Phase 3 of `docs/superpowers/plans/2026-08-23-improvement-roadmap.md`. The roadmap flagged this phase as needing its own `superpowers:brainstorming` pass because of an open question — whether the harness may make real LLM calls — before writing a task-by-task plan. This document is that pass.

## Problem

There is no way to know whether a change to retrieval, prompts, or the evaluator's weights made PaperMind's answers better or worse, other than manually re-running the Streamlit app and eyeballing the result. The existing 60 tests in `tests/` only verify components are wired together correctly using scripted/fake LLM responses — they would all still pass even if a prompt change made every real answer worse, because none of them call a real LLM or check answer quality against known-correct facts.

This matters specifically now because Phase 4 (evaluator weights fit from real feedback data) explicitly depends on it: before trusting a re-fit weight vector, there needs to be a way to confirm it doesn't quietly regress retrieval or scoring on questions with a known-correct answer.

## Investigation findings

- **CI has no `OPENAI_API_KEY` secret.** `.github/workflows/ci.yml`'s test step runs `PYTHONPATH=. pytest -q` with no environment variables set. Any pytest-discovered test that expected real LLM behavior would silently run in `degraded_mode` (the existing heuristic fallback in `SummarizerAgent`/`CriticAgent`) in CI, producing misleading pass/fail results — the harness needs to be built around this constraint, not against it.
- **`examples/sample.pdf` cannot support golden questions.** It is a single page of near-identical placeholder lines ("This is sample paragraph line 1/2/3/4... It contains some example content for testing PDF ingestion.") with no distinguishable facts across pages. A golden set needs a fixture paper written specifically to have distinct, checkable claims per page.

## Decision: real LLM calls via a standalone script, not a pytest-gated test

Three approaches were considered:

- **Chosen — a standalone script (`scripts/run_eval.py`), never pytest-discovered.** It calls `make_orchestrator(...)` for real. If the person running it has `OPENAI_API_KEY` set, they get genuine LLM-backed results; if not, it gracefully falls back to the existing `degraded_mode` heuristic path — the exact same fallback behavior the rest of the app already relies on, so no new "fake" mode needs to be invented. Because it's never collected by pytest, it never runs in CI and needs no CI configuration changes, no `@pytest.mark` exclusion, and no CI secret.
- **Rejected — a pytest test file marked `@pytest.mark.eval`, excluded from CI via `-m "not eval"`.** This requires editing `.github/workflows/ci.yml` to add the exclusion flag, and if anyone ever *does* want to run it in CI on demand, that requires provisioning a real `OPENAI_API_KEY` secret — both are process changes beyond this phase's scope, for no benefit over the standalone-script approach.
- **Rejected — fakes only, fully deterministic.** No real LLM calls ever, scripted responses like the existing unit tests. Fast and free, but defeats a chunk of the point of this phase: it wouldn't catch a real prompt-quality regression, which is exactly the kind of change Phase 4 needs to validate against.

## Decision: a new synthesized fixture paper

`examples/sample.pdf` is unusable (see Investigation findings). A new 5-page paper is generated at run-time via `reportlab` — matching how existing test fixtures already build PDFs in `tests/test_ingest.py` (`make_sample_pdf`) — rather than committing a binary PDF to the repo. Regenerating it fresh on every run is cheap (a handful of `drawString` calls) and avoids any risk of a stale, out-of-sync fixture file.

### Fixture paper content (5 pages, one topic per page)

1. **Abstract:** "Widget Classification with Neural Networks. Abstract. We present a neural network approach for classifying industrial widgets into seven categories. Our model, WidgetNet, achieves 94.2% accuracy on the benchmark WidgetBench dataset, outperforming the previous best result by 8 percentage points. This report describes the dataset, the model architecture, and our experimental results."
2. **Dataset:** "Dataset. The WidgetBench dataset contains 12,400 labeled widget images collected from 6 manufacturing plants between 2024 and 2025. Each image is labeled with one of seven widget categories: bolt, gear, spring, bracket, washer, valve, and hinge. The dataset is split into 9,000 training images, 1,700 validation images, and 1,700 test images."
3. **Method:** "Method. WidgetNet uses a convolutional neural network with 18 layers, trained using the Adam optimizer with a learning rate of 0.0003 for 60 epochs. We apply standard data augmentation including random rotation and color jitter. Training took approximately 4 hours on a single GPU."
4. **Results:** "Results. WidgetNet achieves 94.2% top-1 accuracy on the WidgetBench test set. The strongest baseline, a ResNet-34 classifier, achieves 86.1% accuracy under the same training conditions. WidgetNet's largest error category is confusing washers with brackets, accounting for 41% of all misclassifications."
5. **Limitations and Future Work:** "Limitations and Future Work. WidgetNet was evaluated only on images captured under controlled lighting conditions; performance under variable lighting is untested. Future work will explore domain adaptation techniques to improve robustness to lighting changes, and will extend the widget taxonomy beyond the current seven categories."

All facts are invented for this fixture — no real dataset, model, or benchmark named "WidgetNet"/"WidgetBench" exists, avoiding any factual-accuracy concern about the fixture itself.

## Golden question set

**File:** `scripts/fixtures/golden_qa.jsonl`, one JSON object per line: `{"question": str, "expected_pages": List[int], "adversarial": bool}`.

- `expected_pages`: page numbers where the answer is supported. A question passes its retrieval check if **at least one** listed page appears among the retrieved chunks' page ranges (not all — a question mentioned on two pages only needs one to be found for retrieval to be considered correct, since `top_k` may reasonably surface just one).
- `adversarial`: `true` for the one question deliberately asking about something the paper never mentions. Its `expected_pages` is `[]`. It is never scored pass/fail — see harness design below.

Twelve questions: eleven grounded (one or two per fixture-paper fact), one adversarial.

1. What accuracy does WidgetNet achieve on the WidgetBench test set? — pages [1, 4]
2. How many labeled widget images are in the WidgetBench dataset? — pages [2]
3. How many manufacturing plants contributed to the WidgetBench dataset? — pages [2]
4. What are the seven widget categories used in this study? — pages [2]
5. How many training, validation, and test images are in the dataset split? — pages [2]
6. What optimizer and learning rate were used to train WidgetNet? — pages [3]
7. How many epochs was WidgetNet trained for? — pages [3]
8. How long did training take? — pages [3]
9. What accuracy did the ResNet-34 baseline achieve? — pages [4]
10. What is WidgetNet's most common error, and what fraction of misclassifications does it account for? — pages [4]
11. Under what lighting conditions was WidgetNet evaluated, and what future work is planned? — pages [5]
12. **(adversarial)** What programming language and deep learning framework was WidgetNet implemented in? — pages [] — this fact is never stated anywhere in the fixture paper.

## Harness design (`scripts/run_eval.py`)

- Generates the fixture PDF fresh (via a `build_golden_paper(path)` helper in `scripts/fixtures/golden_paper.py`), loads `scripts/fixtures/golden_qa.jsonl`.
- Chunks, embeds, and builds an **in-memory** `FaissStore` — this eval index is not registered through Phase 1's `paper_registry` (`register_paper`/`index_path_for`), since it is a throwaway evaluation artifact, not a real user paper, and registering it would pollute the "Previously indexed papers" list a real user sees in the sidebar.
- For each question, runs `make_orchestrator(store, embedder).invoke(build_initial_state(question))` and captures the final state.
- **Grounded questions** (`adversarial: false`) are checked against two objective, LLM-nondeterminism-independent criteria:
  1. **Retrieval:** did any retrieved chunk's `[start_page, end_page]` range include at least one of `expected_pages`?
  2. **Citation verification:** did `citation_verification.verified_ratio` clear a floor — `MIN_VERIFIED_RATIO = 0.5`, a single script-level constant, not per-question (all grounded questions in this fixture are equally straightforward, so a per-question threshold isn't justified). 0.5 was chosen to match the same floor `ReliabilityEvaluator._build_critique` already uses to flag citation problems (`src/evaluator.py`), keeping the harness's bar consistent with the pipeline's own internal one rather than inventing a separate number.

  The reliability `decision` (accept/revise/exhausted) is reported for visibility but is not itself a pass/fail criterion — with a real LLM in the loop, a "revise" that still lands on a correct, well-cited answer is not a harness failure.
- **The adversarial question** is run through the same pipeline but only ever *reported* (question, retrieved pages, verified_ratio, score, decision) — never scored pass/fail. There's no principled threshold for "how low is low enough" without real calibration data (that's Phase 4's job), so asserting one here would just be a flaky, made-up number. A human reading the report can judge whether the system appropriately hedged (low score, low verified_ratio) or confidently fabricated an answer.
- **Report:** printed per-question (question text, PASS/FAIL on the two objective checks for grounded questions, or an "(adversarial, informational)" label; score; decision), plus a summary line (`N/11 grounded questions passed both checks`). Exits non-zero if any grounded question fails either objective check; the adversarial question never affects the exit code.
- Whether a real LLM or the heuristic fallback answered is visible in the report (via `degraded_mode` in the final state) so a person reading the output knows which kind of signal they got.

## Error handling

No new failure modes beyond what `make_orchestrator`'s pipeline already handles (its own retry/fallback logic is unchanged by this phase). If `OPENAI_API_KEY` is absent, every question runs in `degraded_mode` and the report should make that visible per-question rather than silently proceeding as if it were a real-LLM run — the person running the script needs to know a degraded-mode pass doesn't confirm prompt-quality, only pipeline plumbing.

## Out of scope

- Any change to `.github/workflows/ci.yml` — the standalone-script approach was chosen specifically to avoid this.
- Per-question confidence thresholds — one global floor is used; if a future phase's fixture set needs graded difficulty, that's a separate design.
- A CI-integrated or scheduled run of this script — it is a manually-invoked developer tool for this phase, not a gate.
- Any change to `ReliabilityEvaluator`'s weights or the calibrator — that is Phase 4, which this harness exists to validate.
