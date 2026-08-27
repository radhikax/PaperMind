# Golden-Question Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A manually-run script that checks whether PaperMind's retrieval and citation verification behave correctly against a small set of known-answer questions, so future changes to retrieval/prompts/evaluator weights can be checked for regressions before they ship.

**Architecture:** A synthetic 5-page fixture paper (generated at run-time, not committed as a binary) plus a 12-question golden set drive the real `make_orchestrator` pipeline end to end. Two objective checks per grounded question — did retrieval surface an expected page, did citation verification clear a floor — are pure functions, unit-tested normally. The orchestration that calls the real pipeline lives only in the standalone script (`scripts/run_eval.py`), which is never collected by pytest and therefore never runs in CI.

**Tech Stack:** Python 3.10, `reportlab` (already a dependency, used to generate the fixture PDF), `pdfplumber`, existing `src.*` modules — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-golden-eval-harness-design.md`.

## Global Constraints

- Don't break the existing 60 passing tests or the CI lint gate (`isort --check-only src tests app.py`, `flake8 --max-line-length=120 src tests app.py`). `scripts/` is outside that gate, but every task still lints its own new files with the same flags for consistency.
- No new hard dependencies — `reportlab` and `pdfplumber` are already in `requirements.txt`.
- No changes to `.github/workflows/ci.yml` — the standalone-script design exists specifically to avoid needing this (see spec's "Decision: real LLM calls via a standalone script" section).
- `scripts/run_eval.py` must be run with `PYTHONPATH=.` (matching how CI already invokes pytest: `PYTHONPATH=. pytest -q`) — verified by hand that `python scripts/run_eval.py` directly fails with `ModuleNotFoundError: No module named 'src'`, the same pre-existing behavior `examples/run_langgraph_demo.py` already has. pytest itself resolves `from src...`/`from scripts...` imports without needing `PYTHONPATH` set explicitly (verified: the existing test suite already does this).

---

## Task 1: Fixture paper generator

**Files:**
- Create: `scripts/fixtures/golden_paper.py`
- Test: `tests/test_golden_paper.py`

**Interfaces:**
- Produces: `PAGES: List[str]` (5 entries); `build_golden_paper(path: str) -> None`.

No `scripts/__init__.py` or `scripts/fixtures/__init__.py` is needed — Python 3's implicit namespace packages (verified by hand) make `from scripts.fixtures.golden_paper import ...` resolve without one, both under pytest and under `PYTHONPATH=. python scripts/run_eval.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_golden_paper.py
import tempfile

import pdfplumber

from scripts.fixtures.golden_paper import PAGES, build_golden_paper


def test_pages_constant_has_five_entries():
    assert len(PAGES) == 5


def test_build_golden_paper_produces_five_pages():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    build_golden_paper(tmp.name)
    with pdfplumber.open(tmp.name) as pdf:
        assert len(pdf.pages) == 5


def test_build_golden_paper_page_content_matches_known_facts():
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    build_golden_paper(tmp.name)
    with pdfplumber.open(tmp.name) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]
    assert "94.2%" in texts[0]
    assert "12,400" in texts[1]
    assert "0.0003" in texts[2]
    assert "86.1%" in texts[3]
    assert "controlled lighting" in texts[4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_golden_paper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/fixtures/golden_paper.py
"""Generates the synthetic fixture paper used by scripts/run_eval.py.

All facts here are invented for this fixture -- WidgetNet/WidgetBench are
not real -- see docs/superpowers/specs/2026-08-27-golden-eval-harness-design.md.
"""
from reportlab.pdfgen import canvas

PAGES = [
    "Widget Classification with Neural Networks. Abstract. We present a neural "
    "network approach for classifying industrial widgets into seven categories. "
    "Our model, WidgetNet, achieves 94.2% accuracy on the benchmark WidgetBench "
    "dataset, outperforming the previous best result by 8 percentage points. "
    "This report describes the dataset, the model architecture, and our "
    "experimental results.",

    "Dataset. The WidgetBench dataset contains 12,400 labeled widget images "
    "collected from 6 manufacturing plants between 2024 and 2025. Each image "
    "is labeled with one of seven widget categories: bolt, gear, spring, "
    "bracket, washer, valve, and hinge. The dataset is split into 9,000 "
    "training images, 1,700 validation images, and 1,700 test images.",

    "Method. WidgetNet uses a convolutional neural network with 18 layers, "
    "trained using the Adam optimizer with a learning rate of 0.0003 for 60 "
    "epochs. We apply standard data augmentation including random rotation "
    "and color jitter. Training took approximately 4 hours on a single GPU.",

    "Results. WidgetNet achieves 94.2% top-1 accuracy on the WidgetBench test "
    "set. The strongest baseline, a ResNet-34 classifier, achieves 86.1% "
    "accuracy under the same training conditions. WidgetNet's largest error "
    "category is confusing washers with brackets, accounting for 41% of all "
    "misclassifications.",

    "Limitations and Future Work. WidgetNet was evaluated only on images "
    "captured under controlled lighting conditions; performance under "
    "variable lighting is untested. Future work will explore domain "
    "adaptation techniques to improve robustness to lighting changes, and "
    "will extend the widget taxonomy beyond the current seven categories.",
]


def build_golden_paper(path: str) -> None:
    """Write the 5-page synthetic golden-eval fixture PDF to `path`."""
    c = canvas.Canvas(path)
    for page_text in PAGES:
        c.drawString(72, 720, page_text)
        c.showPage()
    c.save()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_golden_paper.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint**

Run: `python -m isort --check-only scripts/fixtures/golden_paper.py tests/test_golden_paper.py && python -m flake8 --max-line-length=120 scripts/fixtures/golden_paper.py tests/test_golden_paper.py`
Expected: no output (clean). If isort complains, run `python -m isort scripts/fixtures/golden_paper.py tests/test_golden_paper.py` and re-check.

- [ ] **Step 6: Commit**

```bash
git add scripts/fixtures/golden_paper.py tests/test_golden_paper.py
git commit -m "Add the synthetic fixture paper for the golden-eval harness"
```

---

## Task 2: Golden question set and loader

**Files:**
- Create: `scripts/fixtures/golden_qa.jsonl`
- Create: `scripts/run_eval.py`
- Test: `tests/test_run_eval.py`

**Interfaces:**
- Produces: `GOLDEN_QA_PATH: str`; `load_golden_qa(path: str) -> List[Dict]`.

- [ ] **Step 1: Write the golden question set**

```jsonl
{"question": "What accuracy does WidgetNet achieve on the WidgetBench test set?", "expected_pages": [1, 4], "adversarial": false}
{"question": "How many labeled widget images are in the WidgetBench dataset?", "expected_pages": [2], "adversarial": false}
{"question": "How many manufacturing plants contributed to the WidgetBench dataset?", "expected_pages": [2], "adversarial": false}
{"question": "What are the seven widget categories used in this study?", "expected_pages": [2], "adversarial": false}
{"question": "How many training, validation, and test images are in the dataset split?", "expected_pages": [2], "adversarial": false}
{"question": "What optimizer and learning rate were used to train WidgetNet?", "expected_pages": [3], "adversarial": false}
{"question": "How many epochs was WidgetNet trained for?", "expected_pages": [3], "adversarial": false}
{"question": "How long did training take?", "expected_pages": [3], "adversarial": false}
{"question": "What accuracy did the ResNet-34 baseline achieve?", "expected_pages": [4], "adversarial": false}
{"question": "What is WidgetNet's most common error, and what fraction of misclassifications does it account for?", "expected_pages": [4], "adversarial": false}
{"question": "Under what lighting conditions was WidgetNet evaluated, and what future work is planned?", "expected_pages": [5], "adversarial": false}
{"question": "What programming language and deep learning framework was WidgetNet implemented in?", "expected_pages": [], "adversarial": true}
```

Save this as `scripts/fixtures/golden_qa.jsonl` (one JSON object per line, no trailing commas, no surrounding array brackets).

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_run_eval.py
from scripts.run_eval import GOLDEN_QA_PATH, load_golden_qa


def test_load_golden_qa_returns_all_twelve_questions():
    questions = load_golden_qa(GOLDEN_QA_PATH)
    assert len(questions) == 12


def test_load_golden_qa_entries_have_required_keys():
    questions = load_golden_qa(GOLDEN_QA_PATH)
    for q in questions:
        assert "question" in q
        assert "expected_pages" in q
        assert "adversarial" in q


def test_load_golden_qa_has_exactly_one_adversarial_question():
    questions = load_golden_qa(GOLDEN_QA_PATH)
    adversarial = [q for q in questions if q["adversarial"]]
    assert len(adversarial) == 1
    assert adversarial[0]["expected_pages"] == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_eval'`

- [ ] **Step 4: Write the implementation**

```python
# scripts/run_eval.py
"""Golden-question eval harness for PaperMind.

Runs a small set of known-answer questions against a synthetic fixture
paper through the real reliability-gated pipeline, and reports whether
retrieval and citation verification behaved as expected. See
docs/superpowers/specs/2026-08-27-golden-eval-harness-design.md.

Usage:
    PYTHONPATH=. python scripts/run_eval.py

Uses a real OpenAI call if OPENAI_API_KEY is set, otherwise falls back to
the same degraded_mode heuristic path the rest of the app already uses --
in that case citation verification will fail for every grounded question
(no LLM means no citations to verify), which is expected, not a bug; only
the retrieval check is meaningful without a real key.
"""
import json
import os
from typing import Dict, List

GOLDEN_QA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_qa.jsonl")


def load_golden_qa(path: str) -> List[Dict]:
    """Load golden questions from a JSONL file: one {"question", "expected_pages", "adversarial"} per line."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_eval.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Lint**

Run: `python -m isort --check-only scripts/run_eval.py tests/test_run_eval.py && python -m flake8 --max-line-length=120 scripts/run_eval.py tests/test_run_eval.py`
Expected: no output (clean)

- [ ] **Step 7: Commit**

```bash
git add scripts/fixtures/golden_qa.jsonl scripts/run_eval.py tests/test_run_eval.py
git commit -m "Add the golden question set and its loader"
```

---

## Task 3: Retrieval and verification check helpers

**Files:**
- Modify: `scripts/run_eval.py`
- Test: `tests/test_run_eval.py`

**Interfaces:**
- Produces: `MIN_VERIFIED_RATIO: float` (0.5); `chunk_covers_expected_page(source: dict, expected_pages: List[int]) -> bool`; `passes_verification_floor(verified_ratio: float) -> bool`.

These are the two objective, pure-function checks the harness applies per grounded question — kept separate from the orchestration in Task 4 so they're normally unit-testable without needing a real (or even degraded-mode) pipeline run.

- [ ] **Step 1: Write the failing tests**

First, update the import line at the top of `tests/test_run_eval.py`.

Find:
```python
from scripts.run_eval import GOLDEN_QA_PATH, load_golden_qa
```

Replace with:
```python
from scripts.run_eval import (GOLDEN_QA_PATH, MIN_VERIFIED_RATIO,
                               chunk_covers_expected_page, load_golden_qa,
                               passes_verification_floor)
```

Then add the following tests to `tests/test_run_eval.py`:

```python
def test_chunk_covers_expected_page_true_when_page_in_range():
    assert chunk_covers_expected_page({"start_page": 2, "end_page": 4}, [3]) is True


def test_chunk_covers_expected_page_false_when_page_outside_range():
    assert chunk_covers_expected_page({"start_page": 2, "end_page": 4}, [5]) is False


def test_chunk_covers_expected_page_false_when_no_expected_pages():
    assert chunk_covers_expected_page({"start_page": 2, "end_page": 4}, []) is False


def test_chunk_covers_expected_page_false_when_source_missing_pages():
    assert chunk_covers_expected_page({}, [1]) is False


def test_passes_verification_floor_at_or_above_threshold():
    assert passes_verification_floor(MIN_VERIFIED_RATIO) is True
    assert passes_verification_floor(1.0) is True


def test_passes_verification_floor_below_threshold():
    assert passes_verification_floor(0.1) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_eval.py -v -k "covers_expected_page or verification_floor"`
Expected: FAIL with `ImportError: cannot import name 'chunk_covers_expected_page' from 'scripts.run_eval'`

- [ ] **Step 3: Add the check helpers**

Find (in `scripts/run_eval.py`):
```python
GOLDEN_QA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_qa.jsonl")


def load_golden_qa(path: str) -> List[Dict]:
    """Load golden questions from a JSONL file: one {"question", "expected_pages", "adversarial"} per line."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions
```

Replace with:
```python
GOLDEN_QA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_qa.jsonl")
MIN_VERIFIED_RATIO = 0.5  # mirrors the literal in ReliabilityEvaluator._build_critique (src/evaluator.py)


def load_golden_qa(path: str) -> List[Dict]:
    """Load golden questions from a JSONL file: one {"question", "expected_pages", "adversarial"} per line."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def chunk_covers_expected_page(source: dict, expected_pages: List[int]) -> bool:
    """True if a chunk's [start_page, end_page] range includes at least one expected page."""
    start_page = source.get("start_page")
    end_page = source.get("end_page")
    if start_page is None or end_page is None:
        return False
    return any(start_page <= page <= end_page for page in expected_pages)


def passes_verification_floor(verified_ratio: float) -> bool:
    return verified_ratio >= MIN_VERIFIED_RATIO
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_eval.py -v`
Expected: PASS (9 tests: the 3 from Task 2 plus 6 new ones)

- [ ] **Step 5: Lint**

Run: `python -m isort --check-only scripts/run_eval.py tests/test_run_eval.py && python -m flake8 --max-line-length=120 scripts/run_eval.py tests/test_run_eval.py`
Expected: no output (clean). If isort reorders the import differently than shown above, that's fine — run `python -m isort scripts/run_eval.py tests/test_run_eval.py` and use its output.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_eval.py tests/test_run_eval.py
git commit -m "Add retrieval and citation-verification check helpers"
```

---

## Task 4: Harness orchestration and report

**Files:**
- Modify: `scripts/run_eval.py`

**Interfaces:**
- Consumes: `build_golden_paper` (Task 1); `GOLDEN_QA_PATH`, `load_golden_qa`, `MIN_VERIFIED_RATIO`, `chunk_covers_expected_page`, `passes_verification_floor` (Tasks 2-3); `EmbeddingModel` (`src/embeddings.py`); `chunk_pages_to_chunks`, `load_pdf_pages` (`src/ingest.py`); `build_initial_state`, `make_orchestrator` (`src/langgraph_agents.py`); `FaissStore` (`src/vectorstore.py`).
- Produces: `main() -> int`; `EVAL_CHUNK_SIZE`, `EVAL_CHUNK_OVERLAP`, `EVAL_TOP_K` constants.

This is the piece that actually calls the real pipeline, so unlike Tasks 1-3 it has no pytest test — its "test" is running it directly, per Step 3 below. `main()` builds an in-memory index over the fixture, runs every golden question through `make_orchestrator`, and prints a pass/fail report.

- [ ] **Step 1: Add the orchestration functions and `main()`**

Find (top of `scripts/run_eval.py`):
```python
import json
import os
from typing import Dict, List

GOLDEN_QA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_qa.jsonl")
MIN_VERIFIED_RATIO = 0.5  # mirrors the literal in ReliabilityEvaluator._build_critique (src/evaluator.py)
```

Replace with:
```python
import json
import os
import sys
import tempfile
from typing import Dict, List

from scripts.fixtures.golden_paper import build_golden_paper
from src.embeddings import EmbeddingModel
from src.ingest import chunk_pages_to_chunks, load_pdf_pages
from src.langgraph_agents import build_initial_state, make_orchestrator
from src.vectorstore import FaissStore

GOLDEN_QA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_qa.jsonl")
MIN_VERIFIED_RATIO = 0.5  # mirrors the literal in ReliabilityEvaluator._build_critique (src/evaluator.py)
EVAL_CHUNK_SIZE = 300
EVAL_CHUNK_OVERLAP = 50
EVAL_TOP_K = 2
```

Find (end of `scripts/run_eval.py`, after `passes_verification_floor`):
```python
def passes_verification_floor(verified_ratio: float) -> bool:
    return verified_ratio >= MIN_VERIFIED_RATIO
```

Replace with:
```python
def passes_verification_floor(verified_ratio: float) -> bool:
    return verified_ratio >= MIN_VERIFIED_RATIO


def _build_eval_store():
    """Build an in-memory FaissStore over the synthetic golden-eval fixture paper.

    chunk_size=300 (vs. the app's default 1000) so this short 5-page fixture
    actually splits into roughly one chunk per page instead of collapsing
    into two giant chunks -- verified by hand, see the design spec.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = tmp.name
    build_golden_paper(pdf_path)
    pages = load_pdf_pages(pdf_path)
    chunks = chunk_pages_to_chunks(pages, chunk_size=EVAL_CHUNK_SIZE, overlap=EVAL_CHUNK_OVERLAP)
    os.remove(pdf_path)

    embedder = EmbeddingModel()
    texts = [c.get("text", "") for c in chunks]
    embs = embedder.embed_texts(texts)
    store = FaissStore(dim=embs.shape[1])
    metadatas = [{"text": c.get("text", ""), "source": c.get("source", {}), "id": i} for i, c in enumerate(chunks)]
    store.build_index(embs, metadatas)
    return store, embedder


def _run_question(store, embedder, question: str) -> dict:
    graph = make_orchestrator(store, embedder, top_k=EVAL_TOP_K)
    return graph.invoke(build_initial_state(question, max_attempts=3))


def main() -> int:
    questions = load_golden_qa(GOLDEN_QA_PATH)
    store, embedder = _build_eval_store()

    grounded_total = 0
    grounded_passed = 0

    for entry in questions:
        question = entry["question"]
        expected_pages = entry["expected_pages"]
        adversarial = entry.get("adversarial", False)

        final_state = _run_question(store, embedder, question)
        retrieved_chunks = final_state.get("retrieved_chunks", [])
        citation_verification = final_state.get("citation_verification") or {}
        verified_ratio = citation_verification.get("verified_ratio", 0.0)
        score = final_state.get("reliability_score")
        decision = final_state.get("reliability_decision")
        degraded = final_state.get("degraded_mode")

        retrieval_hit = any(
            chunk_covers_expected_page(c.get("source", {}), expected_pages) for c in retrieved_chunks
        )
        verification_ok = passes_verification_floor(verified_ratio)

        print(f"\nQ: {question}")
        print(f"   degraded_mode={degraded} score={score} decision={decision} verified_ratio={verified_ratio:.2f}")

        if adversarial:
            print("   (adversarial, informational only -- not scored pass/fail)")
            continue

        grounded_total += 1
        passed = retrieval_hit and verification_ok
        if passed:
            grounded_passed += 1
        print(
            f"   retrieval={'PASS' if retrieval_hit else 'FAIL'} (expected pages {expected_pages}) "
            f"verification={'PASS' if verification_ok else 'FAIL'} (floor {MIN_VERIFIED_RATIO})"
        )

    print(f"\n{grounded_passed}/{grounded_total} grounded questions passed both checks")
    return 0 if grounded_passed == grounded_total else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the existing pytest suite to confirm nothing broke**

Run: `python -m pytest -q`
Expected: PASS, same count as before this task (this task adds no new pytest tests — `main()`, `_build_eval_store`, `_run_question` are exercised only by Step 3's manual run)

- [ ] **Step 3: Run the harness for real**

Run: `PYTHONPATH=. python scripts/run_eval.py`

Expected in *this* environment (no `OPENAI_API_KEY` set, confirmed earlier this session — no `.env` file exists): the script runs to completion without a traceback, prints a block for each of the 12 questions followed by a summary line, `degraded_mode=True` for every question, `retrieval=PASS` for most or all of the 11 grounded questions (retrieval doesn't need an LLM — only embeddings — so it should still work correctly in degraded mode), and `verification=FAIL` for **every** grounded question. That last part is expected, not a bug: in degraded mode `SummarizerAgent._heuristic_summary` returns `citations: []`, and `CitationVerifierAgent.verify([], ...)` always returns `verified_ratio: 0.0` when there are no citations to check — there is no way to pass citation verification without a real LLM generating real citations. The script's exit code will be `1` in this environment for the same reason. This confirms the harness's plumbing (fixture generation, chunking, retrieval, the report itself) works correctly; confirming genuine answer-quality behavior requires running it again with a real `OPENAI_API_KEY` set.

- [ ] **Step 4: Lint**

Run: `python -m isort --check-only scripts/run_eval.py && python -m flake8 --max-line-length=120 scripts/run_eval.py`
Expected: no output (clean)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_eval.py
git commit -m "Add the golden-eval harness orchestration and report"
```

---

## Task 5: Full verification

**Files:** none modified — this task only runs checks.

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q`
Expected: PASS, all tests (60 existing + 3 new from Task 1 + 3 new from Task 2 + 6 new from Task 3 = 72 total)

- [ ] **Step 2: Full lint gate (the existing CI-covered paths, unchanged)**

Run: `python -m isort --check-only src tests app.py && python -m flake8 --max-line-length=120 src tests app.py`
Expected: no output (clean)

- [ ] **Step 3: Lint the new scripts/ paths too (not CI-covered, but kept clean)**

Run: `python -m isort --check-only scripts && python -m flake8 --max-line-length=120 scripts`
Expected: no output (clean)

- [ ] **Step 4: Re-run the harness once more against the final state of the code**

Run: `PYTHONPATH=. python scripts/run_eval.py`
Expected: same shape of output as Task 4 Step 3 (all `retrieval=PASS`, all `verification=FAIL` in this no-API-key environment, exit code 1). If a real `OPENAI_API_KEY` is available when this plan is actually executed, expect most or all grounded questions to show `retrieval=PASS` and `verification=PASS`, `degraded_mode=False`, and the adversarial question to show a low score/verified_ratio (informational, not scored).

---

## Self-review

- **Spec coverage:** every section of `docs/superpowers/specs/2026-08-27-golden-eval-harness-design.md` maps to a task: fixture paper → Task 1; golden question set/loader → Task 2; the two objective checks → Task 3; harness orchestration, report, and exit code → Task 4; the calibrated `chunk_size=300`/`top_k=2` parameters and the `MIN_VERIFIED_RATIO=0.5` floor are both reflected in Task 4's code. ✓
- **Placeholders:** no "TBD"/"handle appropriately". The one deliberately-unverifiable-in-this-environment item (real-LLM answer quality, as opposed to plumbing) is explicitly called out with a reason in Task 4 Step 3 and Task 5 Step 4, not left as a silent gap. ✓
- **Type/name consistency:** `GOLDEN_QA_PATH`, `load_golden_qa`, `MIN_VERIFIED_RATIO`, `chunk_covers_expected_page`, `passes_verification_floor`, `build_golden_paper`, `PAGES` are named identically everywhere they're produced and consumed across Tasks 1-4. `EVAL_CHUNK_SIZE`/`EVAL_CHUNK_OVERLAP`/`EVAL_TOP_K` are only introduced in Task 4 and used only there, consistent with the spec's harness-only-parameter framing. ✓
