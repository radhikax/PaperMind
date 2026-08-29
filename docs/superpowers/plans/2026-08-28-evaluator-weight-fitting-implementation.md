# Evaluator Weight Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ReliabilityEvaluator`'s hardcoded blend weights with ones fit live from real labeled feedback once there's enough data, and close a real gap where a low citation-verification signal could be outvoted by other signals and still cross the accept threshold.

**Architecture:** A new `src/weight_fitting.py` module (mirroring `src/calibration.py`'s shape exactly) reads `feedback.jsonl`, fits a `sklearn.linear_model.LogisticRegression` on the four component scores once there are at least 30 records with at least 5 of each label, and re-fits fresh on every `make_orchestrator()` call — no separate script, no persisted file. `ReliabilityEvaluator` uses the fitted model's `sigmoid(...)` probability as the raw score when active, falling back to today's weighted-sum formula otherwise, and now requires `citation_verified_ratio` to independently clear its own floor before accepting, not just contribute to the blend.

**Tech Stack:** Python 3.10, `scikit-learn` (new explicit dependency, already present transitively via `sentence-transformers`, confirmed importable at `1.7.2`), existing `src.*` modules — no other new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-28-evaluator-weight-fitting-design.md`.

## Global Constraints

- Don't break the existing 73 passing tests or the CI lint gate (`isort --check-only src tests app.py`, `flake8 --max-line-length=120 src tests app.py`).
- `scikit-learn` is the one new dependency this plan adds — confirmed already importable in this environment, added to `requirements.txt` explicitly rather than relied on transitively.
- No separate fitting script, no persisted weights file — `load_fitted_weights()` re-fits fresh from `feedback.jsonl` on every call, matching `load_calibrator()`'s existing pattern exactly.
- `accept_threshold` is not touched by this plan — it stays an explicit constructor argument (see spec's "Decision: `accept_threshold` stays an explicit policy constant").

---

## Task 1: Weight-fitting mechanism

**Files:**
- Create: `src/weight_fitting.py`
- Test: `tests/test_weight_fitting.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `COMPONENT_KEYS: List[str]` (`["semantic", "verifier", "critic_confidence", "citation_verified_ratio"]`); `MIN_FIT_SAMPLES: int` (30); `MIN_PER_CLASS_SAMPLES: int` (5); `load_feedback_components(path: str = FEEDBACK_PATH) -> List[dict]`; `FittedWeights` class with `coefficients`, `intercept`, `n_samples`, `active` property, `raw_score(components: dict) -> int`; `fit_fitted_weights(records: List[dict], min_samples: int = MIN_FIT_SAMPLES, min_per_class: int = MIN_PER_CLASS_SAMPLES) -> FittedWeights`; `load_fitted_weights(path: str = FEEDBACK_PATH) -> FittedWeights`.

This module is entirely independent of `ReliabilityEvaluator` — it can be built and tested on its own before anything else touches it, matching how `src/calibration.py` is a standalone module `ReliabilityEvaluator` merely consumes.

- [ ] **Step 1: Add the `scikit-learn` dependency**

Find (in `requirements.txt`):
```
pydantic
flake8
isort
pandas
```

Replace with:
```
pydantic
flake8
isort
pandas
scikit-learn>=1.7.0,<2.0.0
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_weight_fitting.py
import json

from src.weight_fitting import (fit_fitted_weights, load_feedback_components,
                                 load_fitted_weights)


def _write_feedback(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _record(label, semantic=0.8, verifier=0.8, critic_confidence=0.8, citation_verified_ratio=0.8):
    return {
        "label": label,
        "components": {
            "semantic": semantic,
            "verifier": verifier,
            "critic_confidence": critic_confidence,
            "citation_verified_ratio": citation_verified_ratio,
        },
    }


def test_load_feedback_components_skips_records_without_components(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text(
        json.dumps({"label": "accurate"}) + "\n"
        + json.dumps(_record("accurate")) + "\n"
        + "not json\n",
        encoding="utf-8",
    )
    records = load_feedback_components(str(path))
    assert len(records) == 1
    assert records[0]["label"] == "accurate"


def test_load_feedback_components_skips_incomplete_components(tmp_path):
    path = tmp_path / "feedback.jsonl"
    incomplete = _record("accurate")
    del incomplete["components"]["citation_verified_ratio"]
    path.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")
    assert load_feedback_components(str(path)) == []


def test_fit_fitted_weights_inactive_below_min_samples():
    accurate = [_record("accurate")] * 10
    hallucinated = [_record("hallucinated", semantic=0.1, verifier=0.1, critic_confidence=0.1,
                             citation_verified_ratio=0.1)] * 10
    weights = fit_fitted_weights(accurate + hallucinated, min_samples=30, min_per_class=5)
    assert weights.active is False
    assert weights.n_samples == 20


def test_fit_fitted_weights_inactive_below_min_per_class():
    accurate = [_record("accurate")] * 28
    hallucinated = [_record("hallucinated", semantic=0.1, verifier=0.1, critic_confidence=0.1,
                             citation_verified_ratio=0.1)] * 2
    weights = fit_fitted_weights(accurate + hallucinated, min_samples=30, min_per_class=5)
    assert weights.active is False
    assert weights.n_samples == 30


def test_fit_fitted_weights_active_with_enough_balanced_data():
    accurate = [_record("accurate", semantic=0.85, verifier=0.8, critic_confidence=0.85,
                         citation_verified_ratio=0.9)] * 15
    hallucinated = [_record("hallucinated", semantic=0.15, verifier=0.2, critic_confidence=0.15,
                             citation_verified_ratio=0.1)] * 15
    weights = fit_fitted_weights(accurate + hallucinated, min_samples=30, min_per_class=5)
    assert weights.active is True
    assert weights.n_samples == 30
    high_score = weights.raw_score(
        {"semantic": 0.9, "verifier": 0.85, "critic_confidence": 0.9, "citation_verified_ratio": 0.9}
    )
    low_score = weights.raw_score(
        {"semantic": 0.1, "verifier": 0.15, "critic_confidence": 0.1, "citation_verified_ratio": 0.1}
    )
    assert high_score > low_score


def test_load_fitted_weights_reads_from_disk(tmp_path):
    path = tmp_path / "feedback.jsonl"
    accurate = [_record("accurate", semantic=0.85, verifier=0.8, critic_confidence=0.85,
                         citation_verified_ratio=0.9)] * 15
    hallucinated = [_record("hallucinated", semantic=0.15, verifier=0.2, critic_confidence=0.15,
                             citation_verified_ratio=0.1)] * 15
    _write_feedback(path, accurate + hallucinated)
    weights = load_fitted_weights(str(path))
    assert weights.active is True
    assert weights.n_samples == 30


def test_load_fitted_weights_missing_file_is_inactive(tmp_path):
    weights = load_fitted_weights(str(tmp_path / "does-not-exist.jsonl"))
    assert weights.n_samples == 0
    assert weights.active is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_weight_fitting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.weight_fitting'`

- [ ] **Step 4: Write the implementation**

```python
# src/weight_fitting.py
"""Fits ReliabilityEvaluator's blend weights from real accurate/hallucinated
labels in `feedback.jsonl`, replacing the hardcoded SEMANTIC_WEIGHT/etc.
constants with a logistic regression once there's enough labeled data.

Mirrors src/calibration.py's shape: a live re-fit on every call, gated by a
minimum-sample threshold, with a graceful "not enough data yet" inactive
state rather than a hard failure. MIN_FIT_SAMPLES is higher than
calibration's MIN_SAMPLES (8) because a 4-input logistic regression needs
more data to fit stably than a single-dimension isotonic curve does -- a
common rule of thumb wants roughly 10-20 examples per input parameter.
"""
import json
import math
import os
from typing import Dict, List, Optional

from sklearn.linear_model import LogisticRegression

FEEDBACK_PATH = "feedback.jsonl"
MIN_FIT_SAMPLES = 30
MIN_PER_CLASS_SAMPLES = 5
COMPONENT_KEYS = ["semantic", "verifier", "critic_confidence", "citation_verified_ratio"]


def load_feedback_components(path: str = FEEDBACK_PATH) -> List[dict]:
    """Read feedback records that carry all four component scores.

    Older records (from before this phase) lack a "components" dict entirely
    and are silently skipped -- not migrated.
    """
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            components = rec.get("components")
            if (
                isinstance(components, dict)
                and all(isinstance(components.get(k), (int, float)) for k in COMPONENT_KEYS)
                and rec.get("label") in ("accurate", "hallucinated")
            ):
                records.append(rec)
    return records


class FittedWeights:
    """score = sigmoid(coefficients . components + intercept) * 100, or the inactive fallback."""

    def __init__(self, coefficients: Optional[Dict[str, float]], intercept: Optional[float], n_samples: int):
        self.coefficients = coefficients
        self.intercept = intercept
        self.n_samples = n_samples

    @property
    def active(self) -> bool:
        return self.coefficients is not None

    def raw_score(self, components: Dict[str, float]) -> int:
        z = self.intercept + sum(self.coefficients[k] * components[k] for k in COMPONENT_KEYS)
        probability = 1.0 / (1.0 + math.exp(-z))
        return int(round(probability * 100))


def fit_fitted_weights(
    records: List[dict],
    min_samples: int = MIN_FIT_SAMPLES,
    min_per_class: int = MIN_PER_CLASS_SAMPLES,
) -> FittedWeights:
    n = len(records)
    accurate_count = sum(1 for r in records if r["label"] == "accurate")
    hallucinated_count = n - accurate_count

    if n < min_samples or accurate_count < min_per_class or hallucinated_count < min_per_class:
        return FittedWeights(coefficients=None, intercept=None, n_samples=n)

    x = [[r["components"][k] for k in COMPONENT_KEYS] for r in records]
    y = [1 if r["label"] == "accurate" else 0 for r in records]

    try:
        model = LogisticRegression()
        model.fit(x, y)
    except Exception:
        return FittedWeights(coefficients=None, intercept=None, n_samples=n)

    coefficients = {k: float(w) for k, w in zip(COMPONENT_KEYS, model.coef_[0])}
    intercept = float(model.intercept_[0])
    return FittedWeights(coefficients=coefficients, intercept=intercept, n_samples=n)


def load_fitted_weights(path: str = FEEDBACK_PATH) -> FittedWeights:
    return fit_fitted_weights(load_feedback_components(path))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_weight_fitting.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Lint**

Run: `python -m isort --check-only src/weight_fitting.py tests/test_weight_fitting.py && python -m flake8 --max-line-length=120 src/weight_fitting.py tests/test_weight_fitting.py`
Expected: no output (clean)

- [ ] **Step 7: Commit**

```bash
git add src/weight_fitting.py tests/test_weight_fitting.py requirements.txt
git commit -m "Add live weight-fitting mechanism from labeled feedback"
```

---

## Task 2: Citation floor and fitted-weights wiring in `ReliabilityEvaluator`

**Files:**
- Modify: `src/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `FittedWeights` from Task 1 (only via duck typing — `ReliabilityEvaluator` calls `.active` and `.raw_score(components)`, never imports `FittedWeights` directly, matching how it already duck-types `calibrator`).
- Produces: `ReliabilityEvaluator.CITATION_VERIFIED_FLOOR: float` (0.5); `ReliabilityEvaluator.__init__(..., fitted_weights=None)`; `evaluate()`'s returned dict gains `components`, `fitted_weights_active`, `fitted_weights_samples` keys; acceptance now requires `citation_verified_ratio >= CITATION_VERIFIED_FLOOR` in addition to the score threshold.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluator.py`, after `test_verifier_overlap_ratio_no_original_ids_is_zero` (before the existing `test_reliability_evaluator_accepts_high_score`):

```python
def test_reliability_evaluator_blocks_accept_when_citation_floor_not_met_even_with_high_score():
    evaluator = ReliabilityEvaluator(accept_threshold=70)
    result = evaluator.evaluate(
        "a summary",
        retrieved_chunks=[{"id": 1}],
        store=DummyStore(score=1.0, ids=(1,)),
        embedder=DummyEmbedder(),
        critic_assessment={"confidence": 1.0, "hallucination_rate": 0.0},
        citation_verified_ratio=0.3,
        attempt=1,
        max_attempts=3,
    )
    assert result["raw_score"] >= 70
    assert result["decision"] != "accept"


def test_reliability_evaluator_returns_component_scores():
    evaluator = ReliabilityEvaluator(accept_threshold=70)
    result = evaluator.evaluate(
        "a summary",
        retrieved_chunks=[{"id": 1}],
        store=DummyStore(score=0.8, ids=(1,)),
        embedder=DummyEmbedder(),
        critic_assessment={"confidence": 0.7, "hallucination_rate": 0.1},
        citation_verified_ratio=0.9,
        attempt=1,
        max_attempts=3,
    )
    assert result["components"] == {
        "semantic": 0.8,
        "verifier": 1.0,
        "critic_confidence": 0.7,
        "citation_verified_ratio": 0.9,
    }


class FakeFittedWeights:
    def __init__(self, active, n_samples):
        self.active = active
        self.n_samples = n_samples

    def raw_score(self, components):
        return 42


def test_reliability_evaluator_uses_fitted_weights_when_active():
    evaluator = ReliabilityEvaluator(accept_threshold=70, fitted_weights=FakeFittedWeights(active=True, n_samples=40))
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
    assert result["raw_score"] == 42
    assert result["fitted_weights_active"] is True
    assert result["fitted_weights_samples"] == 40


def test_reliability_evaluator_falls_back_to_hardcoded_weights_when_fitted_inactive():
    evaluator = ReliabilityEvaluator(
        accept_threshold=70, fitted_weights=FakeFittedWeights(active=False, n_samples=3)
    )
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
    assert result["raw_score"] != 42
    assert result["fitted_weights_active"] is False
    assert result["fitted_weights_samples"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluator.py -v -k "citation_floor or component_scores or fitted_weights"`
Expected: FAIL — `test_reliability_evaluator_blocks_accept_when_citation_floor_not_met_even_with_high_score` fails on `assert result["decision"] != "accept"` (today's formula ignores the floor entirely); `test_reliability_evaluator_returns_component_scores` and the two `fitted_weights` tests fail with `KeyError: 'components'` / `TypeError: __init__() got an unexpected keyword argument 'fitted_weights'`.

- [ ] **Step 3: Update `ReliabilityEvaluator`**

Find (in `src/evaluator.py`):
```python
class ReliabilityEvaluator:
    """Aggregates multiple signals into a 0-100 reliability score and a routing decision."""

    SEMANTIC_WEIGHT = 0.40
    VERIFIER_WEIGHT = 0.25
    CRITIC_WEIGHT = 0.20
    CITATION_WEIGHT = 0.15

    def __init__(self, accept_threshold: int = 70, calibrator=None):
        self.accept_threshold = accept_threshold
        self.calibrator = calibrator

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
        raw_score = int(round(max(0.0, min(1.0, base)) * 100))
        score = self.calibrator.apply(raw_score) if self.calibrator else raw_score
        calibration_active = bool(self.calibrator and self.calibrator.active)
        calibration_samples = self.calibrator.n_samples if self.calibrator else 0

        verdict = {
            "score": score,
            "raw_score": raw_score,
            "calibration_active": calibration_active,
            "calibration_samples": calibration_samples,
        }

        if score >= self.accept_threshold:
            return {**verdict, "decision": "accept", "critique_feedback": None}

        if attempt >= max_attempts:
            return {**verdict, "decision": "exhausted", "critique_feedback": None}

        critique_feedback = self._build_critique(semantic, verifier, critic_assessment, citation_verified_ratio)
        return {**verdict, "decision": "revise", "critique_feedback": critique_feedback}
```

Replace with:
```python
class ReliabilityEvaluator:
    """Aggregates multiple signals into a 0-100 reliability score and a routing decision."""

    SEMANTIC_WEIGHT = 0.40
    VERIFIER_WEIGHT = 0.25
    CRITIC_WEIGHT = 0.20
    CITATION_WEIGHT = 0.15
    CITATION_VERIFIED_FLOOR = 0.5

    def __init__(self, accept_threshold: int = 70, calibrator=None, fitted_weights=None):
        self.accept_threshold = accept_threshold
        self.calibrator = calibrator
        self.fitted_weights = fitted_weights

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

        components = {
            "semantic": semantic,
            "verifier": verifier,
            "critic_confidence": critic_conf,
            "citation_verified_ratio": citation_verified_ratio,
        }

        if self.fitted_weights and self.fitted_weights.active:
            raw_score = self.fitted_weights.raw_score(components)
        else:
            base = (
                self.SEMANTIC_WEIGHT * semantic
                + self.VERIFIER_WEIGHT * verifier
                + self.CRITIC_WEIGHT * critic_conf
                + self.CITATION_WEIGHT * citation_verified_ratio
            )
            raw_score = int(round(max(0.0, min(1.0, base)) * 100))

        score = self.calibrator.apply(raw_score) if self.calibrator else raw_score
        calibration_active = bool(self.calibrator and self.calibrator.active)
        calibration_samples = self.calibrator.n_samples if self.calibrator else 0
        fitted_weights_active = bool(self.fitted_weights and self.fitted_weights.active)
        fitted_weights_samples = self.fitted_weights.n_samples if self.fitted_weights else 0

        verdict = {
            "score": score,
            "raw_score": raw_score,
            "calibration_active": calibration_active,
            "calibration_samples": calibration_samples,
            "fitted_weights_active": fitted_weights_active,
            "fitted_weights_samples": fitted_weights_samples,
            "components": components,
        }

        if score >= self.accept_threshold and citation_verified_ratio >= self.CITATION_VERIFIED_FLOOR:
            return {**verdict, "decision": "accept", "critique_feedback": None}

        if attempt >= max_attempts:
            return {**verdict, "decision": "exhausted", "critique_feedback": None}

        critique_feedback = self._build_critique(semantic, verifier, critic_assessment, citation_verified_ratio)
        return {**verdict, "decision": "revise", "critique_feedback": critique_feedback}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: PASS (12 tests: the 8 existing plus the 4 new ones)

- [ ] **Step 5: Lint**

Run: `python -m isort --check-only src/evaluator.py tests/test_evaluator.py && python -m flake8 --max-line-length=120 src/evaluator.py tests/test_evaluator.py`
Expected: no output (clean)

- [ ] **Step 6: Commit**

```bash
git add src/evaluator.py tests/test_evaluator.py
git commit -m "Add a hard citation-verification floor and fitted-weights support to ReliabilityEvaluator"
```

---

## Task 3: Thread component scores and fitted weights through the orchestrator

**Files:**
- Modify: `src/langgraph_agents.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `load_fitted_weights` from Task 1 (`src/weight_fitting.py`); the `fitted_weights`/`components`/`fitted_weights_active`/`fitted_weights_samples` keys from Task 2.
- Produces: `GraphState` gains `component_scores: Optional[Dict[str, float]]`, `fitted_weights_active: bool`, `fitted_weights_samples: int`; `make_orchestrator(..., fitted_weights=None)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_orchestrator.py`, after `test_graph_revises_a_low_quality_summary_then_accepts`:

```python
def test_component_scores_and_fitted_weights_thread_through_final_state():
    from src.weight_fitting import FittedWeights

    store = DummyStore(score=0.95, ids=(1,))
    embedder = DummyEmbedder()
    summarizer = ScriptedSummarizer(
        [{"summary": "Good summary.", "citations": [{"chunk_id": 1, "page": "1", "excerpt": "source text"}],
          "valid": True}]
    )
    critic = FixedCritic({"confidence": 0.9, "hallucination_rate": 0.05, "notes": "fine"})
    fitted_weights = FittedWeights(
        coefficients={"semantic": 2.0, "verifier": 0.0, "critic_confidence": 0.0, "citation_verified_ratio": 0.0},
        intercept=0.0,
        n_samples=40,
    )

    graph = make_orchestrator(store, embedder, summarizer=summarizer, critic=critic, fitted_weights=fitted_weights)
    final_state = graph.invoke(build_initial_state("what is this paper about?", max_attempts=1))

    assert final_state["component_scores"] == {
        "semantic": 0.95,
        "verifier": 1.0,
        "critic_confidence": 0.9,
        "citation_verified_ratio": 1.0,
    }
    assert final_state["fitted_weights_active"] is True
    assert final_state["fitted_weights_samples"] == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py -v -k component_scores`
Expected: FAIL with `TypeError: make_orchestrator() got an unexpected keyword argument 'fitted_weights'`

- [ ] **Step 3: Update `GraphState`, `build_initial_state`, and `make_orchestrator`**

Find (in `src/langgraph_agents.py`):
```python
from src.agents import (CitationVerifierAgent, CriticAgent, RetrieverAgent,
                        SummarizerAgent)
from src.calibration import load_calibrator
from src.evaluator import ReliabilityEvaluator


class GraphState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]
    critic_assessment: Optional[Dict[str, Any]]
    citation_verification: Optional[Dict[str, Any]]
    reliability_score: Optional[int]
    reliability_raw_score: Optional[int]
    calibration_active: bool
    calibration_samples: int
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
        "reliability_raw_score": None,
        "calibration_active": False,
        "calibration_samples": 0,
        "reliability_decision": None,
        "critique_feedback": None,
        "attempt": 0,
        "max_attempts": max_attempts,
        "history": [],
        "degraded_mode": False,
    }
```

Replace with:
```python
from src.agents import (CitationVerifierAgent, CriticAgent, RetrieverAgent,
                        SummarizerAgent)
from src.calibration import load_calibrator
from src.evaluator import ReliabilityEvaluator
from src.weight_fitting import load_fitted_weights


class GraphState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]
    critic_assessment: Optional[Dict[str, Any]]
    citation_verification: Optional[Dict[str, Any]]
    reliability_score: Optional[int]
    reliability_raw_score: Optional[int]
    calibration_active: bool
    calibration_samples: int
    fitted_weights_active: bool
    fitted_weights_samples: int
    component_scores: Optional[Dict[str, float]]
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
        "reliability_raw_score": None,
        "calibration_active": False,
        "calibration_samples": 0,
        "fitted_weights_active": False,
        "fitted_weights_samples": 0,
        "component_scores": None,
        "reliability_decision": None,
        "critique_feedback": None,
        "attempt": 0,
        "max_attempts": max_attempts,
        "history": [],
        "degraded_mode": False,
    }
```

Find:
```python
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
    calibrator=None,
):
    """Build and compile the reliability-gated multi-agent graph.

    `summarizer`/`critic`/`citation_verifier`/`evaluator`/`calibrator` are
    optional dependency-injection overrides (used by tests); omit them for
    real use. When `evaluator` is omitted, the default one calibrates its
    score against `feedback.jsonl` (see `src/calibration.py`) unless
    `calibrator` is also given explicitly.
    """
    retriever = RetrieverAgent(store, embedder)
    summarizer = summarizer or SummarizerAgent(llm_model=(llm or "gpt-4o-mini"))
    critic = critic or CriticAgent(llm_model=(llm or "gpt-4o-mini"))
    citation_verifier = citation_verifier or CitationVerifierAgent()
    if evaluator is None:
        calibrator = calibrator if calibrator is not None else load_calibrator()
        evaluator = ReliabilityEvaluator(accept_threshold=accept_threshold, calibrator=calibrator)
```

Replace with:
```python
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
    calibrator=None,
    fitted_weights=None,
):
    """Build and compile the reliability-gated multi-agent graph.

    `summarizer`/`critic`/`citation_verifier`/`evaluator`/`calibrator`/
    `fitted_weights` are optional dependency-injection overrides (used by
    tests); omit them for real use. When `evaluator` is omitted, the default
    one calibrates its score against `feedback.jsonl` (see
    `src/calibration.py`) unless `calibrator` is also given explicitly, and
    uses weights fit live from `feedback.jsonl` (see `src/weight_fitting.py`)
    unless `fitted_weights` is also given explicitly.
    """
    retriever = RetrieverAgent(store, embedder)
    summarizer = summarizer or SummarizerAgent(llm_model=(llm or "gpt-4o-mini"))
    critic = critic or CriticAgent(llm_model=(llm or "gpt-4o-mini"))
    citation_verifier = citation_verifier or CitationVerifierAgent()
    if evaluator is None:
        calibrator = calibrator if calibrator is not None else load_calibrator()
        fitted_weights = fitted_weights if fitted_weights is not None else load_fitted_weights()
        evaluator = ReliabilityEvaluator(
            accept_threshold=accept_threshold, calibrator=calibrator, fitted_weights=fitted_weights
        )
```

- [ ] **Step 4: Update `reliability_evaluator_node`**

Find:
```python
        return {
            "reliability_score": verdict["score"],
            "reliability_raw_score": verdict["raw_score"],
            "calibration_active": verdict["calibration_active"],
            "calibration_samples": verdict["calibration_samples"],
            "reliability_decision": verdict["decision"],
            "critique_feedback": verdict["critique_feedback"],
            "history": state["history"] + [history_entry],
        }
```

Replace with:
```python
        return {
            "reliability_score": verdict["score"],
            "reliability_raw_score": verdict["raw_score"],
            "calibration_active": verdict["calibration_active"],
            "calibration_samples": verdict["calibration_samples"],
            "fitted_weights_active": verdict["fitted_weights_active"],
            "fitted_weights_samples": verdict["fitted_weights_samples"],
            "component_scores": verdict["components"],
            "reliability_decision": verdict["decision"],
            "critique_feedback": verdict["critique_feedback"],
            "history": state["history"] + [history_entry],
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: PASS (5 tests: the 4 existing plus the new one)

- [ ] **Step 6: Lint**

Run: `python -m isort --check-only src/langgraph_agents.py tests/test_orchestrator.py && python -m flake8 --max-line-length=120 src/langgraph_agents.py tests/test_orchestrator.py`
Expected: no output (clean)

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS, all tests (this exercises `load_fitted_weights()` for real against the repo's actual `feedback.jsonl` in every test that doesn't override `fitted_weights` explicitly — since that file has only one pre-Phase-4 record with no `components`, it resolves to `n_samples=0`, inactive, falling back to hardcoded weights, exactly like `load_calibrator()` already behaves today with the same file)

- [ ] **Step 8: Commit**

```bash
git add src/langgraph_agents.py tests/test_orchestrator.py
git commit -m "Thread component scores and fitted weights through the orchestrator"
```

---

## Task 4: Wire into `app.py`

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `MIN_FIT_SAMPLES`, `load_fitted_weights` from `src.weight_fitting` (Task 1); `component_scores`/`fitted_weights_active`/`fitted_weights_samples` from `GraphState` (Task 3).

No existing automated test covers `app.py` (consistent with every prior phase this session) — verified via `py_compile` plus a manual/scripted check, same approach as before.

- [ ] **Step 1: Add the import**

Find:
```python
from src.calibration import MIN_SAMPLES as MIN_CALIBRATION_SAMPLES
from src.embeddings import EmbeddingModel
from src.ingest import format_page_label, load_and_chunk
from src.langgraph_agents import build_initial_state, make_orchestrator
from src.paper_registry import (index_path_for, list_registered_papers,
                                register_paper)
from src.vectorstore import FaissStore
```

Replace with:
```python
from src.calibration import MIN_SAMPLES as MIN_CALIBRATION_SAMPLES
from src.embeddings import EmbeddingModel
from src.ingest import format_page_label, load_and_chunk
from src.langgraph_agents import build_initial_state, make_orchestrator
from src.paper_registry import (index_path_for, list_registered_papers,
                                register_paper)
from src.vectorstore import FaissStore
from src.weight_fitting import MIN_FIT_SAMPLES
```

- [ ] **Step 2: Add a fitted-weights status banner to the Feedback Dashboard**

Find:
```python
    from src.calibration import load_calibrator
    calibrator = load_calibrator()
    if calibrator.active:
        st.success(
            f"Score calibration is active, fit on {calibrator.n_samples} labeled answers. "
            "Reliability scores shown in QA mode are adjusted by this curve before the "
            "accept/revise decision is made."
        )
    else:
        st.info(
            f"Score calibration is inactive — {calibrator.n_samples}/{MIN_CALIBRATION_SAMPLES} "
            "labeled answers collected. Every click below moves it closer to kicking in."
        )
```

Replace with:
```python
    from src.calibration import load_calibrator
    calibrator = load_calibrator()
    if calibrator.active:
        st.success(
            f"Score calibration is active, fit on {calibrator.n_samples} labeled answers. "
            "Reliability scores shown in QA mode are adjusted by this curve before the "
            "accept/revise decision is made."
        )
    else:
        st.info(
            f"Score calibration is inactive — {calibrator.n_samples}/{MIN_CALIBRATION_SAMPLES} "
            "labeled answers collected. Every click below moves it closer to kicking in."
        )

    from src.weight_fitting import load_fitted_weights
    fitted_weights = load_fitted_weights()
    if fitted_weights.active:
        st.success(
            f"Evaluator weights are fitted from data, based on {fitted_weights.n_samples} labeled "
            "answers with component scores. The reliability score's four signals are weighted by "
            "what actually predicted accuracy in past feedback, not hand-picked constants."
        )
    else:
        st.info(
            f"Evaluator weights are still the hardcoded defaults — {fitted_weights.n_samples}/"
            f"{MIN_FIT_SAMPLES} labeled answers with component scores collected."
        )
```

- [ ] **Step 3: Log component scores in the feedback handlers**

Find:
```python
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
```

Replace with:
```python
            if allow_accept and col1.button("Accurate"):
                import datetime
                fb = {
                    "query": query,
                    "summary": summary_text,
                    "confidence": reliability_score,
                    "label": "accurate",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "paper_id": st.session_state.get('paper_id'),
                    "components": final_state.get('component_scores'),
                }
```

Find:
```python
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
```

Replace with:
```python
            if allow_accept and col2.button("Hallucinated"):
                import datetime
                fb = {
                    "query": query,
                    "summary": summary_text,
                    "confidence": reliability_score,
                    "label": "hallucinated",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "paper_id": st.session_state.get('paper_id'),
                    "components": final_state.get('component_scores'),
                }
```

- [ ] **Step 4: Show fitted-weights status in QA mode, next to the calibration caption**

Find:
```python
            calibration_samples = final_state.get('calibration_samples', 0)
            if final_state.get('calibration_active'):
                raw_score = final_state.get('reliability_raw_score')
                if raw_score != reliability_score:
                    st.caption(
                        f"Calibrated using {calibration_samples} past labeled answers — "
                        f"the raw formula scored this {raw_score}."
                    )
                else:
                    st.caption(f"Calibrated using {calibration_samples} past labeled answers.")
            else:
                st.caption(
                    f"Calibration inactive — {calibration_samples}/{MIN_CALIBRATION_SAMPLES} labeled "
                    "answers collected. Score is the raw formula, unchecked against outcomes."
                )
```

Replace with:
```python
            calibration_samples = final_state.get('calibration_samples', 0)
            if final_state.get('calibration_active'):
                raw_score = final_state.get('reliability_raw_score')
                if raw_score != reliability_score:
                    st.caption(
                        f"Calibrated using {calibration_samples} past labeled answers — "
                        f"the raw formula scored this {raw_score}."
                    )
                else:
                    st.caption(f"Calibrated using {calibration_samples} past labeled answers.")
            else:
                st.caption(
                    f"Calibration inactive — {calibration_samples}/{MIN_CALIBRATION_SAMPLES} labeled "
                    "answers collected. Score is the raw formula, unchecked against outcomes."
                )

            fitted_weights_samples = final_state.get('fitted_weights_samples', 0)
            if final_state.get('fitted_weights_active'):
                st.caption(
                    f"Reliability score uses evaluator weights fitted on {fitted_weights_samples} "
                    "labeled answers, not hardcoded defaults."
                )
            else:
                st.caption(
                    f"Evaluator weights are still hardcoded defaults — {fitted_weights_samples}/"
                    f"{MIN_FIT_SAMPLES} labeled answers with component scores collected."
                )
```

- [ ] **Step 5: Verify `app.py` compiles cleanly**

Run: `python -m py_compile app.py`
Expected: no output (clean)

- [ ] **Step 6: Scripted verification**

Run this from the repo root (adjust the script path to your scratch directory):

```python
"""Verifies app.py's new component-score threading and fitted-weights
status without needing the Streamlit UI itself."""
import sys

sys.path.insert(0, ".")

from src.embeddings import EmbeddingModel  # noqa: E402
from src.langgraph_agents import build_initial_state, make_orchestrator  # noqa: E402
from src.vectorstore import FaissStore  # noqa: E402
from src.weight_fitting import load_fitted_weights  # noqa: E402

embedder = EmbeddingModel()
texts = ["WidgetNet achieves 94.2% accuracy on the WidgetBench dataset."]
embs = embedder.embed_texts(texts)
store = FaissStore(dim=embs.shape[1])
store.build_index(embs, [{"text": texts[0], "source": {"start_page": 1, "end_page": 1}, "id": 0}])

graph = make_orchestrator(store, embedder)
final_state = graph.invoke(build_initial_state("What accuracy does WidgetNet achieve?", max_attempts=1))

assert "component_scores" in final_state
assert set(final_state["component_scores"].keys()) == {
    "semantic", "verifier", "critic_confidence", "citation_verified_ratio",
}
assert "fitted_weights_active" in final_state
assert final_state["fitted_weights_active"] is False  # real feedback.jsonl has too little data
assert load_fitted_weights().n_samples == 0  # the one existing record predates the components field
print("APP WIRING VERIFICATION: PASS")
```

Expected output: `APP WIRING VERIFICATION: PASS`

- [ ] **Step 7: Run the full test suite and lint**

Run: `python -m pytest -q && python -m isort --check-only src tests app.py && python -m flake8 --max-line-length=120 src tests app.py`
Expected: all tests pass, lint clean.

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "Log component scores and surface fitted-weights status in the UI"
```

---

## Task 5: Reuse the real citation floor constant in the golden-eval harness

**Files:**
- Modify: `scripts/run_eval.py`

**Interfaces:**
- Consumes: `ReliabilityEvaluator.CITATION_VERIFIED_FLOOR` from Task 2.

`scripts/run_eval.py` currently hardcodes its own `MIN_VERIFIED_RATIO = 0.5` with a comment claiming it mirrors the evaluator's literal. This task makes that literally true instead of coincidentally true.

- [ ] **Step 1: Import and reuse the real constant**

Find:
```python
from scripts.fixtures.golden_paper import build_golden_paper
from src.embeddings import EmbeddingModel
from src.ingest import chunk_pages_to_chunks, load_pdf_pages
from src.langgraph_agents import build_initial_state, make_orchestrator
from src.vectorstore import FaissStore

GOLDEN_QA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_qa.jsonl")
MIN_VERIFIED_RATIO = 0.5  # mirrors the literal in ReliabilityEvaluator._build_critique (src/evaluator.py)
```

Replace with:
```python
from scripts.fixtures.golden_paper import build_golden_paper
from src.embeddings import EmbeddingModel
from src.evaluator import ReliabilityEvaluator
from src.ingest import chunk_pages_to_chunks, load_pdf_pages
from src.langgraph_agents import build_initial_state, make_orchestrator
from src.vectorstore import FaissStore

GOLDEN_QA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden_qa.jsonl")
MIN_VERIFIED_RATIO = ReliabilityEvaluator.CITATION_VERIFIED_FLOOR
```

- [ ] **Step 2: Verify the harness still runs (plumbing check, same no-key environment as before)**

Run: `PYTHONPATH=. python scripts/run_eval.py`
Expected: same shape of output as every prior run this session — `degraded_mode=True`, all 11 grounded questions `retrieval=PASS`, all `verification=FAIL` (no `OPENAI_API_KEY` in this environment), exit code `1`. This step exists to confirm the constant swap didn't break the import chain, not to re-verify behavior already established.

- [ ] **Step 3: Lint**

Run: `python -m isort --check-only scripts/run_eval.py && python -m flake8 --max-line-length=120 scripts/run_eval.py`
Expected: no output (clean)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_eval.py
git commit -m "Reuse ReliabilityEvaluator's real citation floor instead of a duplicated literal"
```

---

## Task 6: Full verification

**Files:** none modified — this task only runs checks.

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q`
Expected: PASS, all tests (73 existing + 7 new from Task 1 + 4 new from Task 2 + 1 new from Task 3 = 85 total)

- [ ] **Step 2: Full lint gate**

Run: `python -m isort --check-only src tests app.py && python -m flake8 --max-line-length=120 src tests app.py`
Expected: no output (clean)

- [ ] **Step 3: Re-run the Task 4 scripted verification and the golden-eval harness once more against the final state of the code**

Run the script from Task 4 Step 6, then `PYTHONPATH=. python scripts/run_eval.py`.
Expected: `APP WIRING VERIFICATION: PASS`, then the same harness output shape as Task 5 Step 2.

- [ ] **Step 4: Confirm the citation floor and fitted-weights constants have exactly one source of truth**

Run: `grep -rn "MIN_VERIFIED_RATIO = 0" scripts 2>/dev/null; grep -rn "CITATION_VERIFIED_FLOOR" src scripts 2>/dev/null`
Expected: the first command produces no output (no more standalone hardcoded `0.5` in `scripts/`); the second shows exactly two hits — the definition in `src/evaluator.py` and the reference in `scripts/run_eval.py`.

---

## Self-review

- **Spec coverage:** every decision in `docs/superpowers/specs/2026-08-28-evaluator-weight-fitting-design.md` maps to a task: the sigmoid-of-fitted-coefficients formula and live-refit mechanism → Task 1; the citation floor and fitted-weights wiring into `evaluate()` → Task 2; `GraphState`/`make_orchestrator` threading → Task 3; feedback logging and UI visibility → Task 4; the single-source-of-truth fix for the citation floor constant → Task 5; `requirements.txt` → Task 1 Step 1. ✓
- **Placeholders:** no "TBD"/"handle appropriately" anywhere in the task steps. ✓
- **Type/name consistency:** `COMPONENT_KEYS`, `FittedWeights`, `.active`, `.raw_score()`, `load_fitted_weights`, `fit_fitted_weights`, `load_feedback_components` are named identically everywhere they're produced (Task 1) and consumed (Tasks 2-4). `component_scores`/`fitted_weights_active`/`fitted_weights_samples` are the same three `GraphState` key names from their introduction in Task 3 through their use in Task 4. `CITATION_VERIFIED_FLOOR` is defined once (Task 2) and referenced, not redefined, in Task 5. ✓
