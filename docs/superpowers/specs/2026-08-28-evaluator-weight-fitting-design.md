# Evaluator Weight Fitting — Design Spec

**Status:** Approved design, ready for implementation planning.

**Origin:** Phase 4 of `docs/superpowers/plans/2026-08-23-improvement-roadmap.md`. Depends on Phase 1 (stable per-paper storage, already landed) and benefits from Phase 3 (the golden-eval harness, already landed, used to check a re-fit doesn't quietly regress retrieval/scoring). This document works out the actual mechanism — the roadmap's original sketch turned out to need real revision once checked against the codebase as it stands.

## Problem

`ReliabilityEvaluator.SEMANTIC_WEIGHT` / `VERIFIER_WEIGHT` / `CRITIC_WEIGHT` / `CITATION_WEIGHT` (`src/evaluator.py:35-38`, values 0.40/0.25/0.20/0.15) are constants chosen once and never checked against real outcomes. `ScoreCalibrator` corrects the *output* of the blend formula against labeled feedback, but nothing checks whether those four proportions are even the right ones — a differently-weighted formula might need far less post-hoc correction in the first place.

## Investigation findings

- **`scikit-learn` is already importable** (version 1.7.2 confirmed) as a transitive dependency of `sentence-transformers`, but isn't declared in `requirements.txt`. This phase adds it explicitly rather than relying on transitive luck.
- **`feedback.jsonl` has exactly one record**, from before this session's changes, and it lacks the four component scores entirely — confirming the roadmap's stated problem and that fitting cannot happen yet regardless of design; the mechanism has to be built now and gate itself until enough data exists.
- **`ScoreCalibrator` (`src/calibration.py`) sets a precedent this phase should match, not diverge from.** It has no separate "fit and save a file" step at all — `load_calibrator()` re-fits fresh from `feedback.jsonl` on every `make_orchestrator()` call, gated by `n_samples >= MIN_SAMPLES`. The roadmap's original sketch proposed a separate `scripts/fit_evaluator_weights.py` writing a persisted `evaluator_weights.json`. That's a second source of truth that can drift out of sync with `feedback.jsonl`, for no benefit the live-refit pattern doesn't already provide.
- **A real gap found by cross-referencing the golden-eval harness (Phase 3) against the live evaluator**: `scripts/run_eval.py`'s `MIN_VERIFIED_RATIO = 0.5` treats citation verification as its own hard pass/fail check, completely independent of any blended score. The live `ReliabilityEvaluator` does not enforce that same hard floor — `citation_verified_ratio` is only one of four weighted inputs, so a low value can be outvoted by high `semantic`/`critic_confidence` and the blended score can still cross `accept_threshold`. That's a real path for an answer with unverifiable citations to get shown as trustworthy. This phase closes that gap.

## Decision: sigmoid of fitted coefficients, not a renormalized weighted sum

Two ways to turn "fit weights from data" into a new score formula were considered:

- **Chosen — replace the formula with the fitted model's own probability estimate.** `raw_score = round(sigmoid(w_semantic·semantic + w_verifier·verifier + w_critic·critic_confidence + w_citation·citation_verified_ratio + bias) × 100)`. The four coefficients and bias come directly from a `LogisticRegression` fit on labeled feedback. This is the mathematically faithful use of "fitted logistic regression weights" — the model directly learns what combination of signals predicts `accurate`.
- **Rejected — keep today's weighted-sum-then-clamp formula, but derive the four `_WEIGHT` constants by normalizing the regression's coefficients to sum to 1.** Logistic regression coefficients aren't proportions — they're log-odds weights, and can come out negative (e.g. under multicollinearity between `semantic` and `verifier`, which both come from the same embedding-similarity mechanism). Clamping negatives to 0 and renormalizing the rest distorts what the regression actually found, for the sake of forcing it into a shape it wasn't built to produce.

## Decision: live refit, not a separate script + persisted file

Matches `ScoreCalibrator`'s existing pattern exactly (see Investigation findings). `load_fitted_weights()` reads `feedback.jsonl` and fits fresh on every call — no `scripts/fit_evaluator_weights.py`, no `evaluator_weights.json`. One source of truth (`feedback.jsonl`), consistent with how calibration already works, and no risk of a stale persisted file diverging from the data that produced it.

## Decision: `accept_threshold` stays an explicit policy constant, not fit from data

The four blend weights answer a factual question data can settle: which signals actually correlate with an answer turning out accurate. `accept_threshold` answers a different kind of question: how much risk of showing a wrong answer is acceptable before the system stops and flags it instead — a risk-tolerance choice, not a fact the data uniquely determines. Fitting it directly off historical labels would likely land near the point where accuracy odds cross 50/50, not a deliberately conservative bar; and since the threshold itself decides which answers get labeled at all (only shown-as-accepted answers are what a user is reacting to), continuously re-tuning it against its own past decisions risks a feedback loop that erodes the bar over time rather than holding it fixed. `accept_threshold` remains an explicit constructor argument, unchanged by this phase.

## Decision: add a hard citation floor alongside the blended threshold

Acceptance now requires **both** `score >= accept_threshold` **and** `citation_verified_ratio >= CITATION_VERIFIED_FLOOR`. `CITATION_VERIFIED_FLOOR = 0.5` becomes a class constant on `ReliabilityEvaluator` — the single source of truth for this number. `scripts/run_eval.py`'s `MIN_VERIFIED_RATIO`, which today independently hardcodes the same `0.5` with a comment claiming it "mirrors" the evaluator's literal, is updated to import and reuse the real constant instead of maintaining its own copy that could silently drift out of sync.

## Components and file changes

### `src/evaluator.py`

- New class constant: `CITATION_VERIFIED_FLOOR = 0.5` (see above).
- `evaluate()`'s acceptance check changes from `if score >= self.accept_threshold` to `if score >= self.accept_threshold and citation_verified_ratio >= self.CITATION_VERIFIED_FLOOR`.
- `evaluate()` computes `raw_score` via the fitted-weights sigmoid formula when a `FittedWeights` instance is active (see below), falling back to today's weighted-sum-then-clamp formula using the existing `SEMANTIC_WEIGHT`/etc. constants otherwise — the same two-tier "fitted if possible, hardcoded fallback if not" shape `ScoreCalibrator` already established for the calibration layer.
- `evaluate()`'s returned verdict dict gains: `components` (a dict `{"semantic", "verifier", "critic_confidence", "citation_verified_ratio"}` — the four raw values, needed downstream for feedback logging), `fitted_weights_active` and `fitted_weights_samples` (mirroring `calibration_active`/`calibration_samples`, for UI visibility into whether fitted weights are actually in effect).
- `ReliabilityEvaluator.__init__` gains a `fitted_weights=None` constructor parameter (DI-friendly, matching the existing `calibrator=None` pattern).

### New: `src/weight_fitting.py` (mirrors `src/calibration.py`'s shape and location, not folded into `evaluator.py`, matching how calibration already lives in its own module)

- `FittedWeights` class: holds `coefficients` (dict or `None`), `intercept` (float or `None`), `n_samples` (int). `active` property is `self.coefficients is not None`. A method to compute the sigmoid raw score from a `components` dict.
- `load_feedback_components(path=FEEDBACK_PATH) -> List[dict]`: reads `feedback.jsonl`, keeping only records that carry a `components` dict with all four sub-fields present — older records (like the one existing record today) are silently skipped, not migrated.
- `fit_fitted_weights(records) -> FittedWeights`: requires **both** a total-sample minimum (see below) **and** a minimum count of each label (`accurate` and `hallucinated`) — fitting on all-one-class data is degenerate (`LogisticRegression` either fails or produces a meaningless always-one-answer model). Below either bar, returns a `FittedWeights` with `coefficients=None` (inactive). Any exception from `sklearn`'s fit is caught the same way, falling back to inactive rather than propagating.
- `load_fitted_weights(path=FEEDBACK_PATH) -> FittedWeights`: the live-refit entry point `make_orchestrator` calls, mirroring `load_calibrator()`.
- **Minimum-sample threshold: 30 total records**, deliberately higher than `ScoreCalibrator`'s `MIN_SAMPLES = 8`. A 4-input logistic regression needs meaningfully more data to fit stably than a single-dimension isotonic curve — a common rule of thumb wants roughly 10-20 examples per input parameter, and this model has 4 inputs plus a bias term. **Minimum per-class count: 5 of each label**, to avoid degenerate single-class fits even once the 30-total bar is cleared.

### `src/langgraph_agents.py`

- `make_orchestrator` gains a `fitted_weights=None` parameter, threaded through exactly like `calibrator` is today: `fitted_weights = fitted_weights if fitted_weights is not None else load_fitted_weights()`, passed into `ReliabilityEvaluator(...)`.
- `GraphState` gains a `component_scores: Optional[Dict[str, float]]` field (nested dict, matching the existing pattern for `citation_verification`/`critic_assessment` rather than four new top-level scalar keys).
- `reliability_evaluator_node` returns `component_scores` alongside its existing keys, sourced from `verdict["components"]`.

### `app.py`

- The "Accurate"/"Hallucinated" feedback handlers add `"components": final_state.get('component_scores')` to the `fb` dict written to `feedback.jsonl` — this is what makes future fitting possible; without it the record is the same shape as today's one existing record and gets skipped by `load_feedback_components`.
- A new caption near the existing calibration status line shows whether fitted weights are active and how many samples they're based on, mirroring the existing `"Calibrated using N past labeled answers..."` caption — visibility into whether a shown score used fitted weights or the hardcoded fallback is exactly the kind of honesty this phase is for.

### `requirements.txt`

- Add `scikit-learn` explicitly, with a version floor matching the currently-installed `1.7.2` (exact floor decided at implementation time, matching the project's existing style of pinning e.g. `langgraph>=1.2.0,<2.0.0`).

### `scripts/run_eval.py`

- `MIN_VERIFIED_RATIO = 0.5` (currently its own hardcoded literal with a comment claiming it mirrors the evaluator) is replaced with `from src.evaluator import ReliabilityEvaluator` and a reference to `ReliabilityEvaluator.CITATION_VERIFIED_FLOOR` — one real source of truth instead of two numbers that happen to currently agree.

## Feature naming

The `components` dict (returned from `evaluate()`, logged to `feedback.jsonl`, and consumed by the fitting step) uses the spelled-out key `critic_confidence`, not the internal local-variable name `critic_conf` already used inside `evaluate()` — clearer for a persisted, human-readable log record. The four keys, in the fixed order the fitting step always uses: `semantic`, `verifier`, `critic_confidence`, `citation_verified_ratio`.

## Test changes

- `tests/test_evaluator.py`: a new case proving the citation floor actually blocks acceptance — a high blended score (semantic/verifier/critic all strong) but `citation_verified_ratio` below `CITATION_VERIFIED_FLOOR` must still result in `revise` or `exhausted`, never `accept`. Nothing existing exercises this combination today (every current test's `citation_verified_ratio` already happens to agree with its expected decision).
- New `tests/test_weight_fitting.py`: covers `load_feedback_components` (filters out records missing `components`), `fit_fitted_weights` (below-threshold → inactive; below-per-class-minimum → inactive even with enough total records; a small synthetic dataset above both bars → active with real coefficients), and `FittedWeights.active`/its sigmoid computation.
- `tests/test_orchestrator.py` and `tests/test_run_eval.py`: fixture updates wherever a `DummyStore`/scripted evaluator path needs to account for the new `component_scores` state key and the citation-floor-aware decision logic — exact scope decided at implementation-plan time once the code is in front of the implementer.

## Error handling

`fit_fitted_weights` never raises — any `sklearn` exception (convergence failure, malformed data) is caught and treated as "not enough good data yet," falling back to the inactive `FittedWeights` the same way too-few-samples does. `load_feedback_components` silently skips any `feedback.jsonl` line that's malformed JSON or missing the `components` key, matching `calibration.py`'s existing `load_feedback`'s handling of malformed lines.

## Out of scope

- Fitting `accept_threshold` itself (see Decision above — stays a policy knob).
- A separate fitting script or persisted weights file (see Decision above — live refit only).
- Migrating old `feedback.jsonl` records that predate the `components` field — they're simply skipped, not backfilled.
- Any change to `max_attempts` or the revise-loop's critique-generation logic (`_build_critique`) beyond what's already accurate given the new hard citation floor.
- Retuning `sklearn.linear_model.LogisticRegression`'s hyperparameters (regularization strength, solver) beyond its defaults — reasonable for this data scale, not worth hand-tuning for a demo-scale dataset.
