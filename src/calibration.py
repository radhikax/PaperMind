"""Calibrates the ReliabilityEvaluator's raw 0-100 score against the
accurate/hallucinated labels users have already been leaving in
`feedback.jsonl`. Without this, the evaluator's blend weights are constants
chosen once and never checked against outcomes: a score of "70" means
whatever the formula says it means, even if 70s have historically turned
out to be hallucinated half the time.

Approach: bucket past (score, label) pairs by score range, compute the
empirical accuracy rate per bucket, and fit a monotonic (isotonic)
curve through those rates via pool-adjacent-violators. A new raw score is
then remapped through that curve. Below `MIN_SAMPLES` feedback records the
calibrator stays inactive and scores pass through unchanged, since a curve
fit on a handful of labels would just be noise.
"""
import json
import os
from typing import List

FEEDBACK_PATH = "feedback.jsonl"
MIN_SAMPLES = 8
BIN_EDGES = [0, 20, 40, 60, 80, 101]


def load_feedback(path: str = FEEDBACK_PATH) -> List[dict]:
    """Read labeled (score, accurate/hallucinated) records from a feedback.jsonl file."""
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
            if isinstance(rec.get("confidence"), (int, float)) and rec.get("label") in (
                "accurate",
                "hallucinated",
            ):
                records.append(rec)
    return records


def _bin_index(score: float) -> int:
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= score < BIN_EDGES[i + 1]:
            return i
    return len(BIN_EDGES) - 2


def isotonic_regression(values: List[float], weights: List[float]) -> List[float]:
    """Pool-adjacent-violators: the closest non-decreasing fit to `values`, weighted least squares."""
    if not values:
        return []
    level_val = list(values)
    level_w = list(weights)
    level_len = [1] * len(values)
    i = 0
    while i < len(level_val) - 1:
        if level_val[i] > level_val[i + 1]:
            merged_w = level_w[i] + level_w[i + 1]
            merged_val = (level_val[i] * level_w[i] + level_val[i + 1] * level_w[i + 1]) / merged_w
            level_val[i:i + 2] = [merged_val]
            level_w[i:i + 2] = [merged_w]
            level_len[i:i + 2] = [level_len[i] + level_len[i + 1]]
            if i > 0:
                i -= 1
        else:
            i += 1
    fitted = []
    for val, length in zip(level_val, level_len):
        fitted.extend([val] * length)
    return fitted


def _interp(x: float, xs: List[float], ys: List[float]) -> float:
    if not xs:
        return x / 100.0
    if len(xs) == 1:
        return ys[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            t = 0.0 if span == 0 else (x - xs[i]) / span
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


class ScoreCalibrator:
    """Maps a raw 0-100 reliability score to a calibrated one using historical accuracy rates."""

    def __init__(self, bin_centers: List[float], fitted_rates: List[float], n_samples: int,
                 min_samples: int = MIN_SAMPLES):
        self.bin_centers = bin_centers
        self.fitted_rates = fitted_rates
        self.n_samples = n_samples
        self.min_samples = min_samples

    @property
    def active(self) -> bool:
        return self.n_samples >= self.min_samples

    def apply(self, raw_score: float) -> int:
        if not self.active:
            return int(round(raw_score))
        calibrated = _interp(raw_score, self.bin_centers, self.fitted_rates) * 100
        return int(round(calibrated))


def fit_calibrator(records: List[dict], min_samples: int = MIN_SAMPLES) -> ScoreCalibrator:
    n = len(records)
    bins: List[List[float]] = [[] for _ in range(len(BIN_EDGES) - 1)]
    for rec in records:
        idx = _bin_index(float(rec["confidence"]))
        bins[idx].append(1.0 if rec["label"] == "accurate" else 0.0)

    centers: List[float] = []
    rates: List[float] = []
    weights: List[float] = []
    for i, values in enumerate(bins):
        if not values:
            continue
        centers.append((BIN_EDGES[i] + min(BIN_EDGES[i + 1], 100)) / 2)
        rates.append(sum(values) / len(values))
        weights.append(float(len(values)))

    fitted = isotonic_regression(rates, weights) if len(centers) >= 2 else rates
    return ScoreCalibrator(centers, fitted, n, min_samples=min_samples)


def load_calibrator(path: str = FEEDBACK_PATH, min_samples: int = MIN_SAMPLES) -> ScoreCalibrator:
    return fit_calibrator(load_feedback(path), min_samples=min_samples)
