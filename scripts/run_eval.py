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
