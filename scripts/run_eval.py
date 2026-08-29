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
