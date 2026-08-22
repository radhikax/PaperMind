from src.evaluator import (ReliabilityEvaluator, semantic_similarity_score,
                           verifier_overlap_ratio)


class DummyStore:
    def __init__(self, score=0.95, ids=(1,)):
        self.score = score
        self.ids = ids

    def search(self, q_emb, top_k=10):
        return [(self.score, {"id": i, "source": {"page": 1}}) for i in self.ids]


class EmptyStore:
    def search(self, q_emb, top_k=10):
        return []


class DummyEmbedder:
    def embed_texts(self, texts, batch_size=64, normalize=True):
        import numpy as np

        return np.ones((len(texts), 8), dtype="float32")


def test_semantic_similarity_score_uses_max_result_score():
    assert semantic_similarity_score("a summary", DummyStore(score=0.8), DummyEmbedder()) == 0.8


def test_semantic_similarity_score_empty_results_is_zero():
    assert semantic_similarity_score("x", EmptyStore(), DummyEmbedder()) == 0.0


def test_verifier_overlap_ratio_full_overlap():
    assert verifier_overlap_ratio("a summary", DummyStore(ids=(1, 2)), DummyEmbedder(), original_ids=[1, 2]) == 1.0


def test_verifier_overlap_ratio_no_original_ids_is_zero():
    assert verifier_overlap_ratio("x", DummyStore(ids=(1,)), DummyEmbedder(), original_ids=[]) == 0.0


def test_reliability_evaluator_accepts_high_score():
    evaluator = ReliabilityEvaluator(accept_threshold=70)
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
    assert result["decision"] == "accept"
    assert result["score"] >= 70
    assert result["critique_feedback"] is None


def test_reliability_evaluator_revises_low_score_with_retries_left():
    evaluator = ReliabilityEvaluator(accept_threshold=70)
    result = evaluator.evaluate(
        "a summary",
        retrieved_chunks=[{"id": 99}],
        store=DummyStore(score=0.0, ids=(1,)),
        embedder=DummyEmbedder(),
        critic_assessment={"confidence": 0.1, "hallucination_rate": 0.8},
        citation_verified_ratio=0.0,
        attempt=1,
        max_attempts=3,
    )
    assert result["decision"] == "revise"
    assert result["critique_feedback"] is not None
    assert "hallucination_rate" in result["critique_feedback"]


def test_reliability_evaluator_exhausted_after_max_attempts():
    evaluator = ReliabilityEvaluator(accept_threshold=70)
    result = evaluator.evaluate(
        "a summary",
        retrieved_chunks=[{"id": 99}],
        store=DummyStore(score=0.0, ids=(1,)),
        embedder=DummyEmbedder(),
        critic_assessment={"confidence": 0.1, "hallucination_rate": 0.8},
        citation_verified_ratio=0.0,
        attempt=3,
        max_attempts=3,
    )
    assert result["decision"] == "exhausted"
    assert result["critique_feedback"] is None
