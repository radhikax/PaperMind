from src.evaluator import compute_numeric_confidence


class DummyStore:
    def search(self, q_emb, top_k=10):
        # return fixed high score
        return [(0.95, {"id": 1, "source": {"page": 1}})]


class DummyEmbedder:
    def embed_texts(self, texts, batch_size=64, normalize=True):
        import numpy as np

        return np.ones((len(texts), 8), dtype="float32")


def test_compute_numeric_confidence():
    store = DummyStore()
    embedder = DummyEmbedder()
    retrieved = [(0.95, {"id": 1, "source": {"page": 1}})]
    summary = "This is a test summary referring to page 1."
    score = compute_numeric_confidence(summary, retrieved, store, embedder, critic_assessment={"confidence": 0.9})
    assert isinstance(score, int)
    assert 0 <= score <= 100
