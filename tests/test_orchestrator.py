from src.langgraph_agents import make_orchestrator


class DummyStore:
    def search(self, q_emb, top_k=5):
        return [(0.9, {"text": "Dummy chunk", "source": {"page": 1}})]


class DummyEmbedder:
    def embed_texts(self, texts, batch_size=64, normalize=True):
        # return a dummy vector
        import numpy as np

        return np.ones((len(texts), 8), dtype="float32")


def test_orchestrator_callables():
    store = DummyStore()
    embedder = DummyEmbedder()
    retriever_fn, summarizer_fn, critic_fn = make_orchestrator(store, embedder)

    results = retriever_fn("what is this?", top_k=1)
    assert isinstance(results, list)
    chunks = [m.get("text") for _, m in results]
    summary = summarizer_fn(chunks)
    # summary may be dict or string
    assert isinstance(summary, (str, dict))
    if isinstance(summary, dict):
        summary_text = summary.get('summary', '')
    else:
        summary_text = str(summary)
    assessment = critic_fn(summary_text)
    assert isinstance(assessment, dict)
