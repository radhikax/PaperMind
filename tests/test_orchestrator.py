import numpy as np

from src.langgraph_agents import build_initial_state, make_orchestrator


class DummyStore:
    def __init__(self, score=0.9, ids=(1,)):
        self.score = score
        self.ids = ids

    def search(self, q_emb, top_k=5):
        return [
            (self.score, {"id": i, "source": {"start_page": 1, "end_page": 1}, "text": "source text"})
            for i in self.ids
        ]


class DummyEmbedder:
    def embed_texts(self, texts, batch_size=64, normalize=True):
        return np.ones((len(texts), 8), dtype="float32")


class ScriptedSummarizer:
    """Returns each entry in `responses` in order, one per call to summarize()."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.llm_available = True

    def summarize(self, chunks, critique_feedback=None, previous_summary=None):
        self.calls.append({"critique_feedback": critique_feedback, "previous_summary": previous_summary})
        return self.responses.pop(0)


class FixedCritic:
    def __init__(self, assessment):
        self.assessment = assessment
        self.llm_available = True

    def assess(self, summary_text):
        return self.assessment


def test_graph_accepts_a_high_quality_first_attempt():
    store = DummyStore(score=0.95, ids=(1,))
    embedder = DummyEmbedder()
    summarizer = ScriptedSummarizer(
        [{"summary": "Good summary.", "citations": [{"chunk_id": 1, "page": "1", "excerpt": "source text"}],
          "valid": True}]
    )
    critic = FixedCritic({"confidence": 0.9, "hallucination_rate": 0.05, "notes": "fine"})

    graph = make_orchestrator(store, embedder, summarizer=summarizer, critic=critic)
    final_state = graph.invoke(build_initial_state("what is this paper about?", max_attempts=3))

    assert final_state["reliability_decision"] == "accept"
    assert final_state["attempt"] == 1
    assert len(summarizer.calls) == 1


def test_graph_revises_a_low_quality_summary_then_accepts():
    store = DummyStore(score=0.5, ids=(1,))
    embedder = DummyEmbedder()
    bad = {"summary": "Bad summary.", "citations": [{"chunk_id": 999, "page": "9", "excerpt": "nope"}],
           "valid": True}
    good = {"summary": "Good revised summary.",
            "citations": [{"chunk_id": 1, "page": "1", "excerpt": "source text"}],
            "valid": True}
    summarizer = ScriptedSummarizer([bad, good])
    critic = FixedCritic({"confidence": 0.9, "hallucination_rate": 0.05, "notes": "fine"})

    graph = make_orchestrator(store, embedder, summarizer=summarizer, critic=critic)
    final_state = graph.invoke(build_initial_state("what is this paper about?", max_attempts=3))

    assert final_state["attempt"] == 2
    assert final_state["reliability_decision"] == "accept"
    assert summarizer.calls[1]["critique_feedback"] is not None
    assert "citation" in summarizer.calls[1]["critique_feedback"]


def test_graph_returns_best_effort_with_low_confidence_after_exhausting_retries():
    store = DummyStore(score=0.0, ids=(1,))
    embedder = DummyEmbedder()
    always_bad = {"summary": "Always bad.", "citations": [], "valid": True}
    summarizer = ScriptedSummarizer([always_bad, always_bad, always_bad])
    critic = FixedCritic({"confidence": 0.1, "hallucination_rate": 0.9, "notes": "bad"})

    graph = make_orchestrator(store, embedder, summarizer=summarizer, critic=critic)
    final_state = graph.invoke(build_initial_state("what is this paper about?", max_attempts=3))

    assert final_state["reliability_decision"] == "exhausted"
    assert final_state["attempt"] == 3
    assert len(final_state["history"]) == 3
    assert len(summarizer.calls) == 3
