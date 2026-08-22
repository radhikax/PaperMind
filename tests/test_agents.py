from src.agents import CitationVerifierAgent, CriticAgent, SummarizerAgent
from src.schemas import Citation, CriticAssessment, SummaryResponse


class FakeMessage:
    def __init__(self, parsed=None, content="", refusal=""):
        self.parsed = parsed
        self.content = content
        self.refusal = refusal


class FakeResponse:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class FakeCompletions:
    def __init__(self, message):
        self._message = message

    def parse(self, **kwargs):
        return FakeResponse(self._message)


class FakeClient:
    def __init__(self, message):
        self.chat = type("Chat", (), {"completions": FakeCompletions(message)})()


class RaisingClient:
    class _RaisingCompletions:
        def parse(self, **kwargs):
            raise RuntimeError("network error")

    def __init__(self):
        self.chat = type("Chat", (), {"completions": self._RaisingCompletions()})()


def test_summarizer_without_client_falls_back_to_heuristic():
    agent = SummarizerAgent(client=None)
    chunks = [{"text": "This is a test passage.", "source": {"page": 3}, "id": 7}]
    res = agent.summarize(chunks)
    assert res["valid"] is False
    assert "[chunk_id:7" in res["summary"]


def test_summarizer_uses_structured_output_when_client_available():
    parsed = SummaryResponse(
        summary="Short summary.",
        citations=[Citation(page=3, chunk_id=7, excerpt="test passage")],
    )
    agent = SummarizerAgent(client=FakeClient(FakeMessage(parsed=parsed)))
    chunks = [{"text": "This is a test passage.", "source": {"page": 3}, "id": 7}]
    res = agent.summarize(chunks)
    assert res["valid"] is True
    assert "Short summary." in res["summary"]
    assert res["citations"][0]["chunk_id"] == 7


def test_summarizer_includes_critique_feedback_in_revision_prompt():
    captured = {}

    class CapturingCompletions:
        def parse(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return FakeResponse(FakeMessage(parsed=SummaryResponse(summary="Revised.", citations=[])))

    client = type("C", (), {"chat": type("Chat", (), {"completions": CapturingCompletions()})()})()
    agent = SummarizerAgent(client=client)
    chunks = [{"text": "Passage.", "source": {"page": 1}, "id": 1}]
    agent.summarize(chunks, critique_feedback="citation missing", previous_summary="Old summary")
    user_msg = captured["messages"][1]["content"]
    assert "citation missing" in user_msg
    assert "Old summary" in user_msg


def test_summarizer_falls_back_when_client_raises():
    agent = SummarizerAgent(client=RaisingClient(), max_attempts=1)
    chunks = [{"text": "Passage text.", "source": {"page": 1}, "id": 1}]
    res = agent.summarize(chunks)
    assert res["valid"] is False


def test_summarizer_max_attempts_attr():
    a = SummarizerAgent(max_attempts=5, client=None)
    assert a.max_attempts == 5


def test_critic_without_client_falls_back_to_heuristic():
    agent = CriticAgent(client=None)
    result = agent.assess("short")
    assert 0.0 <= result["confidence"] <= 1.0
    assert "hallucination_rate" in result


def test_critic_uses_structured_output_when_client_available():
    parsed = CriticAssessment(confidence=0.9, hallucination_rate=0.1, notes="solid")
    agent = CriticAgent(client=FakeClient(FakeMessage(parsed=parsed)))
    result = agent.assess("A well supported summary.")
    assert result == {"confidence": 0.9, "hallucination_rate": 0.1, "notes": "solid"}


def test_critic_falls_back_when_client_raises():
    agent = CriticAgent(client=RaisingClient(), max_attempts=1)
    result = agent.assess("some summary text")
    assert "confidence" in result


def test_citation_verifier_flags_missing_chunk():
    result = CitationVerifierAgent().verify(
        [{"chunk_id": 99, "page": 1, "excerpt": "nope"}],
        [{"id": 1, "text": "Some real text.", "source": {"page": 1}}],
    )
    assert result["verified_ratio"] == 0.0
    assert result["checks"][0]["found_in_chunks"] is False


def test_citation_verifier_matches_excerpt_in_source_text():
    result = CitationVerifierAgent().verify(
        [{"chunk_id": 1, "page": 1, "excerpt": "real text"}],
        [{"id": 1, "text": "Some real text right here.", "source": {"page": 1}}],
    )
    assert result["verified_ratio"] == 1.0
    assert result["checks"][0]["text_match"] is True


def test_citation_verifier_no_citations_returns_zero_ratio():
    result = CitationVerifierAgent().verify([], [{"id": 1, "text": "x", "source": {}}])
    assert result == {"checks": [], "verified_ratio": 0.0}
