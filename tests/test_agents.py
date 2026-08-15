from src.agents import SummarizerAgent


def test_summarizer_accepts_chunk_metadata():
    agent = SummarizerAgent()
    chunks = [{"text": "This is a test passage.", "source": {"page": 3}, "id": 7}]
    res = agent.summarize(chunks)
    assert isinstance(res, (str, dict))
    if isinstance(res, str):
        assert "[chunk_id:7" in res


def test_summarizer_max_attempts_attr():
    a = SummarizerAgent(max_attempts=5)
    assert hasattr(a, 'max_attempts') and a.max_attempts == 5
