import src.llm_client as llm_client


def test_get_openai_client_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_client.get_openai_client() is None


def test_get_openai_client_returns_none_when_sdk_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm_client, "OpenAI", None)
    assert llm_client.get_openai_client() is None


def test_get_openai_client_returns_client_when_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = llm_client.get_openai_client()
    assert client is not None
