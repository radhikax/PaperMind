# Research Assistant (LangGraph demo)

Minimal multi-agent RAG demo for ingesting PDFs, building a FAISS index, and running a simple Streamlit UI.

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

OpenAI configuration

1. Set `OPENAI_API_KEY` in your environment to enable LLM summarization/critique.

PowerShell:
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Or create a `.env` file in the project root with:

```
OPENAI_API_KEY=sk-...
```

Tests

Run tests with `pytest`:

```bash
pytest -q
```

Files
- `app.py`: Streamlit frontend
- `src/ingest.py`: PDF loading + chunking
- `src/embeddings.py`: embedding wrapper (sentence-transformers)
- `src/vectorstore.py`: FAISS index helper
- `src/agents.py`: Retriever, Summarizer, Critic, and CitationVerifier agents
- `src/evaluator.py`: ReliabilityEvaluator — aggregates agent signals into a score and a revise/accept/exhausted decision
- `src/langgraph_agents.py`: the LangGraph StateGraph wiring the agents into a reliability-gated revision loop

The retriever, summarizer, critic, citation-verifier, and reliability-evaluator agents run as a real LangGraph `StateGraph` (see `src/langgraph_agents.py`). If the reliability evaluator scores a summary below its threshold, it sends the summary back to the summarizer with specific critique feedback for revision, up to `max_attempts` times; after that it returns the best attempt flagged low-confidence rather than failing outright.
