"""Example: run the orchestrator pipeline end-to-end on a PDF.

Usage:
    python examples/run_langgraph_demo.py path/to/paper.pdf "your question"

This script is intentionally minimal and uses local fallbacks when
LangGraph is not installed.
"""
import sys
from pprint import pprint

from src.ingest import load_and_chunk
from src.embeddings import EmbeddingModel
from src.vectorstore import FaissStore
from src.langgraph_agents import make_orchestrator


def main(pdf_path: str, query: str):
    chunks = load_and_chunk(pdf_path)
    texts = [c.get('text', '') for c in chunks]

    embedder = EmbeddingModel()
    embs = embedder.embed_texts(texts)

    store = FaissStore(dim=embs.shape[1])
    metadatas = [{"text": t, "source": c.get('source', {}), "id": i} for i, (t, c) in enumerate(zip(texts, chunks))]
    store.build_index(embs, metadatas)

    retriever, summarizer, critic = make_orchestrator(store, embedder)

    # Run pipeline
    results = retriever(query, top_k=5) if callable(retriever) else retriever.retrieve(query, top_k=5)
    retrieved_texts = [m.get('text', '') for _, m in results]
    summary_res = summarizer(retrieved_texts) if callable(summarizer) else summarizer.summarize(retrieved_texts)
    # normalize summary_res to dict
    if isinstance(summary_res, dict):
        summary_text = summary_res.get('summary', '')
    else:
        summary_text = str(summary_res)

    assessment = critic(summary_text) if callable(critic) else critic.assess(summary_text)

    print("--- Retrieved ---")
    for score, m in results:
        print(f"{score:.3f}", m.get('source'))

    print("\n--- Summary ---")
    print(summary_text)

    print("\n--- Critic ---")
    pprint(assessment)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python examples/run_langgraph_demo.py path/to/paper.pdf \"your question\"")
        sys.exit(1)
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    main(pdf_path, query)
