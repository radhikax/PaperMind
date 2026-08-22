import tempfile
from pathlib import Path

import streamlit as st

from src.embeddings import EmbeddingModel
from src.ingest import load_and_chunk
from src.langgraph_agents import build_initial_state, make_orchestrator
from src.vectorstore import FaissStore

st.set_page_config(page_title="Research Assistant (demo)")

st.title("Research Assistant — LangGraph demo (scaffold)")

# Mode selector: main QA or Feedback Dashboard
mode = st.sidebar.radio("Mode", ["QA", "Feedback Dashboard"])

if mode == "Feedback Dashboard":
    st.header("Feedback Dashboard")
    import json
    import os

    import pandas as pd
    if os.path.exists("feedback.jsonl"):
        rows = []
        with open("feedback.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if rows:
            df = pd.DataFrame(rows)
            # ensure timestamp exists
            if 'timestamp' in df.columns:
                df['ts'] = pd.to_datetime(df['timestamp'])
            else:
                df['ts'] = pd.NaT

            st.subheader("Counts")
            counts = df['label'].value_counts().to_dict()
            st.write(counts)

            st.subheader("Label distribution")
            st.bar_chart(pd.Series(counts))

            st.subheader("Feedback over time")
            if not df['ts'].isna().all():
                series = df.set_index('ts').resample('D').size()
                st.line_chart(series)

            st.subheader("Recent feedback")
            st.dataframe(df.sort_values(by='ts', ascending=False).head(50))

            st.subheader("Per-paper breakdown")
            if 'paper_id' in df.columns:
                per_paper = df.groupby('paper_id').size().sort_values(ascending=False)
                st.bar_chart(per_paper)

            csv = df.to_csv(index=False)
            st.download_button("Download feedback CSV", csv, file_name='feedback.csv')
        else:
            st.info("No feedback recorded yet.")
    else:
        st.info("No feedback recorded yet.")

    st.write("\n---\n")

# Continue with QA flow when mode == "QA"
if mode != "Feedback Dashboard":

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is not None:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        t.write(uploaded.getvalue())
        t.flush()
        path = t.name
        # derive a simple paper id from the uploaded filename or temp path
        paper_name = getattr(uploaded, 'name', None) or Path(path).name
        paper_id = f"paper:{paper_name}"
        st.session_state['paper_id'] = paper_id
        st.info("Loading PDF and chunking...")
        chunks = load_and_chunk(path)
        st.success(f"Loaded {len(chunks)} chunks")

        if st.button("Build embeddings & index"):
            with st.spinner("Computing embeddings..."):
                embedder = EmbeddingModel()
                texts = [c.get('text', '') for c in chunks]
                embs = embedder.embed_texts(texts, batch_size=64)
            st.success("Embeddings ready — building FAISS index")
            store = FaissStore(dim=embs.shape[1])
            metadatas = [
                {
                    "text": c.get("text", ""),
                    "source": c.get("source", {}),
                    "id": i,
                }
                for i, c in enumerate(chunks)
            ]
            store.build_index(embs, metadatas)
            st.session_state['store'] = store
            st.session_state['embedder'] = embedder
            st.session_state['paper_id'] = paper_id
            st.success("Index ready")
            # Save index automatically to disk
            try:
                store.save("paper_index")
                st.info("Index saved to 'paper_index.index' and 'paper_index.pkl'")
            except Exception:
                st.warning("Unable to auto-save index to disk")

    if 'store' in st.session_state:
        query = st.text_input("Ask a question about the uploaded paper")
        if query:
            store = st.session_state['store']
            embedder = st.session_state['embedder']

            graph = make_orchestrator(store, embedder)
            initial_state = build_initial_state(query, max_attempts=3)

            status_placeholder = st.empty()
            final_state = initial_state
            for state_update in graph.stream(initial_state, stream_mode="values"):
                final_state = state_update
                status_placeholder.info(f"Running multi-agent pipeline... (attempt {final_state.get('attempt', 0)})")
            status_placeholder.empty()

            st.header("Retrieved chunks")
            for chunk in final_state['retrieved_chunks']:
                src = chunk.get('source', {})
                st.write(f"page: {src.get('page', 'N/A')}")
                st.write(chunk.get('text', '')[:1000])

            if final_state.get('degraded_mode'):
                st.warning(
                    "Running in degraded mode: no OpenAI API key/SDK detected, so summarization "
                    "and scoring are using heuristic fallbacks instead of an LLM."
                )

            summary_res = final_state.get('summary') or {}
            summary_text = summary_res.get('summary', '')
            st.header("Summary")
            st.write(summary_text)
            if not summary_res.get('valid', True):
                st.error("Summarizer did not return valid structured citations. Summary marked as invalid.")

            st.header("Critic")
            st.json(final_state.get('critic_assessment') or {})

            st.header("Citation verification")
            st.json(final_state.get('citation_verification') or {})

            reliability_score = final_state.get('reliability_score')
            st.metric(label="Reliability (0-100)", value=reliability_score)

            decision = final_state.get('reliability_decision')
            if decision == "exhausted":
                st.warning(
                    f"Reliability score stayed below threshold after {final_state.get('attempt')} attempts "
                    "— showing the best attempt. Treat this summary as low-confidence."
                )
            elif decision == "accept" and final_state.get('degraded_mode'):
                st.info(
                    f"Reliability score {reliability_score} met the threshold, but this ran in degraded "
                    "mode (no LLM) — the score reflects text overlap only, not genuine summarization quality."
                )
            elif decision == "accept":
                st.success(f"Reliability check passed after {final_state.get('attempt')} attempt(s).")

            # Save last result
            st.session_state['last_query'] = query
            st.session_state['last_summary'] = summary_text
            st.session_state['last_assessment'] = final_state.get('critic_assessment')
            st.session_state['last_confidence'] = reliability_score

            # Human-in-the-loop feedback
            allow_accept = summary_res.get('valid', False) or decision == "accept"

            st.write("Was this answer accurate?")
            col1, col2 = st.columns(2)
            if allow_accept and col1.button("Accurate"):
                import datetime
                fb = {
                    "query": query,
                    "summary": summary_text,
                    "confidence": reliability_score,
                    "label": "accurate",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "paper_id": st.session_state.get('paper_id'),
                }
                st.session_state.setdefault('feedback', []).append(fb)
                # append to file
                import json

                with open("feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.success("Feedback saved: accurate")

            if allow_accept and col2.button("Hallucinated"):
                import datetime
                fb = {
                    "query": query,
                    "summary": summary_text,
                    "confidence": reliability_score,
                    "label": "hallucinated",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "paper_id": st.session_state.get('paper_id'),
                }
                st.session_state.setdefault('feedback', []).append(fb)
                import json

                with open("feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.error("Feedback saved: hallucinated")
            if not allow_accept:
                st.info("Summary was not structurally valid — ask the question again to retry.")

        # Index persistence controls
        st.sidebar.header("Index storage")
        if st.sidebar.button("Load saved index"):
            try:
                store = FaissStore.load("paper_index")
                st.session_state['store'] = store
                st.success("Loaded index from 'paper_index'")
            except Exception as e:
                st.error(f"Failed to load index: {e}")

        if st.sidebar.button("Save current index"):
            try:
                st.session_state['store'].save("paper_index")
                st.success("Saved index to 'paper_index'")
            except Exception as e:
                st.error(f"Failed to save index: {e}")
