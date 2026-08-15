import streamlit as st
import tempfile
from pathlib import Path

from src.ingest import load_and_chunk
from src.embeddings import EmbeddingModel
from src.vectorstore import FaissStore
from src.langgraph_agents import make_orchestrator

st.set_page_config(page_title="Research Assistant (demo)")

st.title("Research Assistant — LangGraph demo (scaffold)")

# Mode selector: main QA or Feedback Dashboard
mode = st.sidebar.radio("Mode", ["QA", "Feedback Dashboard"]) 

if mode == "Feedback Dashboard":
    st.header("Feedback Dashboard")
    import os, json
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

            st.subheader("Feedback over time")
            if not df['ts'].isna().all():
                series = df.set_index('ts').resample('D').size()
                st.line_chart(series)

            st.subheader("Recent feedback")
            st.dataframe(df.sort_values(by='ts', ascending=False).head(50))

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
            metadatas = [{"text": c.get('text', ''), "source": c.get('source', {}), "id": i} for i, c in enumerate(chunks)]
            store.build_index(embs, metadatas)
            st.session_state['store'] = store
            st.session_state['embedder'] = embedder
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

            # Create orchestrator (retriever_fn, summarizer_fn, critic_fn)
            retriever_fn, summarizer_fn, critic_fn = make_orchestrator(store, embedder)

            # Retrieve
            results = retriever_fn(query, top_k=5)
            st.header("Retrieved chunks")
            retrieved_texts = []
            for score, meta in results:
                src = meta.get('source', {})
                st.write(f"**Score:** {score:.3f} — page: {src.get('page', 'N/A')}")
                st.write(meta.get('text', '')[:1000])
                retrieved_texts.append(meta.get('text', ''))

            # Summarize
            with st.spinner("Summarizing with LLM..."):
                summary_res = summarizer_fn(retrieved_texts)
            st.header("Summary")
            # summary_res is a dict {summary, citations, valid} per new schema
            if isinstance(summary_res, dict):
                summary_text = summary_res.get('summary', '')
                st.write(summary_text)
                if not summary_res.get('valid', True):
                    st.error("Summarizer did not return valid structured citations. Summary marked as invalid.")
            else:
                summary_text = str(summary_res)
                st.write(summary_text)

            # Critic
            with st.spinner("Assessing summary..."):
                assessment = critic_fn(summary_text)
            st.header("Critic")
            st.json(assessment)

            # Compute enhanced numeric confidence
            from src.evaluator import compute_numeric_confidence
            numeric_score = compute_numeric_confidence(summary_text if not isinstance(summary_res, dict) else summary_res, results, store, embedder, critic_assessment=assessment)
            st.metric(label="Confidence (0-100)", value=numeric_score)

            # Verifier agent output
            from src.agents import VerifierAgent
            verifier = VerifierAgent(store, embedder)
            original_ids = [m.get('id') for _, m in results if m.get('id') is not None]
            verify_res = verifier.verify(summary_text, original_ids)
            st.write("Verifier overlap:", verify_res)

            # Citation enforcement note
            from src.evaluator import citation_present
            has_citation = citation_present(summary_res if isinstance(summary_res, dict) else summary_text, [m for _, m in results])
            if not has_citation:
                st.warning("Summary appears to be missing citations — confidence may be downgraded.")

            # Save last result
            st.session_state['last_query'] = query
            st.session_state['last_summary'] = summary_text
            st.session_state['last_assessment'] = assessment
            st.session_state['last_confidence'] = numeric_score

            # Human-in-the-loop feedback
            st.write("Was this answer accurate?")
            col1, col2 = st.columns(2)
            if col1.button("Accurate"):
                import datetime
                fb = {"query": query, "summary": summary_text, "confidence": numeric_score, "label": "accurate", "timestamp": datetime.datetime.utcnow().isoformat()}
                st.session_state.setdefault('feedback', []).append(fb)
                # append to file
                import json

                with open("feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.success("Feedback saved: accurate")

            if col2.button("Hallucinated"):
                import datetime
                fb = {"query": query, "summary": summary_text, "confidence": numeric_score, "label": "hallucinated", "timestamp": datetime.datetime.utcnow().isoformat()}
                st.session_state.setdefault('feedback', []).append(fb)
                import json

                with open("feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.error("Feedback saved: hallucinated")

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
