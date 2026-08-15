import tempfile
import time
from pathlib import Path

import streamlit as st

from src.embeddings import EmbeddingModel
from src.ingest import load_and_chunk
from src.langgraph_agents import make_orchestrator
from src.vectorstore import FaissStore

st.set_page_config(page_title="Research Assistant (demo)")

st.title("Research Assistant — LangGraph demo (scaffold)")


def call_with_retries_validate(
    fn,
    *args,
    validator=None,
    max_attempts=3,
    initial_delay=1.0,
    multiplier=2.0,
    on_attempt=None,
    **kwargs,
):
    """Call `fn(*args, **kwargs)` repeatedly until `validator(result)` is True or attempts exhausted.

    - `validator` may be None (treated as always-true).
    - `on_attempt` is an optional callback receiving the attempt number (1-based).
    """
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        if on_attempt:
            try:
                on_attempt(attempt)
            except Exception:
                pass
        try:
            res = fn(*args, **kwargs)
        except Exception:
            if attempt < max_attempts:
                time.sleep(delay)
                delay *= multiplier
                continue
            raise

        if validator is None:
            return res

        try:
            ok = validator(res)
        except Exception:
            ok = False

        if ok:
            return res
        if attempt < max_attempts:
            time.sleep(delay)
            delay *= multiplier
            continue
    return res


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

            # Create orchestrator (retriever_fn, summarizer_fn, critic_fn)
            retriever_fn, summarizer_fn, critic_fn = make_orchestrator(store, embedder)

            # Retrieve
            results = retriever_fn(query, top_k=5)
            st.header("Retrieved chunks")
            retrieved_texts = []
            retrieved_chunks = []
            for score, meta in results:
                src = meta.get('source', {})
                st.write(f"**Score:** {score:.3f} — page: {src.get('page', 'N/A')}")
                st.write(meta.get('text', '')[:1000])
                retrieved_texts.append(meta.get('text', ''))
                retrieved_chunks.append({"text": meta.get('text', ''), "source": src or {}, "id": meta.get('id')})

            # Summarize
            with st.spinner("Summarizing with LLM..."):
                # pass rich chunk metadata to improve citation accuracy
                status_placeholder = st.empty()

                def _show_summarizer_attempt(a):
                    status_placeholder.info(f"Summarizer attempt {a}...")

                summary_res = call_with_retries_validate(
                    summarizer_fn,
                    retrieved_chunks,
                    validator=lambda r: not (isinstance(r, dict) and r.get('valid', True) is False),
                    max_attempts=3,
                    initial_delay=1.0,
                    multiplier=2.0,
                    on_attempt=_show_summarizer_attempt,
                )
                status_placeholder.empty()
            st.header("Summary")
            # summary_res is a dict {summary, citations, valid} per new schema
            if isinstance(summary_res, dict):
                summary_text = summary_res.get('summary', '')
                st.write(summary_text)
                if not summary_res.get('valid', True):
                    st.error("Summarizer did not return valid structured citations. Summary marked as invalid.")
                    st.warning("You must regenerate a valid structured summary before accepting or saving feedback.")
            else:
                summary_text = str(summary_res)
                st.write(summary_text)

            # Critic
            with st.spinner("Assessing summary..."):
                critic_placeholder = st.empty()

                def _show_critic_attempt(a):
                    critic_placeholder.info(f"Critic attempt {a}...")

                assessment = call_with_retries_validate(
                    critic_fn,
                    summary_text,
                    validator=None,
                    max_attempts=2,
                    initial_delay=0.5,
                    multiplier=2.0,
                    on_attempt=_show_critic_attempt,
                )
                critic_placeholder.empty()
            st.header("Critic")
            st.json(assessment)

            # Compute enhanced numeric confidence
            from src.evaluator import compute_numeric_confidence
            numeric_score = compute_numeric_confidence(
                summary_text if not isinstance(summary_res, dict) else summary_res,
                results,
                store,
                embedder,
                critic_assessment=assessment,
            )
            st.metric(label="Confidence (0-100)", value=numeric_score)

            # Verifier agent output
            from src.agents import VerifierAgent
            verifier = VerifierAgent(store, embedder)
            original_ids = [m.get('id') for _, m in results if m.get('id') is not None]
            verifier_placeholder = st.empty()

            def _show_verifier_attempt(a):
                verifier_placeholder.info(f"Verifier attempt {a}...")

            try:
                verify_res = call_with_retries_validate(
                    verifier.verify,
                    summary_text,
                    original_ids,
                    validator=None,
                    max_attempts=2,
                    initial_delay=0.5,
                    multiplier=2.0,
                    on_attempt=_show_verifier_attempt,
                )
            finally:
                verifier_placeholder.empty()
            st.write("Verifier overlap:", verify_res)

            # Citation enforcement note
            from src.evaluator import citation_present
            has_citation = citation_present(
                summary_res if isinstance(summary_res, dict) else summary_text,
                [m for _, m in results],
            )
            if not has_citation:
                st.warning("Summary appears to be missing citations — confidence may be downgraded.")

            # Save last result
            st.session_state['last_query'] = query
            st.session_state['last_summary'] = summary_text
            st.session_state['last_assessment'] = assessment
            st.session_state['last_confidence'] = numeric_score

            # Human-in-the-loop feedback
            # Only allow acceptance if the summarizer returned a valid structured summary
            allow_accept = True
            if isinstance(summary_res, dict) and not summary_res.get('valid', False):
                allow_accept = False

            st.write("Was this answer accurate?")
            col1, col2 = st.columns(2)
            if allow_accept and col1.button("Accurate"):
                import datetime
                fb = {
                    "query": query,
                    "summary": summary_text,
                    "confidence": numeric_score,
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
                    "confidence": numeric_score,
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
                if st.button("Regenerate structured summary"):
                    with st.spinner("Regenerating summary..."):
                        summary_res = summarizer_fn(retrieved_chunks)
                        if isinstance(summary_res, dict):
                            summary_text = summary_res.get('summary', '')
                        else:
                            summary_text = str(summary_res)
                        st.experimental_rerun()

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
