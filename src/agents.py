from typing import List, Dict

try:
    import openai
except Exception:
    openai = None

# Load .env automatically if python-dotenv is available so OPENAI_API_KEY
# can be provided via a .env file in the project root.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class RetrieverAgent:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embedder.embed_texts([query])
        results = self.store.search(q_emb, top_k=top_k)
        return results


class VerifierAgent:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def verify(self, summary: str, original_ids: list, top_k: int = 10) -> dict:
        """Re-query the store with the summary and return overlap metrics."""
        q_emb = self.embedder.embed_texts([summary])
        results = self.store.search(q_emb, top_k=top_k)
        retrieved_ids = [m.get('id') for _, m in results if m.get('id') is not None]
        common = set(original_ids) & set(retrieved_ids)
        overlap = len(common) / max(1, len(set(original_ids))) if original_ids else 0.0
        return {"overlap": round(float(overlap), 3), "retrieved_ids": retrieved_ids}


class SummarizerAgent:
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.model = llm_model

    def summarize(self, chunks: List[str]) -> str:
        joined = "\n\n".join(chunks)
        # If OpenAI is available, call it for a concise summary.
        if openai is not None:
            # Request structured JSON output containing `summary` and `citations`.
            system = "You are a helpful assistant that summarizes scientific paper snippets. Return a JSON object with keys: summary (string), citations (list of {page:int, chunk_id:int, excerpt:string}). If no exact page is available, omit the citation."
            user = (
                f"Summarize the following passages from a paper, preserving factual statements. "
                f"Return EXACTLY a JSON object. Passages:\n\n{joined}"
            )

            from src.schemas import SummaryResponse, Citation
            import json

            for attempt in range(2):
                try:
                    resp = openai.ChatCompletion.create(model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.0)
                    text = resp.choices[0].message.content.strip()
                    try:
                        data = json.loads(text)
                        # Validate against pydantic model
                        try:
                            parsed = SummaryResponse(**data)
                            # build inline citation markers for display
                            citation_strs = []
                            for c in parsed.citations:
                                if c.page and c.chunk_id is not None:
                                    citation_strs.append(f"[p.{c.page}#id:{c.chunk_id}]")
                            summary_text = parsed.summary
                            if citation_strs:
                                summary_text = summary_text + "\n\nCitations: " + ", ".join(citation_strs)
                            return {"summary": summary_text, "citations": [c.dict() for c in parsed.citations], "valid": True}
                        except Exception:
                            # validation failed
                            if attempt == 0:
                                user = (
                                    "The previous response did not match the required JSON schema. Please RETURN EXACT JSON matching {summary: str, citations: [{page:int, chunk_id:int, excerpt:str}]}. Reply ONLY with JSON."
                                )
                                continue
                            else:
                                # give up: return invalid flag with raw text
                                return {"summary": text, "citations": [], "valid": False}
                    except json.JSONDecodeError:
                        if attempt == 0:
                            user = (
                                "Your response must be JSON with keys 'summary' and 'citations'. Reply ONLY with the JSON."
                            )
                            continue
                        else:
                            return {"summary": text, "citations": [], "valid": False}
                except Exception:
                    break

        # Fallback: simple join+truncate
        return (joined[:1500].strip() + ("..." if len(joined) > 1500 else ""))


class CriticAgent:
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.model = llm_model

    def assess(self, summary: str) -> Dict:
        # If OpenAI available, ask it to rate confidence and hallucination
        if openai is not None:
            prompt = [
                {"role": "system", "content": "You are an expert reviewer who detects hallucinations in summaries of scientific text. Reply JSON with keys: confidence (0-1), hallucination_rate (0-1), notes."},
                {"role": "user", "content": f"Assess the following summary for hallucinations and assign a confidence score:\n\n{summary}"},
            ]
            try:
                resp = openai.ChatCompletion.create(model=self.model, messages=prompt, temperature=0.0)
                text = resp.choices[0].message.content.strip()
                # Try to parse a simple JSON blob from the model; if parsing fails, fall back.
                import json

                try:
                    data = json.loads(text)
                    return data
                except Exception:
                    return {"confidence": 0.5, "notes": text}
            except Exception:
                pass

        # Fallback heuristic based on length
        length = len(summary)
        confidence = min(0.95, max(0.1, length / 2000.0))
        return {"confidence": round(confidence, 2), "notes": "Heuristic fallback: confidence based on summary length."}

