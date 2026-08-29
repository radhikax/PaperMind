from typing import List, Optional


def semantic_similarity_score(summary: str, store, embedder, top_k: int = 10) -> float:
    """Compute semantic similarity score between summary and retrieved chunks.

    Returns a float in [0,1] (higher is more similar).
    """
    q_emb = embedder.embed_texts([summary])
    results = store.search(q_emb, top_k=top_k)
    if not results:
        return 0.0
    scores = [s for s, _ in results]
    norm_scores = [max(0.0, min(1.0, float(sc))) for sc in scores]
    return float(max(norm_scores))


def verifier_overlap_ratio(summary: str, store, embedder, original_ids: List[int], top_k: int = 10) -> float:
    """Re-query the vector DB with the summary and compute overlap ratio with original ids.

    Returns ratio in [0,1].
    """
    q_emb = embedder.embed_texts([summary])
    results = store.search(q_emb, top_k=top_k)
    retrieved_ids = [m.get('id') for _, m in results if m.get('id') is not None]
    if not original_ids:
        return 0.0
    common = set(original_ids) & set(retrieved_ids)
    return len(common) / len(set(original_ids))


class ReliabilityEvaluator:
    """Aggregates multiple signals into a 0-100 reliability score and a routing decision."""

    SEMANTIC_WEIGHT = 0.40
    VERIFIER_WEIGHT = 0.25
    CRITIC_WEIGHT = 0.20
    CITATION_WEIGHT = 0.15

    def __init__(self, accept_threshold: int = 70, calibrator=None):
        self.accept_threshold = accept_threshold
        self.calibrator = calibrator

    def evaluate(
        self,
        summary_text: str,
        retrieved_chunks: List[dict],
        store,
        embedder,
        critic_assessment: Optional[dict],
        citation_verified_ratio: float,
        attempt: int,
        max_attempts: int,
    ) -> dict:
        original_ids = [c.get('id') for c in retrieved_chunks if c.get('id') is not None]
        semantic = semantic_similarity_score(summary_text, store, embedder, top_k=10)
        verifier = verifier_overlap_ratio(summary_text, store, embedder, original_ids, top_k=10)

        critic_conf = 0.5
        if critic_assessment and isinstance(critic_assessment.get('confidence'), (int, float)):
            critic_conf = float(critic_assessment['confidence'])

        base = (
            self.SEMANTIC_WEIGHT * semantic
            + self.VERIFIER_WEIGHT * verifier
            + self.CRITIC_WEIGHT * critic_conf
            + self.CITATION_WEIGHT * citation_verified_ratio
        )
        raw_score = int(round(max(0.0, min(1.0, base)) * 100))
        score = self.calibrator.apply(raw_score) if self.calibrator else raw_score
        calibration_active = bool(self.calibrator and self.calibrator.active)
        calibration_samples = self.calibrator.n_samples if self.calibrator else 0

        verdict = {
            "score": score,
            "raw_score": raw_score,
            "calibration_active": calibration_active,
            "calibration_samples": calibration_samples,
        }

        if score >= self.accept_threshold:
            return {**verdict, "decision": "accept", "critique_feedback": None}

        if attempt >= max_attempts:
            return {**verdict, "decision": "exhausted", "critique_feedback": None}

        critique_feedback = self._build_critique(semantic, verifier, critic_assessment, citation_verified_ratio)
        return {**verdict, "decision": "revise", "critique_feedback": critique_feedback}

    def _build_critique(
        self,
        semantic: float,
        verifier: float,
        critic_assessment: Optional[dict],
        citation_verified_ratio: float,
    ) -> str:
        issues = []
        if citation_verified_ratio < 0.5:
            issues.append(
                "citations could not be verified against the source text — cite exact "
                "page/chunk ids that appear in the retrieved passages, with excerpts copied verbatim"
            )
        hallucination_rate = (critic_assessment or {}).get('hallucination_rate')
        if isinstance(hallucination_rate, (int, float)) and hallucination_rate > 0.3:
            issues.append(f"hallucination_rate {hallucination_rate:.2f} exceeds 0.3 — remove unsupported claims")
        if verifier < 0.3:
            issues.append("the summary drifts from the retrieved passages — stay closer to the source wording")
        if semantic < 0.3:
            issues.append(
                "the summary is not semantically close to the retrieved passages — reground it in the provided text"
            )
        if not issues:
            issues.append("overall reliability score is below threshold — tighten factual grounding and citations")
        return "; ".join(issues)
