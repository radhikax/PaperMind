import re
from typing import List, Tuple


def semantic_similarity_score(summary: str, store, embedder, top_k: int = 10) -> float:
    """Compute semantic similarity score between summary and retrieved chunks.

    Returns a float in [0,1] (higher is more similar).
    """
    import numpy as np

    q_emb = embedder.embed_texts([summary])
    results = store.search(q_emb, top_k=top_k)
    if not results:
        return 0.0
    # results is list of (score, metadata) pairs; scores are inner-product (~cosine)
    scores = [s for s, _ in results]
    # normalize from [-1,1] to [0,1]
    norm_scores = [max(0.0, min(1.0, float(sc))) for sc in scores]
    # use max score as representative
    return float(max(norm_scores))


def citation_present(summary, retrieved_meta: List[dict]) -> bool:
    """Check whether summary contains citations (page numbers or chunk ids).

    Heuristic: look for 'page' or 'p.' followed by digits, or presence of any page number.
    """
    # summary may be a dict returned from summarizer
    if isinstance(summary, dict):
        text = summary.get("summary", "").lower()
    else:
        text = str(summary).lower()
    if re.search(r'page\s*\d+|p\.\s*\d+', text):
        return True
    # check numeric tokens against retrieved pages
    pages = {str(m.get('source', {}).get('page')) for m in retrieved_meta if m.get('source', {}).get('page')}
    tokens = re.findall(r'\d+', summary)
    for t in tokens:
        if t in pages:
            return True
    return False


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


def compute_numeric_confidence(summary: str, retrieved_results: List[Tuple[float, dict]], store, embedder, critic_assessment: dict = None) -> int:
    """Combine semantic similarity, verifier overlap, citation presence, and critic to produce 0-100 score."""
    # prepare metadata
    retrieved_meta = [m for _, m in retrieved_results]
    original_ids = [m.get('id') for m in retrieved_meta if m.get('id') is not None]

    semantic = semantic_similarity_score(summary, store, embedder, top_k=10)
    verifier = verifier_overlap_ratio(summary, store, embedder, original_ids, top_k=10)
    citation = 1.0 if citation_present(summary, retrieved_meta) else 0.0

    critic_conf = 0.5
    if critic_assessment and isinstance(critic_assessment.get('confidence'), (int, float)):
        critic_conf = float(critic_assessment.get('confidence'))

    # Weights (tunable): semantic 45%, verifier 30%, critic 20%, citation bonus 5%
    base = 0.45 * semantic + 0.30 * verifier + 0.20 * critic_conf
    if citation:
        base = base + 0.05 * 1.0

    # Clip and scale to 0-100
    score = max(0.0, min(1.0, base))
    return int(round(score * 100))
