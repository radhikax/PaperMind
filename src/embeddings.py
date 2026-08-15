from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for embeddings")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Embed a list of texts with batching and optional L2-normalization.

        Returns a float32 numpy array of shape (len(texts), dim).
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype='float32')

        embs_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embs = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embs_list.append(batch_embs)

        embs = np.vstack(embs_list).astype('float32')

        if normalize and embs.size > 0:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms

        return embs

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
