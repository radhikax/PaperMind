from typing import List, Tuple
import numpy as np
import pickle

try:
    import faiss
except Exception:
    faiss = None


class FaissStore:
    def __init__(self, dim: int):
        if faiss is None:
            raise ImportError("faiss is required to use FaissStore")
        self.dim = dim
        # using inner-product on normalized vectors approximates cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[dict] = []

    def build_index(self, embeddings: np.ndarray, metadatas: List[dict]):
        emb = embeddings.astype('float32')
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self.index.reset()
        self.index.add(emb)
        self.metadatas = list(metadatas)

    def add(self, embeddings: np.ndarray, metadatas: List[dict]):
        emb = embeddings.astype('float32')
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self.index.add(emb)
        self.metadatas.extend(metadatas)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[float, dict]]:
        q = query_emb.astype('float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)
        D, I = self.index.search(q, top_k)
        results: List[Tuple[float, dict]] = []
        for score_row, idx_row in zip(D, I):
            for score, idx in zip(score_row, idx_row):
                if idx < len(self.metadatas) and idx != -1:
                    results.append((float(score), self.metadatas[idx]))
        return results

    def save(self, path: str):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.metadatas, f)

    @classmethod
    def load(cls, path: str):
        if faiss is None:
            raise ImportError("faiss is required to load FaissStore")
        index = faiss.read_index(path + ".index")
        with open(path + ".pkl", "rb") as f:
            metadatas = pickle.load(f)
        inst = cls(index.d)
        inst.index = index
        inst.metadatas = metadatas
        return inst

