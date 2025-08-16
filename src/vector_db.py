# src/vector_db.py
import os, pickle
import faiss
import numpy as np
from typing import List, Dict, Tuple
from .log import log

def _normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / norms

class FaissDB:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized inner product
        self.meta: List[Dict] = []
        self._built = False

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        assert embeddings.shape[0] == len(metadatas)
        emb = _normalize(embeddings.astype("float32"))
        self.index.add(emb)
        self.meta.extend(metadatas)
        self._built = True
        log(f"Built FAISS index with {len(self.meta)} chunks.")

    def search(self, query_emb: np.ndarray, k: int = 4):
        q = _normalize(query_emb.reshape(1, -1).astype("float32"))
        D, I = self.index.search(q, k)
        out = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0: continue
            out.append((self.meta[idx], float(score)))
        return out

    def save(self, dirpath="vectordb"):
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "index.faiss"))
        with open(os.path.join(dirpath, "meta.pkl"), "wb") as f:
            pickle.dump({"dim": self.dim, "meta": self.meta}, f)
        log("Saved FAISS index & metadata.")

    @staticmethod
    def load(dirpath="vectordb"):
        with open(os.path.join(dirpath, "meta.pkl"), "rb") as f:
            payload = pickle.load(f)
        db = FaissDB(payload["dim"])
        db.index = faiss.read_index(os.path.join(dirpath, "index.faiss"))
        db.meta = payload["meta"]
        db._built = True
        return db
