# src/hybrid_search.py
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from .embeddings import embed_text
from .vector_db import FaissDB

def build_bm25(corpus_texts: List[str]):
    tokenized = [t.lower().split() for t in corpus_texts]
    return BM25Okapi(tokenized), tokenized

def bm25_search(bm25: BM25Okapi, tokenized_corpus, query: str, k: int = 8):
    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)
    idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in idx]

def mmr(query_emb: np.ndarray, doc_embs: np.ndarray, lambda_mult=0.5, top_k=6):
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    sim_to_query = np.array([cos(e, query_emb) for e in doc_embs])
    selected, candidates = [], list(range(len(doc_embs)))
    while len(selected) < min(top_k, len(candidates)):
        if not selected:
            idx = int(np.argmax(sim_to_query[candidates]))
            selected.append(candidates.pop(idx))
            continue
        cand_scores = []
        for c in candidates:
            diversity = max([cos(doc_embs[c], doc_embs[s]) for s in selected])
            score = lambda_mult * sim_to_query[c] - (1 - lambda_mult) * diversity
            cand_scores.append((c, score))
        best = sorted(cand_scores, key=lambda x: x[1], reverse=True)[0][0]
        selected.append(best)
        candidates.remove(best)
    return selected
