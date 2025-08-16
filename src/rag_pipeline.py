# src/rag_pipeline.py
from typing import List, Dict, Generator, Tuple
import numpy as np
from .embeddings import embed_text, embed_batch
from .vector_db import FaissDB
from .ollama_client import stream_chat
from .hybrid_search import build_bm25, bm25_search, mmr
from .query_rewriter import rewrite_query

DEFAULT_LLM = "mistral:instruct"

PROMPT_TEMPLATE = """You are a helpful assistant that answers strictly based on the provided context.
If the answer is not contained in the context, say you don't know.

# Question
{question}

# Context
{context}

# Output style (very important)
- If multiple points apply, return a **numbered list** (1., 2., 3., …).
- Keep sentences crisp and factual.
- Do not include inline source citations in the response.
"""

def build_or_load_db(chunks: List[Dict], db_dir="vectordb", embed_model="nomic-embed-text") -> FaissDB:
    import os
    idx_path = os.path.join(db_dir, "index.faiss")
    meta_path = os.path.join(db_dir, "meta.pkl")
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        return FaissDB.load(db_dir)

    texts = [c["text"] for c in chunks]
    X = embed_batch(texts, model=embed_model)
    db = FaissDB(dim=X.shape[1])
    db.add(X, chunks)
    db.save(db_dir)
    return db

def format_context(hits: List[Dict]) -> str:
    parts = []
    for h in hits:
        tag = f"[source:{h['source']}#{h['chunk_id']}]"
        parts.append(f"{tag}\n{h['text']}")
    return "\n\n---\n\n".join(parts)

def auto_params(query: str, base_k: int = 4, base_temp: float = 0.2) -> Tuple[int, float]:
    qlen = len(query.split())
    k = base_k + (1 if qlen > 18 else 0) + (1 if any(w in query.lower() for w in ["explain","compare","steps","detail"]) else 0)
    k = max(3, min(8, k))
    t = base_temp + (0.1 if any(w in query.lower() for w in ["brainstorm","ideas"]) else 0.0)
    t = max(0.0, min(0.6, t))
    return k, t

def retrieve(db: FaissDB, chunks: List[Dict], query: str, k: int, mmr_lambda=0.5, embed_model="nomic-embed-text"):
    texts = [c["text"] for c in chunks]
    bm25, toks = build_bm25(texts)
    bm_hits = bm25_search(bm25, toks, query, k=max(k,6))
    qv = np.array(embed_text(query, model=embed_model), dtype="float32")
    vec_results = db.search(qv, k=max(k,6))
    pool_idx = []
    for i, _ in bm_hits:
        pool_idx.append(i)
    for m, _ in vec_results:
        try:
            i = chunks.index(m)
            if i not in pool_idx:
                pool_idx.append(i)
        except ValueError:
            continue
    cand_texts = [chunks[i]["text"] for i in pool_idx]
    if not cand_texts:
        return []
    cand_embs = embed_batch(cand_texts, model=embed_model)
    selected_local = mmr(qv, cand_embs, lambda_mult=mmr_lambda, top_k=k)
    selected_idx = [pool_idx[j] for j in selected_local]
    hits = [chunks[i] for i in selected_idx]
    return hits

def rag_stream_answer(db: FaissDB, all_chunks: List[Dict], query: str,
                      k: int = 4, llm=DEFAULT_LLM, temperature=0.2,
                      auto_tune=False, rewrite=False, mmr_lambda=0.5) -> Generator:
    _k, _temp = (k, temperature)
    if auto_tune:
        _k, _temp = auto_params(query, base_k=k, base_temp=temperature)
    q_final = rewrite_query([], query, model=llm) if rewrite else query

    hits = retrieve(db, all_chunks, q_final, k=_k, mmr_lambda=mmr_lambda)
    context = format_context(hits)
    prompt = PROMPT_TEMPLATE.format(question=q_final, context=context)

    messages = [
        {"role": "system", "content": "You are a precise RAG assistant that provides answers without inline citations."},
        {"role": "user", "content": prompt},
    ]
    opts = {"temperature": _temp}
    for token in stream_chat(model=llm, messages=messages, options=opts):
        yield token, hits, _k, _temp