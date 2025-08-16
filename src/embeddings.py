# src/embeddings.py
import requests
import numpy as np
from typing import List
from tqdm import tqdm
from .log import log

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"  # change if you use a different local embed model

def embed_text(text: str, model: str = EMBED_MODEL) -> List[float]:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": model, "prompt": text}, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["embedding"]

def embed_batch(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    log("Embedding batches …")
    vecs = []
    for t in tqdm(texts, desc="Batches", unit="it"):
        vecs.append(embed_text(t, model=model))
    arr = np.array(vecs, dtype="float32")
    log(f"Embedding dimension: {arr.shape[1]}")
    return arr
