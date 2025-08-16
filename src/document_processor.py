# src/document_processor.py
import os, json, hashlib
from typing import List, Dict
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from docx import Document
from .utils import clean_text, join_sentences, file_sha, safe_name
from .log import log

SUPPORTED = {".txt", ".pdf", ".docx"}

def read_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    if ext == ".docx":
        doc = Document(path)
        paras = [p.text for p in doc.paragraphs]
        return "\n".join(paras)
    raise ValueError(f"Unsupported file type: {ext}")

def chunk_document(text: str, source_name: str,
                   target_words=200, overlap_words=50) -> List[Dict]:
    text = clean_text(text)
    sents = sent_tokenize(text)
    chunks = join_sentences(sents, target_words=target_words, overlap_words=overlap_words)
    result = []
    for i, ch in enumerate(chunks):
        uid = hashlib.md5(f"{source_name}::{i}::{len(ch)}".encode()).hexdigest()[:12]
        result.append({"id": uid, "text": ch, "source": source_name, "chunk_id": i})
    return result

def process_and_save(files: List[str], chunks_dir="chunks",
                     target_words=200, overlap_words=50) -> List[Dict]:
    os.makedirs(chunks_dir, exist_ok=True)
    all_chunks = []
    total_new = 0
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED: 
            continue
        base = os.path.basename(path)
        name = safe_name(base)
        sig = file_sha(path)
        manifest_path = os.path.join(chunks_dir, f"{name}.meta.json")
        out_path = os.path.join(chunks_dir, f"{name}.jsonl")

        if os.path.exists(manifest_path):
            meta = json.load(open(manifest_path, "r", encoding="utf-8"))
            if meta.get("sha") == sig and os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    for line in f:
                        all_chunks.append(json.loads(line))
                log(f"Cached chunks loaded for {name} ({meta.get('num_chunks',0)} chunks).")
                continue

        log(f"Reading {name} …")
        raw = read_text(path)
        log("Tokenizing and chunking …")
        doc_chunks = chunk_document(raw, name, target_words, overlap_words)

        with open(out_path, "w", encoding="utf-8") as f:
            for row in doc_chunks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        json.dump({"sha": sig, "num_chunks": len(doc_chunks)}, open(manifest_path, "w", encoding="utf-8"))

        all_chunks.extend(doc_chunks)
        total_new += len(doc_chunks)
        log(f"Created {len(doc_chunks)} chunks for {name}.")

    if total_new:
        log(f"Created {total_new} chunks in total.")
    else:
        log("No new chunks created (everything cached).")
    return all_chunks

def load_all_chunks(chunks_dir="chunks") -> List[Dict]:
    res = []
    if not os.path.isdir(chunks_dir): return res
    for fn in os.listdir(chunks_dir):
        if not fn.endswith(".jsonl"): continue
        with open(os.path.join(chunks_dir, fn), "r", encoding="utf-8") as f:
            for line in f:
                res.append(json.loads(line))
    return res
