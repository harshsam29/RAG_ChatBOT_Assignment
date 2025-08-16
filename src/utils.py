# src/utils.py
import re, hashlib

def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def count_words(s: str) -> int:
    return len(re.findall(r"\w+", s))

def join_sentences(sentences, target_words=200, overlap_words=50):
    chunks = []
    cur, cur_count = [], 0
    for sent in sentences:
        w = count_words(sent)
        if cur_count + w <= target_words or not cur:
            cur.append(sent); cur_count += w
        else:
            chunk = " ".join(cur).strip()
            if chunk: chunks.append(chunk)
            # overlap
            overlap, ow = [], 0
            for s in reversed(cur):
                w2 = count_words(s)
                if ow + w2 > overlap_words: break
                overlap.insert(0, s); ow += w2
            cur = overlap + [sent]
            cur_count = count_words(" ".join(cur))
    if cur: chunks.append(" ".join(cur).strip())
    return chunks

def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()[:16]

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
