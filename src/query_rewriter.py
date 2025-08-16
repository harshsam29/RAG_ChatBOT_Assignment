# src/query_rewriter.py
from .ollama_client import stream_chat

REWRITE_SYS = "You rewrite follow-up questions into a single, standalone question with any missing context filled in. Only rewrite — do not answer."

TEMPLATE = """Conversation (most recent first):
{history}

User question:
{question}

Rewrite this into a standalone question suitable for retrieval. Keep it concise."""

def rewrite_query(history_pairs, question, model="mistral:instruct"):
    history_txt = ""
    for role, msg in history_pairs[-6:]:
        history_txt += f"{role.upper()}: {msg}\n"
    prompt = TEMPLATE.format(history=history_txt, question=question)
    messages = [{"role":"system","content":REWRITE_SYS}, {"role":"user","content":prompt}]
    out = []
    for tok in stream_chat(model, messages):
        out.append(tok)
    return "".join(out).strip()
