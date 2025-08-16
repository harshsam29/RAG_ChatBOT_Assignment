# app.py
import os, json, time, streamlit as st
import pandas as pd
from src.document_processor import process_and_save, load_all_chunks, SUPPORTED
from src.rag_pipeline import build_or_load_db, rag_stream_answer, DEFAULT_LLM
from src.utils import count_words, safe_name
from src.embeddings import EMBED_MODEL

st.set_page_config(page_title="RAG Chatbot - Amlgo Labs", page_icon="🤖", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }
.system-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
               border-radius: 14px; padding: 16px 14px; }
.metric-head { color:#9CA3AF; font-size:0.85rem; }
.metric-val { font-weight:700; font-size:1.25rem; }
.msg-user { background:#1F2937; padding:14px 16px; border-radius:12px; border:1px solid #253041; }
.msg-assistant { background:#0F172A; padding:14px 16px; border-radius:12px; border:1px solid #1a2540; }
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
colL, colR = st.columns([1.1, 2.2], gap="large")
with colR:
    st.markdown("## 🤖 **RAG Chatbot - Amlgo Labs**")

# ---------- Session ----------
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("db_ready", False)
ss.setdefault("db", None)
ss.setdefault("chunks", [])
ss.setdefault("history_df", pd.DataFrame(columns=["role","content","time"]))

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded = st.file_uploader("Drag & drop (TXT, PDF, DOCX)",
                                accept_multiple_files=True,
                                type=[e.strip(".") for e in SUPPORTED])
    st.divider()
    st.header("⚙️ Settings")
    llm = st.selectbox("Model", [DEFAULT_LLM, "llama3.1:8b-instruct", "zephyr"], index=0)
    k = st.slider("Context chunks (k)", 2, 8, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    auto_tune = st.checkbox("Auto-tune k & temperature", value=True)
    rewrite = st.checkbox("Rewrite follow-ups", value=True)
    mmr_lambda = st.slider("MMR diversity λ", 0.1, 0.9, 0.5, 0.1)
    st.divider()
    colA, colB = st.columns(2)
    with colA:
        reset_chat = st.button("🔄 Reset chat", use_container_width=True)
    with colB:
        clear_db = st.button("🧹 Clear vectordb", use_container_width=True)

if reset_chat:
    ss["messages"] = []
    ss["history_df"] = pd.DataFrame(columns=["role","content","time"])
    st.rerun()

if clear_db:
    import shutil
    if os.path.isdir("vectordb"): shutil.rmtree("vectordb")
    ss["db"] = None; ss["db_ready"] = False
    st.toast("Vector DB cleared. Rebuild required after upload.", icon="🧹")

# ---------- Left system card ----------
with colL:
    with st.container():
        st.markdown('<div class="system-card">', unsafe_allow_html=True)
        st.markdown("### 🧩 System Information")
        chunks_now = len(ss["chunks"]) or len(load_all_chunks("chunks"))
        status = "Ready ✅" if ss.get("db_ready") else "Not Ready ⚠️"
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-head">Model</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-val">{llm}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-head">Embedding</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-val">{EMBED_MODEL}</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="metric-head">Chunks</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-val">{chunks_now}</div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-head">Status</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-val">{status}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Upload -> chunk ----------
if uploaded:
    os.makedirs("data", exist_ok=True)
    saved_paths = []
    for f in uploaded:
        path = os.path.join("data", safe_name(f.name))
        with open(path, "wb") as w: w.write(f.read())
        saved_paths.append(path)
    with st.spinner("Processing document…"):
        chunks = process_and_save(saved_paths, chunks_dir="chunks", target_words=200, overlap_words=50)
        if chunks:
            existing = {c["id"] for c in ss["chunks"]}
            for c in chunks:
                if c["id"] not in existing:
                    ss["chunks"].append(c)
        st.success(f"Created {len(chunks)} chunks ✅")

# ---------- Build/load FAISS once ----------
if not ss["db_ready"]:
    docs = ss["chunks"] or load_all_chunks("chunks")
    if docs:
        with st.spinner("Building FAISS index…"):
            from src.embeddings import EMBED_MODEL as EMB
            ss["db"] = build_or_load_db(docs, db_dir="vectordb", embed_model=EMB)
            ss["db_ready"] = True
            st.toast("Vector DB ready ✅", icon="✅")

# ---------- Welcome banners ----------
if not ss["db_ready"]:
    st.warning("Please upload and process a document to start chatting!")
    st.info("Use the sidebar to upload a document file")
    st.markdown("### 🎯 What this chatbot can do:")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("- Process large documents\n- Extract key information\n- Understand context")
    with cols[1]:
        st.markdown("- Semantic similarity search\n- Find relevant passages\n- Rank by relevance")
    with cols[2]:
        st.markdown("- Stream responses in real-time\n- Provide source citations\n- Answer follow-up questions")

# ---------- Render chat so far ----------
for m in ss["messages"]:
    with st.chat_message(m["role"]):
        klass = "msg-user" if m["role"]=="user" else "msg-assistant"
        st.markdown(f'<div class="{klass}">{m["content"]}</div>', unsafe_allow_html=True)
        # Display sources only for assistant messages
        if m["role"] == "assistant" and "sources" in m:
            with st.expander("📚 Source Documents", expanded=False):
                for h in m["sources"]:
                    st.markdown(f"- **`{h['source']}#{h['chunk_id']}`** — {h['text'][:220].strip()}…")

def add_history(role, content, sources=None):
    msg = {"role": role, "content": content}
    if sources:
        msg["sources"] = sources
    ss["messages"].append(msg)
    ss["history_df"].loc[len(ss["history_df"])] = [role, content, time.strftime("%H:%M:%S")]

# ---------- Input & stream ----------
query = st.chat_input("Ask a question about the document…")
if query:
    if not ss.get("db_ready") or not ss.get("db"):
        st.warning("Please upload documents first.")
    else:
        add_history("user", query)
        with st.chat_message("user"):
            st.markdown(f'<div class="msg-user">🟥 {query}</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            spot = st.empty()
            buf = []
            hits_once = None
            used_k = k; used_temp = temperature
            for token, hits, used_k, used_temp in rag_stream_answer(
                ss["db"], ss["chunks"] or load_all_chunks("chunks"),
                query, k=k, llm=llm, temperature=temperature,
                auto_tune=auto_tune, rewrite=rewrite, mmr_lambda=mmr_lambda
            ):
                buf.append(token)
                spot.markdown(f'<div class="msg-assistant">🟨 {"".join(buf)}</div>', unsafe_allow_html=True)
                if hits_once is None:
                    hits_once = hits
            if hits_once:
                with st.expander("📚 Source Documents", expanded=False):
                    for h in hits_once:
                        st.markdown(f"- **`{h['source']}#{h['chunk_id']}`** — {h['text'][:220].strip()}…")
        add_history("assistant", "".join(buf), sources=hits_once)