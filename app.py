"""
AI PDF Chatbot (Groq + HuggingFace Embeddings version)
=======================================================
FREE to run: uses Groq API for LLM + local HuggingFace model for embeddings.
"""

import streamlit as st
import os
from pathlib import Path

from pdf_processor import extract_text_from_pdf, split_text_into_chunks
from embeddings import get_embedding, get_embeddings_batch, EMBEDDING_DIM
from vector_store import VectorStore
from llm import generate_answer

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-header { color: #666; margin-bottom: 1.5rem; }
    .chat-user { background: #e8f4fd; border-left: 4px solid #2196F3; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
    .chat-ai { background: #f0f7f0; border-left: 4px solid #4CAF50; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "index_name" not in st.session_state:
    st.session_state.index_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

st.markdown('<div class="main-header">📄 AI PDF Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF and chat with it — powered by Groq (free) + Endee vector DB</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    st.info("🆓 Embeddings are FREE (local model). Only Groq API key needed!", icon="✅")

    groq_key = st.text_input(
        "Groq API Key (free)",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your free key at https://console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.markdown("[Get free Groq API key →](https://console.groq.com)", unsafe_allow_html=False)

    st.markdown("---")
    st.subheader("🗄️ Endee Vector DB")
    endee_url = st.text_input("Endee Base URL", value=os.getenv("ENDEE_BASE_URL", "http://localhost:8080"))
    endee_token = st.text_input("Endee Auth Token (optional)", type="password", value="")
    os.environ["ENDEE_BASE_URL"] = endee_url

    st.markdown("---")
    st.subheader("🔧 Chunking Settings")
    chunk_size = st.slider("Chunk size (chars)", 200, 2000, 800, 100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 100, 50)

    st.markdown("---")
    st.subheader("🔍 Search Settings")
    top_k = st.slider("Top-K results", 1, 10, 4)

    st.markdown("---")
    if st.session_state.pdf_loaded:
        st.success(f"✅ PDF loaded ({st.session_state.chunk_count} chunks)")
        if st.button("🗑️ Clear & Reset"):
            try:
                vs = VectorStore(endee_url, endee_token)
                if st.session_state.index_name:
                    vs.delete_index(st.session_state.index_name)
            except Exception:
                pass
            st.session_state.pdf_loaded = False
            st.session_state.chat_history = []
            st.session_state.index_name = None
            st.session_state.chunk_count = 0
            st.rerun()

# Layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and not st.session_state.pdf_loaded:
        if not groq_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar first.")
        else:
            with st.spinner("Processing PDF…"):
                try:
                    progress = st.progress(0, text="Extracting text from PDF…")
                    raw_text = extract_text_from_pdf(uploaded_file)
                    if not raw_text.strip():
                        st.error("Could not extract text from this PDF.")
                        st.stop()

                    progress.progress(20, text="Splitting into chunks…")
                    chunks = split_text_into_chunks(raw_text, chunk_size, chunk_overlap)
                    if not chunks:
                        st.error("No chunks produced. Try a different chunk size.")
                        st.stop()

                    progress.progress(40, text=f"Generating embeddings for {len(chunks)} chunks (local model)…")
                    embeddings = get_embeddings_batch(chunks)  # no API key needed

                    progress.progress(70, text="Storing vectors in Endee…")
                    safe_name = Path(uploaded_file.name).stem.lower()
                    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in safe_name)[:40]
                    index_name = f"pdf_{safe_name}"

                    vs = VectorStore(endee_url, endee_token)
                    vs.create_or_reset_index(index_name, dimension=EMBEDDING_DIM)
                    vs.upsert_chunks(index_name, chunks, embeddings)
                    progress.progress(100, text="Done!")

                    st.session_state.pdf_loaded = True
                    st.session_state.index_name = index_name
                    st.session_state.chunk_count = len(chunks)
                    st.success(f"✅ Processed **{uploaded_file.name}** — {len(chunks)} chunks stored in Endee.")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if uploaded_file and st.session_state.pdf_loaded:
        with st.expander("📝 Preview extracted text"):
            try:
                preview = extract_text_from_pdf(uploaded_file)[:2000]
                st.text_area("First 2000 characters", preview, height=200, disabled=True)
            except Exception:
                st.info("Preview unavailable.")

    st.markdown("---")
    with st.expander("🏗️ How it works"):
        st.markdown("""
**Pipeline:**
```
PDF Upload
   ↓
Text Extraction (PyMuPDF)
   ↓
Text Splitting (overlapping chunks)
   ↓
HuggingFace Embeddings (FREE, local)
   ↓
Endee Vector DB (cosine similarity)
   ↓  ← at query time
User Question → Embed → Search Endee
   ↓
Top-K Chunks → Groq LLaMA 3 (FREE)
   ↓
Answer
```
        """)

with col_right:
    st.subheader("💬 Chat with your PDF")

    if not st.session_state.pdf_loaded:
        st.info("👈 Upload and process a PDF first to start chatting.")
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 <strong>AI:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about your PDF…", placeholder="e.g. What is the main topic?")
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

        if submitted:
            if not user_question.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("Thinking…"):
                    try:
                        query_embedding = get_embedding(user_question)  # local, no key needed
                        vs = VectorStore(endee_url, endee_token)
                        retrieved_chunks = vs.search(st.session_state.index_name, query_embedding, top_k=top_k)

                        if not retrieved_chunks:
                            answer = "I couldn't find relevant content in the PDF to answer your question."
                        else:
                            answer = generate_answer(
                                question=user_question,
                                context_chunks=retrieved_chunks,
                                api_key=groq_key,
                            )

                        st.session_state.chat_history.append({"role": "user", "content": user_question})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat history"):
                st.session_state.chat_history = []
                st.rerun()
