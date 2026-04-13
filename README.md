# 📄 AI PDF Chatbot — RAG with Endee Vector DB
#To see the project ru this: https://ai-pdf-chatbot-ij8vjvqxyiza9jwchtvzjk.streamlit.app/

A beginner-friendly Python web application that lets you **upload any PDF and chat with it** using Retrieval-Augmented Generation (RAG).  
The app uses **OpenAI** for embeddings and chat, and **Endee** as the vector database for blazing-fast semantic search.

---

## 🖼️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                    │
│                                                         │
│  PDF Upload  →  Text Extraction  →  Text Chunking       │
│     (PyMuPDF)      (overlapping windows)                │
│                          ↓                             │
│              OpenAI Embeddings API                      │
│          (text-embedding-3-small, 1536-dim)             │
│                          ↓                             │
│              Endee Vector DB (cosine index)             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                       │
│                                                         │
│  User Question  →  OpenAI Embedding                     │
│                          ↓                             │
│           Endee Similarity Search (top-K)               │
│                          ↓                             │
│         Retrieved Chunks  →  GPT-4o-mini (RAG)          │
│                          ↓                             │
│                     Answer                             │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Runtime |
| Docker | any recent | Run Endee server |
| OpenAI API key | — | Embeddings + LLM |

---

### Step 1 — Start Endee (vector database)

Endee runs as a Docker container. One command is all you need:

```bash
docker run \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```

Verify it is running: open [http://localhost:8080](http://localhost:8080) in your browser — you should see the Endee dashboard.

To stop: `docker stop endee-server`  
To restart: `docker start endee-server`

---

### Step 2 — Clone / download the project

```bash
git clone <repo-url>
cd pdf_chatbot
```

---

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Set up environment variables

Copy the example file and fill in your OpenAI key:

```bash
cp .env.example .env
# Then edit .env and add:  OPENAI_API_KEY=sk-...
```

Alternatively you can enter the key directly in the Streamlit sidebar at runtime.

---

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app opens automatically at [http://localhost:8501](http://localhost:8501).

---

## 📖 How to use

1. **Enter your OpenAI API Key** in the sidebar (if not set in `.env`).
2. **Upload a PDF** using the file uploader on the left panel.
3. Wait for the progress bar to finish (extract → chunk → embed → store).
4. **Ask any question** about the PDF in the chat panel on the right.
5. The AI answers based only on the content of your PDF.

---

## 🗄️ How Endee is used

[Endee](https://endee.io) is a high-performance, open-source vector database capable of handling up to 1 billion vectors on a single node.

In this project Endee serves three roles:

| Role | Details |
|------|---------|
| **Index creation** | A new cosine-similarity index is created per PDF, with dimension matching the OpenAI embedding model (1536). INT8 quantisation is used to reduce memory while keeping high recall. |
| **Vector upsert** | Each text chunk is stored as a vector record with `id`, `vector` (the embedding), and `meta.text` (the raw chunk text). Records are sent in batches of 500. |
| **Similarity search** | At query time the user's question is embedded and passed to Endee's `query()` method. Endee returns the top-K most similar chunks by cosine distance in milliseconds. |

The Endee Python SDK (`pip install endee`) is used for all interactions. No custom HTTP calls required.

---

## 📁 Project Structure

```
pdf_chatbot/
├── app.py              # Streamlit UI + main orchestration
├── pdf_processor.py    # PDF text extraction (PyMuPDF) + chunking
├── embeddings.py       # OpenAI embedding generation (batch)
├── vector_store.py     # Endee integration (create, upsert, search)
├── llm.py              # RAG answer generation (GPT-4o-mini)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## ⚙️ Configuration

All settings can be adjusted in the **Streamlit sidebar** at runtime:

| Setting | Default | Description |
|---------|---------|-------------|
| OpenAI API Key | — | Required for embeddings + LLM |
| Endee Base URL | `http://localhost:8080` | Change if Endee runs elsewhere |
| Endee Auth Token | _(blank)_ | Only needed if Endee started with `NDD_AUTH_TOKEN` |
| Chunk size | 800 chars | Larger = more context per chunk; smaller = more precise retrieval |
| Chunk overlap | 100 chars | Prevents context loss at boundaries |
| Top-K results | 4 | Number of chunks retrieved per query |

---

## 🔧 Troubleshooting

**"Connection refused" when processing PDF**  
→ Ensure Endee is running: `docker ps` should show `endee-server`.

**"Invalid API key" error**  
→ Check your OpenAI API key in the sidebar or `.env` file.

**Empty answers / "I couldn't find relevant content"**  
→ Try increasing the Top-K value in the sidebar, or reduce the chunk size.

**PDF shows no text**  
→ The PDF may be a scanned image. PyMuPDF extracts selectable text only; OCR is not included in this version.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | Endee (Docker) |
| LLM | OpenAI `gpt-4o-mini` |
| Language | Python 3.9+ |

---

## 📜 License

MIT — free to use, modify, and distribute.
