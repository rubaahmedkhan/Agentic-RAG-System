# 🤖 Agentic AI & OpenAI SDK RAG System

This project is a **multi-source RAG (Retrieval-Augmented Generation)** system built with:
- 📄 OpenAI Agents SDK Documentation
- 🎥 Transcribed Video on Agentic AI
- 🔍 Dual Tool Agent using **OpenAI Agents SDK**
- 🧠 Powered by Google's **Gemini 1.5 Flash**

It answers questions using the best source (Docs or Video), can generate MCQs, and provide summaries — all through an intelligent Agent.

---

## 🚀 Features

-  Full website scraping and chunking of [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/)
-  ChromaDB storage for video context
-  FAISS for document vector search
-  Tools for answering via:
  - OpenAI SDK documentation
  - Agentic AI YouTube video transcript
-  Combined Agent with intelligent tool selection
-  Gemini Flash as the language model
-  Streamlit-compatible (UI-ready)

---

## 📁 Folder Structure

```
ragwebsite/
│
├── agent.py                  # Main agent logic with dual tools
├── scraper.py               # Website scraper for OpenAI Docs
├── transcribe_and_embed.py  # Transcribes YouTube video and stores in ChromaDB
├── chunks.pkl               # Pickled text chunks from docs
├── embeddings.npy           # FAISS vector embeddings
├── index.faiss              # FAISS index for document search
├── chroma_store/            # Persistent ChromaDB storage
├── .env                     # Contains your Gemini API key
├── ui_app.py (optional)     # Streamlit UI (you can add)
└── README.md                # You're reading it 🙂
```

---

## 🔧 Setup Instructions

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/yourname/agentic-rag.git
cd agentic-rag
UV init
uv add pakages
```

### 2. Add `.env` File

```
GEMINI_API_KEY=your-gemini-key-here
```

---

## 🧠 Step-by-Step Workflow

###  Scrape Documentation

```bash
python scraper.py
```

###  Transcribe YouTube Video & Store to Chroma

```bash
python transcribe_and_embed.py
```

###  Run Agent (via CLI or Streamlit)

```bash
python agent.py
```

---

## 🧠 Agent Instructions (for LLM)

The agent understands:
- If the question is related to OpenAI SDK → use `answer_query`
- If it's about Agentic AI (video) → use `answer_from_video`
- If asked for MCQs or summary → use context + generate accordingly

> “You are an expert on the OpenAI Agents SDK and also have access to a transcribed video about Agentic AI.  
You MUST use the `answer_query` tool for SDK-related questions and `answer_from_video` tool for Agentic AI video.  
Always rely on tools for MCQs, summaries, or answers — never guess.”

---

## ✅ Example Queries to Test

```text
1. What is handoff in the OpenAI SDK?
2. Give a summary of tracing.
3. What is agentic AI?
4. Create MCQs on guardrails from docs.
5. Generate MCQs from the Agentic AI video.
```

---

