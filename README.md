# Hybrid RAG + Vision Agent

A hybrid retrieval-augmented chatbot with a ReAct agent that routes between document search, image search, and a calculator — built as a FastAPI service, containerized with Docker.

## Features

- **Hybrid retrieval** — dense (FAISS + sentence-transformers) and keyword (BM25) search, reranked with Cohere
- **Vision search** — CLIP-based image retrieval over a small image set
- **ReAct agent** — routes each query to the right tool (document Q&A, calculator, web search, or image search) using LangChain + Groq
- **Session-isolated memory** — each conversation keeps its own context, safe for multiple concurrent users
- **Gradio UI** — simple chat interface on top of the API

## Tech stack

FastAPI · Docker · LangChain · Groq · Cohere (rerank) · FAISS · sentence-transformers (MiniLM + CLIP) · Gradio

## Architecture

```
User message
    │
    ▼
ReAct Agent (Groq LLM) ──┬──▶ DocumentQA  → hybrid retrieval + rerank → grounded answer
                          ├──▶ Calculator → arithmetic
                          ├──▶ ImageSearch → CLIP + FAISS
                          └──▶ Web Search  → DuckDuckGo (fallback for anything else)
```

## Project structure

```
app/
├── llm_clients.py       # Groq, Cohere, vision API clients
├── retrieval/           # dense (FAISS) + keyword (BM25) search, RAG pipeline
├── vision/              # CLIP image search
├── agent/               # tool definitions + ReAct orchestrator, session memory
├── api/                 # FastAPI routes
└── main.py              # app entrypoint
artifacts/                # pre-built FAISS indexes, BM25 corpus, images
scripts/build_artifacts.py  # one-time script to build the above
gradio_app.py              # UI, calls the FastAPI backend
Dockerfile
requirements.txt
```

## Running locally

1. Clone the repo and install dependencies:
   ```bash
   python -m venv venv
   source venv/Scripts/activate   # Windows Git Bash
   pip install -r requirements.txt
   ```
2. Add a `.env` file with your API keys:
   ```
   GROQ_API_KEY=...
   COHERE_API_KEY=...
   VISION_MODEL_KEY=...
   HF_TOKEN=...
   ```
3. Build the artifacts (only needed once, or after changing the source corpus):
   ```bash
   python scripts/build_artifacts.py
   ```
4. Start the API:
   ```bash
   uvicorn app.main:app --reload
   ```
5. (Optional) Start the UI in a separate terminal:
   ```bash
   python gradio_app.py
   ```

API docs available at `http://localhost:8000/docs`.

## Running with Docker

```bash
docker build -t llm-agent-app .
docker run -p 8000:8000 --env-file .env llm-agent-app
```

## Deployment status

This app is fully containerized and deploy-ready, but is **not currently hosted publicly**. It loads two transformer models (MiniLM + CLIP) at startup, which exceeds the free-tier memory limits on platforms like Render — running it continuously would require a paid instance. The project runs and has been verified end-to-end locally and via Docker; deploying to a paid tier is a deliberate cost decision,not a technical limitation.
