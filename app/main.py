from fastapi import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel

from app.retrieval.dense import load_dense_index
from app.retrieval.keyword import load_bm25
from app.vision.image_search import load_image_index
from app.retrieval.rag import rag_answer
from app.agent.orchestrator import run_agent

app = FastAPI(title="Hybrid RAG Agent")

@app.on_event("startup")
def load_artifacts():
    load_dense_index()
    load_bm25()
    load_image_index()

@app.get("/health")
def health():
    return {"status": "ok"}

class RagRequest(BaseModel):
    query: str

class RagResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/rag", response_model=RagResponse)
def rag_endpoint(req: RagRequest):
    answer, chunks = rag_answer(req.query)
    return RagResponse(answer=answer, sources=chunks)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    reply = run_agent(req.session_id, req.message)
    return ChatResponse(reply=reply)

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        reply = run_agent(req.session_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=503, detail="The assistant is temporarily unavailable. Please try again shortly.")
    return ChatResponse(reply=reply)