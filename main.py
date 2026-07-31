from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app=FastAPI()

class chat_define(BaseModel):
    message: str

@app.get("/")
def serve_chat_page():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat")
def chat(request:chat_define):
    url="http://host.docker.internal:11434/api/chat"
    dic={
        "model": "llama3.2",
        "messages": [
            { "role": "user", "content": request.message}
            ],
        "stream": False
    }
    response=requests.post(url,json=dic)
    reply=response.json()["message"]["content"]
    return {"reply":reply}

@app.get("/health")
def check_running():
    return {"status":"Model is running"}