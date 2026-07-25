from fastapi import FastAPI
from pydantic import BaseModel
import requests

app=FastAPI()
class chat_define(BaseModel):
    message: str

@app.post("/chat")
def chat(request:chat_define):
    url='http://localhost:11434/api/chat'
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

@app.get("/")
def check_running():
    return {"status":"Model is running"}