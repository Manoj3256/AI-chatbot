import uuid
import requests
import gradio as gr

API_URL = "http://localhost:8000/chat"
session_id = str(uuid.uuid4())  
def chat_with_agent(message, history):
    response = requests.post(
        API_URL,
        json={"session_id": session_id, "message": message},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["reply"]

demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="Hybrid RAG Agent",
    description="Ask about the Interstellar knowledge base, do a calculation, or ask about anything else.",
)

if __name__ == "__main__":
    demo.launch()