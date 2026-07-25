# from fastapi import Fastapi
import requests
url='http://localhost:11434/api/chat'
dic={
    "model": "llama3.2",
    "messages": [
        { "role": "user", "content": "Say hello in one sentence." }
        ],
    "stream": False
}
response = requests.post(url, json=dic)
print(response.json())
reply=response.json()["message"]["content"]
print(reply)