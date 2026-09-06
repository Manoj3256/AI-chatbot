import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1024,
    reasoning_effort="low",
)

def build_gpt(messages):
    prompt = "<|begin_of_text|>"
    for m in messages:
        prompt += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def generate(messages, max_new_tokens=200, do_sample=True, temperature=1, top_p=0.25):
    prompt = build_gpt(messages)
    return llm.invoke(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p,
    )