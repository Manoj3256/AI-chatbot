# app/vision/image_search.py
import json
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
clip_model = None
image_index = None
image_names = None

def load_image_index():
    global clip_model, image_index, image_names
    clip_model = SentenceTransformer("clip-ViT-B-32", device="cpu")
    image_index = faiss.read_index("artifacts/image_faiss.index")
    with open("artifacts/image_names.json") as f:
        image_names = json.load(f)

def search_images(query, number_of_results=3):
    query_embed = clip_model.encode([query], convert_to_numpy=True)
    distances, ids = image_index.search(np.float32(query_embed), number_of_results)
    results = []
    for idx in ids[0]:
        name = image_names[idx]
        img = Image.open(f"artifacts/images/{name}.png")
        results.append(img)
    return results

vision_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("VISION_MODEL_KEY"),
)

def generate_vision(messages, max_new_tokens=300, temperature=0.2):
    response = vision_client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        extra_body={"reasoning": {"enabled": False}},
    )
    return response.choices[0].message.content