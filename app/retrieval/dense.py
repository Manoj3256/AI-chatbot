import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

embedder = None
index = None
texts = None

def load_dense_index():
    global embedder, index, texts
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    index = faiss.read_index("artifacts/faiss.index")
    with open("artifacts/texts.json") as f:
        texts = json.load(f)

def search(query, number_of_results=3):
    query_embed = embedder.encode([query], convert_to_numpy=True)
    distances, similar_item_ids = index.search(np.float32(query_embed), number_of_results)
    texts_np = np.array(texts)
    results = pd.DataFrame(data={'texts': texts_np[similar_item_ids[0]], 'distance': distances[0]})
    return results