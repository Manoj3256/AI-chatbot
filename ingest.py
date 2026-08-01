import os
import requests
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"

def get_embedding(text: str) -> list[float]:
    response = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]

def load_documents(data_dir: str) -> list[dict]:
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"filename": filename, "text": f.read()})
    return docs

def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece,
                "source": doc["filename"],
                "chunk_id": f"{doc['filename']}_{i}"
            })
    return chunks

def main():
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents.")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print("Embedding and storing chunks...")
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[chunk["chunk_id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}]
        )
        print(f"  Stored: {chunk['chunk_id']}")

    print(f"\nDone. {collection.count()} chunks in collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()