import requests
import chromadb

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 3

def get_embedding(text: str) -> list[float]:
    response = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]

def retrieve(query: str, top_k: int = TOP_K):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

def main():
    query = input("Enter a query: ")
    results = retrieve(query)

    print(f"\nTop {TOP_K} results for: '{query}'\n")
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        source = results["metadatas"][0][i]["source"]
        distance = results["distances"][0][i]
        print(f"--- Result {i+1} (source: {source}, distance: {distance:.4f}) ---")
        print(doc)
        print()

if __name__ == "__main__":
    main()