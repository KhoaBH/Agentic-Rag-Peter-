import json
import numpy as np
from src.embed import get_embedding

STORE_PATH = "data/vector_store.json"

def load_store():
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, k=5, store=None):
    if store is None:
        store = load_store()

    query_vec = get_embedding(query)

    scored = []
    for item in store:
        sim = cosine_sim(query_vec, item["embedding"])
        scored.append((sim, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = scored[:k]

    return [
        {"score": float(score), "source": item["source"], "text": item["text"], "id": item["id"]}
        for score, item in top_k
    ]

if __name__ == "__main__":
    results = retrieve("Who developed backpropagation?", k=3)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['source']}) {r['text'][:150]}...")
        print()