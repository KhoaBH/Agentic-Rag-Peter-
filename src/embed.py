import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

EMBED_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

def get_embedding(text: str):
    response = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=text,
    )
    return response.data[0].embedding

def embed_all_chunks():
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    store = []
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk["text"])
        store.append({
            "id": chunk["id"],
            "source": chunk["source"],
            "text": chunk["text"],
            "embedding": vec
        })
        print(f"[{i+1}/{len(chunks)}] embedded {chunk['id']}")
        time.sleep(0.05)  # tiny delay to be nice to the API

    with open("data/vector_store.json", "w", encoding="utf-8") as f:
        json.dump(store, f)
    print(f"\nSaved {len(store)} embeddings to data/vector_store.json")

if __name__ == "__main__":
    embed_all_chunks()