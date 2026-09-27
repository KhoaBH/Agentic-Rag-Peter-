import os
import json

RAW_DIR = "data/raw"
CHUNK_SIZE = 800      # characters per chunk
OVERLAP = 100         # overlap between chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_all():
    all_chunks = []
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".txt"):
            continue
        source = fname.replace(".txt", "")
        with open(os.path.join(RAW_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{source}_{i}",
                "source": source,
                "text": chunk
            })
        print(f"{source}: {len(chunks)} chunks")
    return all_chunks

if __name__ == "__main__":
    chunks = chunk_all()
    os.makedirs("data", exist_ok=True)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nTotal chunks: {len(chunks)}")