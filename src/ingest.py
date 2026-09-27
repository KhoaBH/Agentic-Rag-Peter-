import requests
import os

TITLES = [
    "Backpropagation",
    "Gradient descent",
    "Artificial neural network",
    "Geoffrey Hinton",
    "Multilayer perceptron",
    "Yann LeCun",
    "Yoshua Bengio",
    "Deep learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Ilya Sutskever",
    "AlexNet",
    "Turing Award",          # connects Hinton/LeCun/Bengio (2018 co-winners)
    "Universal approximation theorem",
]
OUT_DIR = "data/raw"

HEADERS = {
    "User-Agent": "AgenticRAG-Project/1.0 (student project; contact: you@example.com)"
}

def slugify(title):
    return title.lower().replace(" ", "_") + ".txt"

def fetch_page(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
        "redirects": 1,   # <-- follow redirects
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "")
    if not extract.strip():
        raise ValueError(f"Empty extract for '{title}' (page id: {page.get('pageid')}, may be a disambiguation/redirect issue)")
    return extract

def fetch_and_save():
    os.makedirs(OUT_DIR, exist_ok=True)
    for title in TITLES:
        try:
            text = fetch_page(title)
            path = os.path.join(OUT_DIR, slugify(title))
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"OK: {title} -> {path} ({len(text)} chars)")
        except Exception as e:
            print(f"FAILED: {title} -> {e}")

if __name__ == "__main__":
    fetch_and_save()