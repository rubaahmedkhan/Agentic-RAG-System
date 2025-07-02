# embedder.py
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "chunks.pkl"
SCRAPED_PATH = "scraped_pages.pkl"
EMBEDDINGS_PATH = "embeddings.npy"
INDEX_PATH = "index.faiss"

def chunk_text(text, chunk_size=500, overlap=50):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

def create_chunks():
    if os.path.exists(CHUNKS_PATH):
        print("📂 Chunks already saved.")
        return

    with open(SCRAPED_PATH, "rb") as f:
        pages = pickle.load(f)

    all_chunks = []
    for url, txt in pages:
        for chunk in chunk_text(txt):
            all_chunks.append((url, chunk))

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Created and saved {len(all_chunks)} chunks.")

def create_embeddings():
    if os.path.exists(INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        print("📂 Embeddings and FAISS index already exist.")
        return

    with open(CHUNKS_PATH, "rb") as f:
        all_chunks = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk for _, chunk in all_chunks]
    embs = model.encode(texts, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embs)

    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(np.array(embs))
    faiss.write_index(index, INDEX_PATH)

    print("✅ Embeddings and FAISS index saved.")

if __name__ == "__main__":
    create_chunks()
    create_embeddings()
