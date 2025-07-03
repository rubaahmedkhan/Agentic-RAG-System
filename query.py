# query.py
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

CHUNKS_PATH = "chunks.pkl"
EMBEDDINGS_PATH = "embeddings.npy"
INDEX_PATH = "index.faiss"

# Load all
with open(CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)

embs = np.load(EMBEDDINGS_PATH)
index = faiss.read_index(INDEX_PATH)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini Setup
genai.configure(api_key="API-KEY")
llm = genai.GenerativeModel("gemini-2.5-flash")

# Search
def search_docs(query, k=5):
    qv = model.encode([query])
    D, I = index.search(np.array(qv), k)
    return [(all_chunks[i][0], all_chunks[i][1]) for i in I[0]]

# Ask Gemini
def answer_query(query):
    docs = search_docs(query)
    context = "\n\n".join(f"From {url}:\n{chunk}" for url, chunk in docs)
    prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
    res = llm.generate_content(prompt)
    return res.text

if __name__ == "__main__":
    query = "what is handoff and guardrils"
    answer = answer_query(query)

    print("🔍 Question:", query)
    print("\n🧠 Gemini's Answer:\n", answer)
