from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import os

# === Step 1: Transcribe video ===

video_path = r"D:\RAG\ragwebsite\downloaded_video.mp4"  # ✅ Use raw string

if not os.path.exists(video_path):
    print("❌ File not found!")
    exit()

# Load Whisper (fast CPU inference)
print("📢 Transcribing video...")
model = WhisperModel("base", compute_type="int8")
segments, info = model.transcribe(video_path, language="en")

# Combine segments into full text
text = ""
for segment in segments:
    text += segment.text + " "

print("✅ Transcript ready")

# === Step 2: Chunk the transcript ===

def chunk_text(text, chunk_size=500, overlap=50):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

chunks = chunk_text(text)
print(f"🧩 Total chunks: {len(chunks)}")

# === Step 3: Generate embeddings ===

print("🧠 Generating embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, show_progress_bar=True)

# === Step 4: Store in ChromaDB ===

print("💾 Saving to ChromaDB...")
client = PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("video_docs")

# ✅ Fix: safely delete existing entries with same IDs
collection.delete(ids=[f"video_chunk_{i}" for i in range(len(chunks))])

collection.add(
    documents=chunks,
    ids=[f"video_chunk_{i}" for i in range(len(chunks))],
)

print("✅ ChromaDB updated with video transcript.")


