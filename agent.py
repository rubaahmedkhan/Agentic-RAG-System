import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from agents import function_tool, Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
import google.generativeai as genai

from chromadb import PersistentClient

import os
from dotenv import load_dotenv
load_dotenv()


from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings
from agents.run import RunConfig

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model_settings = ModelSettings(
    max_tokens=2000,
                                    
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    model_settings=model_settings,
    tracing_disabled=True
)

# === Setup ChromaDB and Embedder ===
client = PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("video_docs")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Setup Gemini ===
genai.configure(api_key="API_KEY")  # 🔑 Replace with your Gemini API key
llm = genai.GenerativeModel("gemini-1.5-flash")


# === Load vector data ===
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

embs = np.load("embeddings.npy")
index = faiss.read_index("index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")


# === RAG Search Function ===
def search_docs(query, k=5):
    qv = model.encode([query])
    D, I = index.search(np.array(qv), k)
    return [(all_chunks[i][0], all_chunks[i][1]) for i in I[0]]

# === Tool Definition ===
@function_tool
def answer_query(query: str) -> str:
    """Answer user questions using the OpenAI Agents SDK documentation."""
    docs = search_docs(query)
    context = "\n\n".join(f"From {url}:\n{chunk}" for url, chunk in docs)
    prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
    res = llm.generate_content(prompt)
    return res.text

@function_tool
def answer_from_video(query: str) -> str:
    """Answer questions based on the transcribed video content."""
    query_emb = embedder.encode([query])[0]
    results = collection.query(query_embeddings=[query_emb], n_results=5)
    docs = results["documents"][0]
    
    context = "\n\n".join(docs)
    prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.generate_content(prompt)
    return response.text

# === Agent Definition ===
agent = Agent(
    name="combined_agent",
    instructions=(
        "You are an expert on the OpenAI Agents SDK and also have access to a transcribed video about Agentic AI. "
        "You MUST use the `answer_query` tool to answer all questions related to the OpenAI Agents SDK. "
        "If the user asks a general question, answer it using the tool. "
        "If the user asks to generate MCQs, create 3-5 MCQs based on the context from the tool and include the correct answer after each MCQ. "
        "If the user asks for a summary of a topic (e.g., 'Give a summary of tracing'), use the tool to get relevant info and write a short, clear summary. "
        "Do not guess anything outside the tool's information. "
        "Always rely on the tool for facts, examples, summaries, and answers. "
        "You MUST use the `answer_from_video` tool to answer all questions based on the transcribed video about Agentic AI. "
        "You can summarize, explain, or generate MCQs with correct answers based on the video context."
    ),
    tools=[answer_query, answer_from_video],
)


# result = Runner.run_sync(agent,"Give a summary of the video.",run_config=config)
# print(result.final_output)

async def main():
  result = Runner.run_streamed(agent, input="what is handsoff",run_config=config)
  async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

asyncio.run(main())
