"""
Sovren backend — a grounded Q&A API over a private knowledge base.

Flow of one request:
  1. On startup, read every .md file in knowledge/, split into chunks, embed each chunk.
  2. When /ask is hit, embed the question, find the closest chunks (cosine similarity),
     and pass only those chunks to the LLM to answer — grounded, with sources.

This is the "grounded" layer. The agent (tools) and policy rules come on top later.
"""

import glob
import os

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from agent import run_agent

load_dotenv()

# --- Config ---
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
TOP_K = 3  # how many chunks to retrieve per question
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs locally — no API key needed

app = FastAPI(title="Sovren API")

# Allow the React frontend (any origin for now) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# These get filled in on startup and reused across requests.
embedder: SentenceTransformer | None = None
chunks: list[str] = []          # the raw text of every chunk
chunk_sources: list[str] = []   # which file each chunk came from (for citations)
chunk_vectors: np.ndarray | None = None  # one embedding row per chunk


def load_and_chunk() -> tuple[list[str], list[str]]:
    """Read every markdown file and split it into chunks, one per '## section'."""
    texts, sources = [], []
    for path in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Split on '## ' headings so each chunk is one coherent section.
        for section in content.split("\n## "):
            section = section.strip()
            if section:
                texts.append(section)
                sources.append(name)
    return texts, sources


@app.on_event("startup")
def startup() -> None:
    """Load the model and pre-compute an embedding for every chunk, once."""
    global embedder, chunks, chunk_sources, chunk_vectors
    embedder = SentenceTransformer(EMBED_MODEL)
    chunks, chunk_sources = load_and_chunk()
    chunk_vectors = embedder.encode(chunks, normalize_embeddings=True)
    print(f"Sovren: loaded {len(chunks)} chunks from {KNOWLEDGE_DIR}")


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Embed the question and return the k most similar chunks with their scores."""
    q_vec = embedder.encode([question], normalize_embeddings=True)[0]
    # Because vectors are normalized, a dot product IS the cosine similarity.
    scores = chunk_vectors @ q_vec
    top_idx = np.argsort(scores)[::-1][:k]
    return [
        {"text": chunks[i], "source": chunk_sources[i], "score": float(scores[i])}
        for i in top_idx
    ]


def answer_with_llm(question: str, retrieved: list[dict]) -> str:
    """Ask an LLM to answer using ONLY the retrieved chunks. Falls back if no key."""
    context = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # No key yet — return the retrieved text so you can still see the pipeline work.
        return "(no LLM key set) Most relevant policy section:\n\n" + retrieved[0]["text"]

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# --- API ---
class AskRequest(BaseModel):
    question: str


@app.get("/")
def health() -> dict:
    return {"status": "ok", "chunks_loaded": len(chunks)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    retrieved = retrieve(req.question)
    # The agent decides whether to answer from context or call a tool, and
    # returns a trace of every step it took.
    result = run_agent(req.question, retrieved)
    return {
        "answer": result["answer"],
        "trace": result["trace"],
        "sources": [
            {"source": r["source"], "score": round(r["score"], 3)} for r in retrieved
        ],
    }
