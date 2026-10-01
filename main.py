"""
Sovren backend — a grounded Q&A API over a private knowledge base.

Flow of one request:
  1. On startup, read every .md file in knowledge/, split into chunks, embed each chunk.
  2. When /ask is hit, embed the question, find the closest chunks (cosine similarity),
     and pass only those chunks to the agent to answer — grounded, with sources.

Embeddings are computed via the OpenAI API (tiny memory footprint) instead of a
local model, so this runs comfortably on a 512MB free instance. If no API key is
set, retrieval falls back to simple keyword overlap so the service still boots.
"""

import glob
import os

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import run_agent
from llm import EMBED_MODEL, client, get_key

load_dotenv()

# --- Config ---
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
TOP_K = 3  # how many chunks to retrieve per question

app = FastAPI(title="Sovren API")

# Allow the React frontend (any origin for now) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Filled in on startup and reused across requests.
chunks: list[str] = []          # the raw text of every chunk
chunk_sources: list[str] = []   # which file each chunk came from (for citations)
chunk_vectors: np.ndarray | None = None  # one embedding row per chunk, or None


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


def embed(texts: list[str]) -> np.ndarray:
    """Turn texts into normalized embedding vectors using the Gemini API."""
    resp = client().embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # Normalize so a dot product equals cosine similarity.
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


@app.on_event("startup")
def startup() -> None:
    """Load chunks and, if a key is set, pre-compute an embedding for each one."""
    global chunks, chunk_sources, chunk_vectors
    chunks, chunk_sources = load_and_chunk()
    if get_key():
        try:
            chunk_vectors = embed(chunks)
            print(f"Sovren: embedded {len(chunks)} chunks")
        except Exception as e:  # don't let embedding kill startup — fall back below
            print(f"Sovren: embedding failed ({e}); using keyword fallback")
            chunk_vectors = None
    else:
        print("Sovren: no GEMINI_API_KEY; using keyword-overlap retrieval")


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return the k most relevant chunks. Uses embeddings if available, else keywords."""
    if chunk_vectors is not None:
        q_vec = embed([question])[0]
        scores = chunk_vectors @ q_vec  # cosine similarity (vectors are normalized)
    else:
        # Fallback: score by how many question words appear in each chunk.
        q_words = set(question.lower().split())
        scores = np.array(
            [len(q_words & set(c.lower().split())) for c in chunks], dtype=np.float32
        )
    top_idx = np.argsort(scores)[::-1][:k]
    return [
        {"text": chunks[i], "source": chunk_sources[i], "score": float(scores[i])}
        for i in top_idx
    ]


# --- API ---
class AskRequest(BaseModel):
    question: str


@app.get("/")
def health() -> dict:
    return {"status": "ok", "chunks_loaded": len(chunks), "embeddings": chunk_vectors is not None}


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
