# Sovren

A domain-grounded AI agent over a private knowledge base. It answers **only** from
your documents (with citations), can **call tools** to take actions, and records a
**trace** of every retrieval and tool call so each answer is auditable.

Built to mirror the shape of production enterprise AI: grounded, agentic, traceable.

## Stack
- **Backend:** FastAPI, sentence-transformers (local embeddings), OpenAI for generation
- **Frontend:** single-page React (no build step)
- **Retrieval:** cosine similarity over embedded document chunks

## Run locally

Backend:
```bash
cd backend
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...      # Windows: set OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```
API is now at http://localhost:8000 (docs at /docs).

Frontend: open `frontend/index.html` in a browser. It calls the backend at
`http://localhost:8000`. After deploying, change the `API` constant in that file
to your live backend URL.

## How it works
1. On startup, every `.md` in `backend/knowledge/` is split into sections and embedded.
2. `/ask` embeds the question, retrieves the closest sections, and passes only those to the agent.
3. The agent answers from that context, or calls a tool (`raise_ticket`, `check_leave_balance`) when the user asks for an action.
4. The response includes the answer, the sources used, and a step-by-step trace.

## Deploy
Backend deploys to Render as a web service (see `render.yaml`). The frontend is a
static file that can be served anywhere (Render static site, Netlify, GitHub Pages).
