# gateway/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from openai import OpenAI

# ---------------- Env & clients ----------------
load_dotenv("env/.env")

# Qdrant (embedded local if QDRANT_LOCAL_PATH is set; otherwise host/port)
_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH")
if _LOCAL_PATH:
    qc = QdrantClient(path=_LOCAL_PATH)
else:
    qc = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# RBAC policy (reuse same file as MCP server)
PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "finrag")
RBAC_PATH = Path(__file__).resolve().parents[1] / "mcp_server" / "policies" / "rbac.yaml"
RBAC = yaml.safe_load(open(RBAC_PATH, "r", encoding="utf-8")) if RBAC_PATH.exists() else {"roles": {}}


def allowed_domains(role: str) -> List[str]:
    roles = (RBAC or {}).get("roles", {})
    entry = roles.get(role) or roles.get(role.upper()) or {}
    allow = entry.get("allow") or []
    return list(allow) if allow else ["general"]


def embed(text: str) -> List[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


# ---------------- Qdrant helpers ----------------
def _query_points(collection: str, vector: List[float], top_k: int):
    """
    Use legacy `search` for maximum compatibility across qdrant-client versions,
    and to ensure we get objects with payload/score.
    """
    return qc.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )


def _normalize_results(res: Any) -> List[Any]:
    """
    Normalize qdrant results into a flat list of items that (ideally) have .payload/.score.
    Handles QueryResponse, list, tuple, or nested structures.
    """
    if hasattr(res, "points"):  # e.g., QueryResponse
        items = res.points
    else:
        items = list(res or [])
    flat: List[Any] = []
    for it in items:
        if isinstance(it, (list, tuple)):
            flat.extend(list(it))
        else:
            flat.append(it)
    return flat


# ---------------- Retrieval ----------------
def retrieve(
    role: str,
    query: str,
    top_k: int = 50,
    include_general: bool = True,
) -> List[Dict]:
    """
    RBAC-aware retrieval across allowed collections.
    Returns dicts with: score, domain, collection, source, title, doc_type, tags, chunk_id, snippet
    On failure for a collection, appends {'error': ...} so the caller can surface issues.
    """
    domains = allowed_domains(role)

    # Optional filter to hide `general` unless asked
    if not include_general:
        domains = [d for d in domains if d != "general"]
        # Safety: never end up with zero domains
        if not domains:
            domains = allowed_domains(role)

    vector = embed(query)
    hits: List[Dict] = []
    role_domain = role.lower().strip()

    for d in domains:
        col = f"{PREFIX}_{d}"
        try:
            res = _query_points(col, vector, top_k)
            for p in _normalize_results(res):
                # tolerate tuple/dict shapes across client versions
                payload = getattr(p, "payload", None)
                score = float(getattr(p, "score", 0.0))
                if payload is None and isinstance(p, dict):
                    payload = p.get("payload", {})
                    score = float(p.get("score", 0.0))
                if payload is None and isinstance(p, (list, tuple)) and p:
                    maybe = p[0]
                    payload = getattr(maybe, "payload", None) or {}
                    score = float(getattr(maybe, "score", 0.0) or score)

                pl = payload or {}
                dom = (pl.get("domain") or d).lower()

                # Domain prior: boost same-domain results slightly so FINANCE prefers finance docs, etc.
                prior = 1.15 if dom == role_domain else 1.0

                hits.append(
                    {
                        "score": score * float(pl.get("boost", 1.0)) * prior,
                        "domain": pl.get("domain", d),
                        "collection": col,
                        "source": pl.get("source"),
                        "title": pl.get("title"),
                        "doc_type": pl.get("doc_type"),
                        "tags": pl.get("tags", []),
                        "chunk_id": pl.get("chunk_id"),
                        "snippet": (pl.get("text") or "")[:1000],
                    }
                )
        except Exception as e:
            hits.append({"error": f"{col}: {e}"})

    hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return hits[:top_k]


# ---------------- API models ----------------
class ChatRequest(BaseModel):
    query: str
    role: str = Field(default="EMPLOYEE")
    top_k: int = Field(default=50, ge=1, le=200)   # how many results to return & display
    ctx_k: int = Field(default=20, ge=1, le=100)   # how many chunks to include in the LLM context
    include_general: bool = Field(default=True)    # include `general` alongside role-specific domains


class SourceItem(BaseModel):
    score: float
    domain: str
    source: str | None = None
    snippet: str | None = None
    title: str | None = None
    doc_type: str | None = None
    tags: List[str] = []
    chunk_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []


# ---------------- FastAPI app ----------------
app = FastAPI(title="FinRAG Gateway", version="1.3")

# Allow Streamlit (localhost) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/chat", "/health", "/docs"]}


@app.get("/health")
def health():
    try:
        cols = [c.name for c in qc.get_collections().collections]
    except Exception as e:
        cols = [f"error: {e}"]
    return {
        "ok": True,
        "collections": cols,
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "local_qdrant": bool(_LOCAL_PATH),
    }


def _clean_hits(raw_hits: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Separate out any {'error': ...} rows so response building never crashes."""
    hits, errors = [], []
    for h in raw_hits:
        if isinstance(h, dict) and "error" in h:
            errors.append(str(h["error"]))
        else:
            hits.append(h)
    return hits, errors


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1) Retrieval
    try:
        raw_hits = retrieve(
            req.role,
            req.query,
            top_k=req.top_k,
            include_general=req.include_general,
        )
    except Exception as e:
        return ChatResponse(answer=f"Sorry, retrieval failed: {e}", sources=[])

    hits, errors = _clean_hits(raw_hits)
    if not hits and errors:
        msg = "Sorry, I couldn't retrieve any context:\n- " + "\n- ".join(errors[:10])
        return ChatResponse(answer=msg, sources=[])

    # 2) Build context from the first ctx_k hits (but return ALL hits for display)
    ctx_hits = hits[: min(req.ctx_k, len(hits))]
    context = "\n\n".join(
        f"[{i+1}] ({h.get('domain','?')}) {h.get('snippet','')}\nSOURCE: {h.get('source','')}"
        for i, h in enumerate(ctx_hits)
    )

    system = (
        "You are an enterprise RAG assistant. Use ONLY the provided context to answer. "
        "If the context is insufficient, say you don't know. Always cite sources as [1], [2], etc."
    )
    user = f"Question: {req.query}\n\nContext:\n{context}"

    # 3) LLM completion (graceful fallback)
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = (
            f"Model call failed: {e}\n\nTop sources:\n"
            + "\n".join(f"[{i+1}] {h.get('source','')}" for i, h in enumerate(ctx_hits))
        )

    sources = [
        SourceItem(
            score=float(h.get("score", 0.0)),
            domain=h.get("domain", "unknown"),
            source=h.get("source"),
            snippet=h.get("snippet"),
            title=h.get("title"),
            doc_type=h.get("doc_type"),
            tags=h.get("tags", []),
            chunk_id=h.get("chunk_id"),
        )
        for h in hits  # return ALL hits (no dedupe)
    ]
    return ChatResponse(answer=answer, sources=sources)
