# mcp_server/server.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from openai import OpenAI

from pathlib import Path
# add this:
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# load env/.env using an absolute path so it works no matter the CWD
ENV_FILE = ROOT / "env" / ".env"
load_dotenv(ENV_FILE)



# ---------------- Paths & Constants ----------------
ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = ROOT / "data"
POLICY_PATH = Path(__file__).resolve().parent / "policies" / "rbac.yaml"

# Domain prior: nudge results from the active role's domain
DOMAIN_BIAS = float(os.getenv("DOMAIN_BIAS", "1.15"))  # 15% boost by default

# Default domain lists (also used as a fallback if policy file is missing)
DEFAULT_ROLES = {
    "C_LEVEL":     ["engineering", "finance", "hr", "marketing", "general"],
    "ENGINEERING": ["engineering", "general"],
    "FINANCE":     ["finance", "general"],
    "HR":          ["hr", "general"],
    "MARKETING":   ["marketing", "general"],
    "EMPLOYEE":    ["general"],
}

# ---------------- MCP Server ----------------
mcp = FastMCP("finrag-mcp")

# ---------------- Lazy singletons ----------------
_qc: QdrantClient | None = None
_oa: OpenAI | None = None
_policy: Dict[str, Any] | None = None


def qdrant() -> QdrantClient:
    """Return a cached Qdrant client. Uses local path if QDRANT_LOCAL_PATH is set, else host/port."""
    global _qc
    if _qc is None:
        local_path = os.getenv("QDRANT_LOCAL_PATH")
        if local_path:
            _qc = QdrantClient(path=local_path)
        else:
            _qc = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
            )
    return _qc


def openai_client() -> OpenAI:
    """Return a cached OpenAI client."""
    global _oa
    if _oa is None:
        _oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _oa


def embed(text: str) -> List[float]:
    """Create an embedding for a single string."""
    model = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    return openai_client().embeddings.create(model=model, input=[text]).data[0].embedding


def load_policy() -> Dict[str, Any]:
    """Load RBAC policy from YAML; fall back to DEFAULT_ROLES."""
    global _policy
    if _policy is None:
        if POLICY_PATH.exists():
            with POLICY_PATH.open("r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
        else:
            y = {}
        roles = y.get("roles") or {}
        # Normalize: ensure all keys are uppercase and values are lowercase domains
        norm: Dict[str, List[str]] = {}
        for role, ds in roles.items():
            allow = ds.get("allow", []) if isinstance(ds, dict) else (ds or [])
            norm[role.upper()] = [d.lower() for d in allow]
        # Merge defaults for any missing roles
        for r, ds in DEFAULT_ROLES.items():
            norm.setdefault(r, ds)
        _policy = {"roles": norm}
    return _policy


def allowed_domains(role: str) -> List[str]:
    """Domains allowed for a role according to policy."""
    role_u = (role or "EMPLOYEE").upper()
    return load_policy().get("roles", {}).get(role_u, DEFAULT_ROLES.get(role_u, ["general"]))


def current_role() -> str:
    """Active role stored in env (session-scoped)."""
    return os.getenv("FINRAG_ROLE", "EMPLOYEE")


def _role_allowed_for_path(role: str, path: Path) -> bool:
    """Infer domain from path (…/data/<domain>/…) and check if role can access it."""
    parts = list(path.parts)
    domain = "general"
    if "data" in parts:
        i = parts.index("data")
        if i + 1 < len(parts):
            domain = parts[i + 1].lower()
    return domain in allowed_domains(role)


# ---------------- Tools ----------------
@mcp.tool()
def set_role(
    role: Literal["EMPLOYEE", "ENGINEERING", "FINANCE", "HR", "MARKETING", "C_LEVEL"]
) -> List[str]:
    """
    Set the active role for this MCP session.
    Returns the domains the role can access.
    """
    os.environ["FINRAG_ROLE"] = role
    return allowed_domains(role)


@mcp.tool()
def list_files(
    domain: Literal["engineering", "finance", "hr", "marketing", "general"]
) -> List[str]:
    """
    List files under data/<domain>, RBAC-enforced.
    Returns relative paths from repo root.
    """
    role = current_role()
    if domain not in allowed_domains(role):
        return [f"[denied for role={role}] {domain}"]
    base = DATA_DIR / domain
    if not base.exists():
        return [f"[missing] {base}"]
    out: List[str] = []
    for p in base.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            out.append(str(p.relative_to(ROOT)))
    return out[:500]


@mcp.tool()
def read_file(rel_path: str, max_chars: int = 4000) -> str:
    """
    Read a text-like file relative to repo root, RBAC-enforced by folder domain.
    """
    role = current_role()
    abs_path = (ROOT / rel_path).resolve()
    if not abs_path.exists() or not abs_path.is_file():
        return f"[not found] {rel_path}"
    if not _role_allowed_for_path(role, abs_path):
        # redact exact location but reveal denied domain for clarity
        parts = list(abs_path.parts)
        domain = "general"
        if "data" in parts:
            i = parts.index("data")
            if i + 1 < len(parts):
                domain = parts[i + 1]
        return f"[denied for role={role}] {domain}/{abs_path.name}"

    suf = abs_path.suffix.lower()
    if suf in (".txt", ".md", ".csv", ".json", ".yaml", ".yml"):
        try:
            return abs_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception as e:
            return f"[error reading file] {e}"
    return f"[unsupported preview: {suf}] {rel_path}"


def _qdrant_search(
    collection: str,
    vector: List[float],
    top_k: int,
) -> List[Any]:
    """
    Query Qdrant with best-available API:
    - Prefer query_points if present
    - Fallback to legacy search
    Always request payload; vectors not needed.
    """
    qc = qdrant()
    try:
        # Newer client
        return qc.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
    except AttributeError:
        # Older client
        return qc.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )


@mcp.tool()
def search(
    query: str,
    top_k: int = 8,
    include_general: bool = True,
) -> List[Dict[str, Any]]:
    """
    RBAC-aware vector search across allowed domains/collections.
    Args:
      query: natural language query
      top_k: number of results to return
      include_general: include 'general' alongside role-specific domains
    Returns a list of hits with:
      score, collection, domain, source, title, doc_type, tags, chunk_id, snippet
    """
    role = current_role()
    role_domains = allowed_domains(role)
    if not include_general:
        role_domains = [d for d in role_domains if d != "general"] or role_domains

    prefix = os.getenv("QDRANT_COLLECTION_PREFIX", "finrag")
    collections = [f"{prefix}_{d}" for d in role_domains]

    vector = embed(query)
    hits: List[Dict[str, Any]] = []
    role_domain = role.lower().strip()

    for col in collections:
        try:
            res = _qdrant_search(col, vector, top_k)
            # normalize result items
            items: List[Any]
            if hasattr(res, "points"):
                items = list(res.points)
            else:
                items = list(res or [])

            for p in items:
                payload = getattr(p, "payload", None)
                score = float(getattr(p, "score", 0.0))
                if payload is None and isinstance(p, dict):
                    payload = p.get("payload", {})
                    score = float(p.get("score", 0.0))
                # Some client versions may return (point,) tuples
                if payload is None and isinstance(p, (list, tuple)) and p:
                    maybe = p[0]
                    payload = getattr(maybe, "payload", None) or {}
                    score = float(getattr(maybe, "score", 0.0) or score)

                pl = payload or {}
                dom = (pl.get("domain") or "").lower()
                prior = DOMAIN_BIAS if dom == role_domain and DOMAIN_BIAS > 0 else 1.0
                final_score = score * float(pl.get("boost", 1.0)) * prior

                hits.append(
                    {
                        "score": final_score,
                        "collection": col,
                        "domain": pl.get("domain"),
                        "source": pl.get("source"),
                        "title": pl.get("title"),
                        "doc_type": pl.get("doc_type"),
                        "tags": pl.get("tags", []),
                        "chunk_id": pl.get("chunk_id"),
                        "snippet": (pl.get("text") or "")[:600],
                    }
                )
        except Exception as e:
            hits.append({"collection": col, "error": str(e)})

    hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return hits[:top_k]


# ---------------- Main ----------------
if __name__ == "__main__":
    mcp.run()   # runs over stdio by default

