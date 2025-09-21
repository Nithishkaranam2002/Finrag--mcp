# vectorstore/qdrant_init.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

def connect_qdrant(host: str | None = None, port: int | None = None) -> QdrantClient:
    """
    Prefer local mode if QDRANT_LOCAL_PATH is set; otherwise use host/port.
    """
    local_path = os.getenv("QDRANT_LOCAL_PATH")
    if local_path:
        # Persistent local storage in the given folder (no Docker needed)
        return QdrantClient(path=local_path)
    return QdrantClient(host=host or "localhost", port=port or 6333)

def ensure_collection(client: QdrantClient, name: str, dim: int = 3072):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
