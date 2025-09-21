import os
from typing import List
from openai import OpenAI

class Embedder:
    def __init__(self, model: str = "text-embedding-3-large", api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI supports batching; keep batches reasonable
        out = []
        B = 64
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out
