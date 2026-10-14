import json
from pathlib import Path
import numpy as np

class VectorStore:
    """Small persistent cosine-similarity store; appropriate for a demo/small corpus."""
    def __init__(self, path: str, embedder):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.records = []
        self.vectors = np.empty((0, 0), dtype=np.float32)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = payload.get("records", [])
            vectors = payload.get("vectors", [])
            self.vectors = np.asarray(vectors, dtype=np.float32) if vectors else np.empty((0, 0), dtype=np.float32)

    def _save(self):
        self.path.write_text(json.dumps({"records": self.records, "vectors": self.vectors.tolist()}), encoding="utf-8")

    def add_document(self, name: str, chunks: list[str]) -> int:
        if not chunks:
            return 0
        # Re-ingestion replaces the document, avoiding duplicate stale chunks.
        keep = [i for i, r in enumerate(self.records) if r["document_name"] != name]
        if len(keep) != len(self.records):
            self.records = [self.records[i] for i in keep]
            self.vectors = self.vectors[keep] if len(keep) else np.empty((0, 0), dtype=np.float32)
        new_vectors = self.embedder.encode(chunks)
        if self.vectors.size and self.vectors.shape[1] != new_vectors.shape[1]:
            raise ValueError("Embedding dimensions changed; clear the vector store before changing models")
        self.records.extend({"document_name": name, "text": chunk} for chunk in chunks)
        self.vectors = new_vectors if not self.vectors.size else np.vstack([self.vectors, new_vectors])
        self._save()
        return len(chunks)

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        if not self.records:
            return []
        query = self.embedder.encode([question])[0]
        # Embeddings are normalized, so dot product is cosine similarity.
        scores = self.vectors @ query
        indices = np.argsort(scores)[::-1][:top_k]
        return [{"document_name": self.records[int(i)]["document_name"],
                 "text": self.records[int(i)]["text"], "score": float(scores[int(i)])}
                for i in indices]
