from fastapi.testclient import TestClient
import numpy as np
import app.main as main

class FakeEmbedder:
    def encode(self, texts):
        # Deterministic toy embeddings for isolated tests; not semantic model quality.
        out = []
        for t in texts:
            t = t.lower()
            out.append([sum(w in t for w in ["leave", "vacation", "annual"]),
                        sum(w in t for w in ["office", "location", "address"]),
                        1.0])
        a = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(norms, 1e-9)

class FakeStore:
    def __init__(self):
        from app.store import VectorStore
        self.store = VectorStore(":memory:", FakeEmbedder())
    @property
    def records(self): return self.store.records
    def add_document(self, name, chunks): return self.store.add_document(name, chunks)
    def search(self, q, k=5): return self.store.search(q, k)

class FakeAnswerer:
    def answer(self, question, evidence):
        if "Mars" in question:
            return "The information is unavailable in the provided documents."
        return f"According to {evidence[0]['document_name']}, " + evidence[0]["text"]

main._store = FakeStore()
main._answerer = FakeAnswerer()
client = TestClient(main.app)

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_ingest_and_answer():
    r = client.post("/documents", files={"file": ("policy.txt", b"Annual leave is 20 days per year.", "text/plain")})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1
    answer = client.post("/ask", json={"question": "How much annual leave is provided?"})
    assert answer.status_code == 200
    assert answer.json()["supported"] is True
    assert answer.json()["evidence"][0]["document_name"] == "policy.txt"

def test_unsupported_file():
    r = client.post("/documents", files={"file": ("bad.exe", b"no", "application/octet-stream")})
    assert r.status_code == 415

def test_unanswerable_question():
    r = client.post("/ask", json={"question": "What is the population of Mars?"})
    assert r.status_code == 200
    assert r.json()["supported"] is False
