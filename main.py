from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from io import BytesIO
from app.config import settings
from app.chunking import split_text
from app.embeddings import SentenceTransformerEmbedder
from app.store import VectorStore
from app.llm import OpenAIAnswerer
from app.schemas import IngestResponse, AskRequest, AskResponse, Evidence

SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}
app = FastAPI(title="AI Knowledge Assistant", version="1.0.0")

# Lazy construction prevents model downloads during module import and makes tests easier.
_store = None
_answerer = None

def get_store():
    global _store
    if _store is None:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        _store = VectorStore(str(Path(settings.data_dir) / "vectors.json"), embedder)
    return _store

def get_answerer():
    global _answerer
    if _answerer is None:
        _answerer = OpenAIAnswerer(settings.openai_api_key, settings.llm_model)
    return _answerer

def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc
    if suffix in {".txt", ".md", ".markdown"}:
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Text documents must be UTF-8 encoded") from exc
    raise ValueError("Unsupported file type. Upload PDF, TXT, or Markdown.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/documents", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=415, detail="Supported formats: PDF, TXT, Markdown")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum upload size is 20 MB")
    try:
        text = extract_text(filename, content)
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(status_code=422, detail="Document contains no extractable text")
        store = get_store()
        added = store.add_document(filename, chunks)
        return IngestResponse(document_name=filename, chunks_added=added, total_chunks=len(store.records))
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {type(exc).__name__}") from exc

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        matches = get_store().search(request.question, request.top_k or settings.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Retrieval unavailable: {type(exc).__name__}") from exc
    matches = [m for m in matches if m["score"] >= settings.min_relevance]
    if not matches:
        return AskResponse(answer="The information is unavailable in the provided documents.",
                           supported=False, evidence=[])
    evidence = [Evidence(document_name=m["document_name"], excerpt=m["text"], relevance=round(m["score"], 4))
                for m in matches]
    try:
        answer = get_answerer().answer(request.question, matches)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {type(exc).__name__}") from exc
    unavailable = "information is unavailable in the provided documents" in answer.lower()
    return AskResponse(answer=answer, supported=not unavailable, evidence=evidence)
