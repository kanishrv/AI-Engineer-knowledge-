# AI-Powered Knowledge Assistant

A small document-grounded Q&A API using FastAPI, local sentence-transformer embeddings, cosine-similarity retrieval, and an LLM API. It accepts PDF, TXT, and Markdown files, returns evidence excerpts, and has an explicit no-evidence response.

## Architecture

1. **Ingestion** (`POST /documents`): validates the extension and size, extracts PDF text with `pypdf` or decodes UTF-8 text, then chunks with overlap.
2. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` runs locally and creates normalized vectors.
3. **Vector store**: a simple JSON-backed NumPy vector store persists chunk metadata and vectors. Cosine similarity is computed as a dot product.
4. **Answer generation**: OpenAI Chat Completions receives only retrieved excerpts, temperature 0, and explicit instructions not to infer facts.
5. **Reliability**: if no chunk reaches `MIN_RELEVANCE`, `/ask` returns a fixed unavailable answer without calling the LLM. API errors are mapped to HTTP errors.
6. **Tests**: pytest uses deterministic fake embeddings and a fake answerer so unit/API tests do not require network access or API keys.

For a larger corpus, replace the JSON store with FAISS, Chroma, Qdrant, or a managed vector database. JSON/NumPy is intentionally simple, not a production-scale index.

## Requirements

- Python 3.11+
- An OpenAI API key for real answer generation
- Internet access the first time the embedding model is downloaded

## Local setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
uvicorn app.main:app --reload
```

Open interactive API docs at http://localhost:8000/docs.

## Docker

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
docker compose up --build
```

The `data/` directory is mounted for persistence. The first start may take time while the embedding model downloads.

## API

### `POST /documents`

Multipart form upload field: `file`. Accepted extensions: `.pdf`, `.txt`, `.md`, `.markdown`; max upload size is 20 MB.

Example:

```bash
curl -F "file=@sample_docs/leave_policy.md" http://localhost:8000/documents
```

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many annual leave days do full-time employees receive?"}'
```

Response contains `answer`, `supported`, and `evidence` with `document_name`, `excerpt`, and cosine `relevance`.

### `GET /health`

Returns `{"status":"ok"}`.

## Automated tests

```bash
pytest -q
```

The test suite checks chunking, health endpoint, ingestion, evidence metadata, unsupported file handling, and unavailable-answer behavior. It uses fakes, so it tests application behavior rather than the quality of the real embedding model or LLM. Run the command in your own environment and include its actual output in your submission; results are not claimed here because they have not been executed in this package-building step.

## Evaluation set

`evaluation/questions.json` contains 12 questions:
- 6 single-document answerable questions
- 2 questions requiring multiple facts from the same source document
- 4 unanswerable questions

Suggested manual evaluation:
1. Ingest both sample documents.
2. Submit each question to `/ask`.
3. For answerable questions, verify every claim against evidence excerpts.
4. For unanswerable questions, check that the response clearly says information is unavailable.
5. Record answer correctness, evidence precision, and abstention correctness.

## Design decisions and limitations

- **Local embeddings** reduce embedding API cost and keep document text local for embedding, but the model download is required.
- **OpenAI generation** is configurable through environment variables. The API key is never hard-coded.
- **Relevance threshold** is configurable. Cosine scores are model- and corpus-dependent; tune `MIN_RELEVANCE` on a validation set rather than assuming one threshold fits all.
- **PDF extraction** only extracts selectable text; scanned PDFs need OCR, which is not included.
- **Storage** rewrites a JSON file and loads vectors into memory. It is for a small assignment corpus, not concurrent or large-scale production use.
- **Prompting is not a complete security boundary**: retrieved text is treated as untrusted, but production systems should add robust prompt-injection evaluation, access control, deletion endpoints, observability, and possibly reranking.
- **Evidence indicates retrieval provenance**, not a formal proof that every generated sentence is entailed. Evaluate grounding and consider sentence-level citations for higher assurance.
- API calls require network access and a valid LLM key. A missing key produces a clear `503` error rather than fabricated answers.

## Repository layout

```text
app/
  config.py       # environment configuration
  chunking.py     # overlapping text chunking
  embeddings.py   # local embedding adapter
  store.py        # persistent vector search
  llm.py          # grounded LLM wrapper
  schemas.py      # request/response models
  main.py         # FastAPI routes
tests/
sample_docs/
evaluation/questions.json
Dockerfile
docker-compose.yml
```
