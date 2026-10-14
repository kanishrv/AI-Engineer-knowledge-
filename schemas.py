from pydantic import BaseModel, Field

class IngestResponse(BaseModel):
    document_name: str
    chunks_added: int
    total_chunks: int

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)

class Evidence(BaseModel):
    document_name: str
    excerpt: str
    relevance: float

class AskResponse(BaseModel):
    answer: str
    supported: bool
    evidence: list[Evidence]
