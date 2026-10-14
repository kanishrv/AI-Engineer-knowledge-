import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "data")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.25"))

settings = Settings()
