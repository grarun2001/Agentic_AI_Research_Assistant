import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project Information
    PROJECT_NAME: str = "Agentic AI Research Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM Settings (Google Gemini)
    GEMINI_API_KEY: str = Field(default="", validation_alias="GEMINI_API_KEY")
    GOOGLE_API_KEY: str = Field(default="", validation_alias="GOOGLE_API_KEY")
    GEMINI_MODEL: str = "gemini-flash-lite-latest"
    GEMINI_TEMPERATURE: float = 0.2

    # External Web Research (Tavily)
    TAVILY_API_KEY: str = Field(default="", validation_alias="TAVILY_API_KEY")

    # Milvus Vector Database
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "agentic_rag_documents"

    # Embeddings & Reranking
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval Hyperparameters
    SEMANTIC_TOP_K: int = 10
    KEYWORD_TOP_K: int = 10
    HYBRID_TOP_K: int = 8
    RRF_K: int = 60

    # Ingestion & Text Splitting
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 150

    # Storage Paths
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = BASE_DIR / "data" / "documents"
    INDEX_FILE: Path = BASE_DIR / "data" / "index" / "chunks.json"

    # Reliability & Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 45
    MAX_RETRIES: int = 2
    INITIAL_BACKOFF_SECONDS: float = 1.0

    # API Configuration
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    BACKEND_API_URL: str = "http://127.0.0.1:8000"

    # Redis & Caching Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600
    MAX_CONCURRENT_REQUESTS: int = 10

    # Celery Background Task Queue
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    def get_gemini_api_key(self) -> str:
        """Returns the Gemini API key from GEMINI_API_KEY or GOOGLE_API_KEY."""
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    def setup_directories(self) -> None:
        """Ensure necessary directories exist on startup."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

# Centralized Settings Instance
settings = Settings()
settings.setup_directories()
