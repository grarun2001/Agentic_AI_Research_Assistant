import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_INSTANCE: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-loads and caches the SentenceTransformer embedding model on CPU.
    """
    global _EMBEDDING_MODEL_INSTANCE
    if _EMBEDDING_MODEL_INSTANCE is None:
        logger.info(f"Loading SentenceTransformer model '{settings.EMBEDDING_MODEL}' on CPU...")
        _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
        logger.info("SentenceTransformer model loaded successfully.")
    return _EMBEDDING_MODEL_INSTANCE

def embed_text(text: str) -> List[float]:
    """Computes a single normalized 384-d dense embedding vector for a query."""
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.tolist()

def embed_documents(texts: List[str]) -> List[List[float]]:
    """Computes normalized embeddings for a batch of text chunks."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )
    return embeddings.tolist()
