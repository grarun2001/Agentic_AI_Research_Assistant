import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

from app.config import settings
from app.reliability.resilient import resilient_call

logger = logging.getLogger(__name__)

_RERANKER_MODEL_INSTANCE: Optional[CrossEncoder] = None

def get_reranker_model() -> CrossEncoder:
    """Lazy-loads and caches the Cross-Encoder model on CPU."""
    global _RERANKER_MODEL_INSTANCE
    if _RERANKER_MODEL_INSTANCE is None:
        logger.info(f"Loading Cross-Encoder reranker '{settings.RERANKER_MODEL}' on CPU...")
        _RERANKER_MODEL_INSTANCE = CrossEncoder(settings.RERANKER_MODEL, device="cpu")
        logger.info("Cross-Encoder reranker loaded successfully.")
    return _RERANKER_MODEL_INSTANCE

def rerank_chunks(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Reranks candidate chunks from hybrid retrieval using a Cross-Encoder.
    Protected with independent circuit breaker and fallback to RRF order.
    """
    if not candidate_chunks:
        return []

    if len(candidate_chunks) <= 1:
        return candidate_chunks

    def _execute_rerank() -> List[Dict[str, Any]]:
        model = get_reranker_model()
        # Prepare input sentence pairs: (query, passage)
        sentence_pairs = [[query, chunk.get("text", "")] for chunk in candidate_chunks]
        scores = model.predict(sentence_pairs, show_progress_bar=False)

        # Attach rerank score and sort
        scored_chunks = []
        for chunk, score in zip(candidate_chunks, scores):
            c_copy = dict(chunk)
            c_copy["rerank_score"] = float(round(score, 5))
            scored_chunks.append(c_copy)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info(f"Cross-Encoder reranked {len(scored_chunks)} chunks -> selected top {top_n}.")
        return scored_chunks[:top_n]

    def _rerank_fallback() -> List[Dict[str, Any]]:
        logger.warning("Reranker fallback activated: preserving RRF hybrid rank order without cross-encoding.")
        # Simply return the top candidates in their original RRF order
        for c in candidate_chunks:
            c["rerank_score"] = c.get("rrf_score", 0.0)
        return candidate_chunks[:top_n]

    return resilient_call(
        service_name="reranker",
        func=_execute_rerank,
        fallback=_rerank_fallback,
        max_retries=1,
        backoff_factor=0.5
    )
