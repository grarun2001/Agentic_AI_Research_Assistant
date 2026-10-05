import logging
from typing import List, Dict, Any, Optional

from app.rag.retrieval import hybrid_retrieval
from app.rag.reranking import rerank_chunks

logger = logging.getLogger(__name__)

def query_internal_rag(
    query: str,
    selected_document_ids: Optional[List[str]] = None,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Executes the full internal company RAG pipeline:
    1. Hybrid Retrieval (Milvus Semantic + Local BM25 + RRF)
    2. Cross-Encoder Reranking
    Returns structured, citation-ready text chunks.
    """
    logger.info(f"Querying internal RAG for: '{query}' with doc scope: {selected_document_ids}")

    # 1. Hybrid Retrieval
    candidates = hybrid_retrieval(
        query=query,
        selected_document_ids=selected_document_ids
    )

    if not candidates:
        logger.info("Internal RAG: No candidate chunks found matching criteria.")
        return []

    # 2. Cross-Encoder Reranking
    reranked_chunks = rerank_chunks(
        query=query,
        candidate_chunks=candidates,
        top_n=top_n
    )

    return reranked_chunks
