import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from app.config import settings
from app.rag.embeddings import embed_text
from app.rag.milvus_store import milvus_store
from app.rag.ingestion import load_chunks_index

logger = logging.getLogger(__name__)

def semantic_search(
    query: str,
    top_k: int = 10,
    selected_document_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Performs dense vector retrieval via Milvus.
    """
    try:
        query_vector = embed_text(query)
        results = milvus_store.search(
            query_vector=query_vector,
            top_k=top_k,
            selected_document_ids=selected_document_ids
        )
        return results
    except Exception as e:
        logger.error(f"Semantic search error in Milvus: {e}")
        return []

def bm25_keyword_search(
    query: str,
    top_k: int = 10,
    selected_document_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Performs sparse keyword retrieval using BM25 on the persisted chunk corpus.
    Applies document_id scope filtering to candidate chunks.
    """
    all_chunks = load_chunks_index()
    if not all_chunks:
        logger.warning("BM25: No indexed chunks found in index store.")
        return []

    # 1. Filter by selected document IDs if provided
    if selected_document_ids:
        candidate_chunks = [
            c for c in all_chunks
            if c.get("document_id") in selected_document_ids
        ]
    else:
        candidate_chunks = all_chunks

    if not candidate_chunks:
        return []

    # 2. Tokenize corpus & query
    tokenized_corpus = [
        c.get("text", "").lower().split()
        for c in candidate_chunks
    ]
    query_tokens = query.lower().split()

    if not query_tokens:
        return []

    try:
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(query_tokens)

        # Pair scores with chunk objects
        scored_chunks = []
        for idx, score in enumerate(doc_scores):
            if score > 0:  # Only include matching keyword hits
                chunk_copy = dict(candidate_chunks[idx])
                chunk_copy["score"] = float(score)
                scored_chunks.append(chunk_copy)

        # Sort descending by BM25 score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
    except Exception as e:
        logger.error(f"BM25 keyword search error: {e}")
        return []

def reciprocal_rank_fusion(
    semantic_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Combines ranked lists from semantic and keyword search using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score = sum(1 / (rrf_k + rank))
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}
    semantic_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}

    # Process Semantic Results (1-indexed rank)
    for rank, item in enumerate(semantic_results, start=1):
        cid = item.get("chunk_id")
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        chunk_map[cid] = item
        semantic_ranks[cid] = rank

    # Process BM25 Results (1-indexed rank)
    for rank, item in enumerate(bm25_results, start=1):
        cid = item.get("chunk_id")
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        if cid not in chunk_map:
            chunk_map[cid] = item
        bm25_ranks[cid] = rank

    # Sort combined pool by fused RRF score
    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_results: List[Dict[str, Any]] = []
    for cid in sorted_chunk_ids[:top_k]:
        item = dict(chunk_map[cid])
        item["rrf_score"] = round(scores[cid], 5)
        item["semantic_rank"] = semantic_ranks.get(cid)
        item["bm25_rank"] = bm25_ranks.get(cid)
        fused_results.append(item)

    return fused_results

def hybrid_retrieval(
    query: str,
    selected_document_ids: Optional[List[str]] = None,
    semantic_top_k: Optional[int] = None,
    keyword_top_k: Optional[int] = None,
    hybrid_top_k: Optional[int] = None,
    rrf_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Executes end-to-end Hybrid Search combining Semantic Vector Search and BM25 with RRF.
    """
    sem_k = semantic_top_k or settings.SEMANTIC_TOP_K
    kw_k = keyword_top_k or settings.KEYWORD_TOP_K
    hyb_k = hybrid_top_k or settings.HYBRID_TOP_K
    k_val = rrf_k or settings.RRF_K

    # 1. Semantic Search
    semantic_hits = semantic_search(query, top_k=sem_k, selected_document_ids=selected_document_ids)

    # 2. BM25 Keyword Search
    bm25_hits = bm25_keyword_search(query, top_k=kw_k, selected_document_ids=selected_document_ids)

    # 3. Reciprocal Rank Fusion
    fused_hits = reciprocal_rank_fusion(
        semantic_results=semantic_hits,
        bm25_results=bm25_hits,
        rrf_k=k_val,
        top_k=hyb_k
    )

    logger.info(
        f"Hybrid Retrieval completed for query '{query[:40]}...': "
        f"{len(semantic_hits)} semantic, {len(bm25_hits)} bm25 -> {len(fused_hits)} fused candidates."
    )
    return fused_hits
