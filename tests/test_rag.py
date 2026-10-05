import pytest
from app.rag.retrieval import reciprocal_rank_fusion, bm25_keyword_search

def test_reciprocal_rank_fusion_math():
    # Mock semantic and BM25 results
    semantic_results = [
        {"chunk_id": "c1", "text": "leave policy text", "score": 0.9},
        {"chunk_id": "c2", "text": "remote work text", "score": 0.8},
        {"chunk_id": "c3", "text": "401k text", "score": 0.7}
    ]
    bm25_results = [
        {"chunk_id": "c2", "text": "remote work text", "score": 4.5},
        {"chunk_id": "c1", "text": "leave policy text", "score": 3.2},
        {"chunk_id": "c4", "text": "cyber security text", "score": 2.1}
    ]

    fused = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        rrf_k=60,
        top_k=4
    )

    assert len(fused) == 4
    # c1 and c2 appear in both lists, so their RRF scores should be highest
    top_cids = [item["chunk_id"] for item in fused[:2]]
    assert "c1" in top_cids
    assert "c2" in top_cids

    # RRF score should be greater than single list item
    c1_score = next(item["rrf_score"] for item in fused if item["chunk_id"] == "c1")
    c3_score = next(item["rrf_score"] for item in fused if item["chunk_id"] == "c3")
    assert c1_score > c3_score

def test_bm25_keyword_search_with_filtering():
    # Test query on indexed chunks
    results = bm25_keyword_search("cyber security internet harassment", top_k=5)
    assert isinstance(results, list)
    if results:
        assert "chunk_id" in results[0]
        assert "text" in results[0]
        assert results[0]["score"] > 0
