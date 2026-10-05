import time
import pytest
from app.cache.redis_cache import cache_service
from app.concurrency.limiter import concurrency_controller
from app.services.pipeline_service import execute_scalable_research_pipeline

def test_cache_service_set_and_get():
    """Verify cache set, get, and key generation."""
    key = cache_service.generate_cache_key("test question about policies", ["doc-1"])
    test_payload = {
        "final_answer": "Test answer",
        "route": "rag",
        "sources": {"internal": [], "external": []}
    }
    
    # Cache set
    cache_service.set(key, test_payload, ttl=60)
    
    # Cache get
    cached = cache_service.get(key)
    assert cached is not None
    assert cached["final_answer"] == "Test answer"
    assert cached["route"] == "rag"

def test_concurrency_controller():
    """Verify concurrency acquire and release."""
    initial_active = concurrency_controller.active_requests
    with concurrency_controller.acquire(timeout=2.0):
        assert concurrency_controller.active_requests == initial_active + 1
    assert concurrency_controller.active_requests == initial_active

def test_scalable_pipeline_caching():
    """Verify end-to-end pipeline returns cached=True on second execution."""
    test_query = f"Unique scalability test query {time.time()}"
    
    # 1. First execution -> Cache Miss (is_cached=False)
    res1, is_cached1 = execute_scalable_research_pipeline(test_query)
    assert is_cached1 is False
    assert res1 is not None
    
    # 2. Second execution -> Cache Hit (is_cached=True)
    res2, is_cached2 = execute_scalable_research_pipeline(test_query)
    assert is_cached2 is True
    assert res2["final_answer"] == res1["final_answer"]
