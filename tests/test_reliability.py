import pytest
import time
from app.reliability.circuit_breaker import CircuitBreaker, CircuitState, get_circuit_breaker
from app.reliability.resilient import resilient_call

def test_circuit_breaker_transitions():
    cb = CircuitBreaker(name="test_service", failure_threshold=2, recovery_timeout=0.2)
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # 1st failure: still closed
    cb.record_failure(Exception("fail 1"))
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # 2nd failure: trips to OPEN
    cb.record_failure(Exception("fail 2"))
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False

    # Wait for recovery timeout
    time.sleep(0.25)
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Success resets to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0

def test_resilient_call_fallback_on_failure():
    def failing_func():
        raise ValueError("Simulated remote dependency crash")

    def fallback_func():
        return "fallback_value_success"

    # Use a dedicated circuit breaker
    res = resilient_call(
        service_name="test_mock_service",
        func=failing_func,
        fallback=fallback_func,
        max_retries=1,
        backoff_factor=0.05
    )
    assert res == "fallback_value_success"

def test_independent_circuit_breakers():
    cb_milvus = get_circuit_breaker("milvus")
    cb_gemini = get_circuit_breaker("gemini")
    cb_reranker = get_circuit_breaker("reranker")
    cb_tavily = get_circuit_breaker("tavily")

    cb_milvus.reset()
    cb_gemini.reset()
    cb_reranker.reset()
    cb_tavily.reset()

    # Trip milvus only
    for _ in range(cb_milvus.failure_threshold):
        cb_milvus.record_failure(Exception("Milvus connection timeout"))

    assert cb_milvus.state == CircuitState.OPEN
    # Other circuit breakers remain CLOSED
    assert cb_gemini.state == CircuitState.CLOSED
    assert cb_reranker.state == CircuitState.CLOSED
    assert cb_tavily.state == CircuitState.CLOSED

    # Clean up
    cb_milvus.reset()
