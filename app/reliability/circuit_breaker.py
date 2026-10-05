import time
import logging
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CircuitState(str, Enum):
    CLOSED = "CLOSED"       # Normal operation; requests pass through
    OPEN = "OPEN"           # Failed repeatedly; fail fast without calling service
    HALF_OPEN = "HALF_OPEN" # Recovery timeout passed; testing service health with next request

class CircuitBreakerOpenException(Exception):
    """Raised when an operation is rejected because the circuit is open."""
    def __init__(self, service_name: str, message: str = ""):
        self.service_name = service_name
        super().__init__(message or f"Circuit breaker is OPEN for service '{service_name}'. Requests are temporarily blocked.")

class CircuitBreaker:
    """
    Independent Circuit Breaker tracking service failure states.
    Prevents cascading failure and enables graceful fallbacks.
    """
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 45.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_error_message: Optional[str] = None

    def can_execute(self) -> bool:
        """Determines if a request should be allowed to proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            now = time.time()
            if self.last_failure_time and (now - self.last_failure_time >= self.recovery_timeout):
                logger.info(f"CircuitBreaker [{self.name}]: Recovery timeout elapsed. Transitioning from OPEN to HALF_OPEN.")
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self) -> None:
        """Records a successful execution and resets or transitions state."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"CircuitBreaker [{self.name}]: Probe call succeeded. Resetting from HALF_OPEN to CLOSED.")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_error_message = None
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
        self.success_count += 1

    def record_failure(self, error: Exception) -> None:
        """Records a failed execution and trips circuit if threshold exceeded."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.last_error_message = str(error)
        logger.warning(f"CircuitBreaker [{self.name}]: Failure recorded ({self.failure_count}/{self.failure_threshold}). Error: {error}")

        if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"CircuitBreaker [{self.name}]: Threshold reached! Circuit TRIPPED to OPEN. Blocking requests for {self.recovery_timeout}s.")

    def reset(self) -> None:
        """Manually reset the circuit breaker state to CLOSED."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_error_message = None

    def get_status(self) -> Dict[str, Any]:
        """Returns the current snapshot of the circuit breaker."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self.last_failure_time,
            "last_error": self.last_error_message
        }

# Global registry of independent circuit breakers for the 4 critical dependencies
_CIRCUIT_REGISTRY: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(service_name: str, failure_threshold: int = 3, recovery_timeout: float = 45.0) -> CircuitBreaker:
    """Retrieves or registers a circuit breaker for the given service."""
    if service_name not in _CIRCUIT_REGISTRY:
        _CIRCUIT_REGISTRY[service_name] = CircuitBreaker(
            name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
    return _CIRCUIT_REGISTRY[service_name]

# Initialize standard circuit breakers
milvus_circuit = get_circuit_breaker("milvus")
gemini_circuit = get_circuit_breaker("gemini")
reranker_circuit = get_circuit_breaker("reranker")
tavily_circuit = get_circuit_breaker("tavily")

def get_all_circuit_statuses() -> Dict[str, Dict[str, Any]]:
    """Returns status dictionary for all registered circuit breakers."""
    return {name: cb.get_status() for name, cb in _CIRCUIT_REGISTRY.items()}
