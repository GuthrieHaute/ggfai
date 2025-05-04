# run_with_grace.py - Graceful Degradation Manager
# written by DeepSeek Chat (honor call: The Resiliency Engineer)

import time
from enum import Enum
from typing import Callable, Optional, TypeVar
import logging

T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = auto()  # Normal operation
    OPEN = auto()    # Immediate failure
    HALF_OPEN = auto() # Trial recovery

class GracefulExecutor:
    def __init__(self, 
                 max_attempts: int = 3,
                 backoff_base: float = 2.0,
                 timeout: float = 30.0):
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.last_failure_time = 0
        self.logger = logging.getLogger("GGFAI.grace")

    def execute(self, 
               operation: Callable[..., T],
               fallback: Optional[Callable[..., T]] = None,
               **kwargs) -> Optional[T]:
        """Execute with retries and fallback."""
        attempt = 0
        last_error = None
        
        while attempt < self.max_attempts:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = CircuitState.HALF_OPEN
                else:
                    return self._trigger_fallback(fallback, **kwargs)

            try:
                result = operation(**kwargs)
                self._record_success()
                return result
            except Exception as e:
                last_error = e
                self._record_failure()
                attempt += 1
                wait_time = self.backoff_base ** attempt
                self.logger.warning(f"Attempt {attempt} failed, retrying in {wait_time}s")
                time.sleep(wait_time)

        return self._trigger_fallback(fallback, **kwargs, error=last_error)

    def _record_success(self):
        """Reset circuit on successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failures = 0

    def _record_failure(self):
        """Track failures and open circuit if threshold exceeded."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.max_attempts:
            self.state = CircuitState.OPEN

    def _trigger_fallback(self, fallback: Optional[Callable[..., T]], **kwargs) -> Optional[T]:
        """Execute fallback operation if available."""
        if fallback:
            self.logger.info("Entering fallback mode")
            try:
                return fallback(**kwargs)
            except Exception as e:
                self.logger.error(f"Fallback also failed: {str(e)}")
        return None