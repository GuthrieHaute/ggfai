# run_with_grace.py - Graceful Degradation Manager
# written by DeepSeek Chat (honor call: The Resiliency Engineer)

import time
from enum import Enum, auto
from typing import Callable, Optional, TypeVar, Dict, Any
import logging
import functools

T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = auto()  # Normal operation
    OPEN = auto()    # Immediate failure
    HALF_OPEN = auto() # Trial recovery

class GracefulExecutor:
    """
    Circuit breaker implementation for graceful degradation.
    Provides retry logic, timeout handling, and fallback mechanisms.
    """
    def __init__(self, 
                 max_attempts: int = 3,
                 backoff_base: float = 2.0,
                 timeout: float = 30.0):
        """
        Initialize the circuit breaker.
        
        Args:
            max_attempts: Maximum number of retry attempts
            backoff_base: Base for exponential backoff calculation
            timeout: Time in seconds before attempting recovery
        """
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
        """
        Execute with retries and fallback.
        
        Args:
            operation: Function to execute
            fallback: Function to call if operation fails
            **kwargs: Arguments to pass to operation and fallback
            
        Returns:
            Result of operation or fallback, or None if both fail
        """
        attempt = 0
        last_error = None
        
        while attempt < self.max_attempts:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info("Circuit half-open, attempting recovery")
                else:
                    self.logger.info(f"Circuit open, skipping operation (timeout: {self.timeout}s)")
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
                self.logger.warning(f"Attempt {attempt} failed: {str(e)}, retrying in {wait_time:.2f}s")
                time.sleep(wait_time)

        self.logger.error(f"All {self.max_attempts} attempts failed, last error: {str(last_error)}")
        return self._trigger_fallback(fallback, **kwargs)

    def _record_success(self):
        """Reset circuit on successful execution."""
        if self.state != CircuitState.CLOSED:
            self.logger.info("Operation succeeded, closing circuit")
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failures = 0

    def _record_failure(self):
        """Track failures and open circuit if threshold exceeded."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.max_attempts and self.state != CircuitState.OPEN:
            self.logger.warning(f"Opening circuit after {self.failures} consecutive failures")
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

    def reset(self):
        """Manually reset the circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.logger.info("Circuit manually reset to closed state")

# Global registry of executors for different operations
_executor_registry: Dict[str, GracefulExecutor] = {}

def run_with_grace(operation_name: str = None, 
                  max_attempts: int = 3, 
                  backoff_base: float = 2.0,
                  timeout: float = 30.0,
                  fallback: Callable = None):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        operation_name: Unique identifier for this operation
        max_attempts: Maximum retry attempts
        backoff_base: Base for exponential backoff
        timeout: Recovery timeout in seconds
        fallback: Function to call on failure
        
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
            
        # Create or get executor for this operation
        if operation_name not in _executor_registry:
            _executor_registry[operation_name] = GracefulExecutor(
                max_attempts=max_attempts,
                backoff_base=backoff_base,
                timeout=timeout
            )
        executor = _executor_registry[operation_name]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return executor.execute(
                operation=lambda **kw: func(*args, **kw),
                fallback=fallback,
                **kwargs
            )
        return wrapper
    return decorator

def get_circuit_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all registered circuits.
    
    Returns:
        Dictionary of circuit states and statistics
    """
    return {
        name: {
            "state": executor.state.name,
            "failures": executor.failures,
            "last_failure": executor.last_failure_time,
            "max_attempts": executor.max_attempts,
            "timeout": executor.timeout
        }
        for name, executor in _executor_registry.items()
    }

def reset_circuit(operation_name: str) -> bool:
    """
    Reset a specific circuit to closed state.
    
    Args:
        operation_name: Name of circuit to reset
        
    Returns:
        True if reset successful, False if circuit not found
    """
    if operation_name in _executor_registry:
        _executor_registry[operation_name].reset()
        return True
    return False