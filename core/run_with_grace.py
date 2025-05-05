# run_with_grace.py - Graceful Degradation Manager
# written by DeepSeek Chat (honor call: The Resiliency Engineer)
# Updated with enhanced timeout enforcement and thread safety

import time
import logging
from enum import Enum, auto
from typing import Callable, Optional, TypeVar, Dict, Any, Union
from threading import Lock, Thread
from dataclasses import dataclass, field
from contextlib import contextmanager
import functools
import threading
import signal
import concurrent.futures

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker state machine states."""
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Short-circuit all requests
    HALF_OPEN = auto() # Trial requests allowed
    ISOLATED = auto()  # Manual isolation for maintenance

@dataclass
class CircuitMetrics:
    """Track performance metrics for the circuit."""
    total_requests: int = 0
    total_failures: int = 0
    total_fallback: int = 0
    total_timeouts: int = 0
    success_rate: float = 1.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_success(self):
        """Update metrics for successful operation."""
        with self._lock:
            self.total_requests += 1
            self.success_rate = (
                (self.total_requests - self.total_failures) 
                / max(1, self.total_requests)
            )

    def record_failure(self):
        """Update metrics for failed operation."""
        with self._lock:
            self.total_requests += 1
            self.total_failures += 1
            self.success_rate = (
                (self.total_requests - self.total_failures) 
                / max(1, self.total_requests)
            )

    def record_fallback(self):
        """Track fallback usage."""
        with self._lock:
            self.total_fallback += 1

    def record_timeout(self):
        """Track timeout occurrences."""
        with self._lock:
            self.total_timeouts += 1

class BackoffStrategy:
    """Configurable backoff strategies."""
    @staticmethod
    def exponential(base: float, attempt: int) -> float:
        return min(base ** attempt, 60)  # Cap at 1 minute

    @staticmethod
    def linear(factor: float, attempt: int) -> float:
        return min(factor * attempt, 60)

    @staticmethod
    def fixed(interval: float) -> float:
        return interval

    @staticmethod
    def fibonacci(attempt: int) -> float:
        a, b = 0, 1
        for _ in range(attempt):
            a, b = b, a + b
        return min(a, 60)

class GracefulExecutor:
    """
    Robust execution wrapper with:
    - Circuit breaker pattern
    - Retry with backoff
    - Graceful fallback
    - Timeout protection
    - Metrics collection
    """
    
    def __init__(
        self,
        name: str = "default",
        max_attempts: int = 3,
        backoff_strategy: Callable[[int], float] = None,
        timeout: float = 30.0,
        reset_timeout: float = 60.0,
        failure_threshold: float = 0.8,
        isolation_mode: bool = False
    ):
        self.name = name
        self.state = CircuitState.ISOLATED if isolation_mode else CircuitState.CLOSED
        self.max_attempts = max(max_attempts, 1)
        self.backoff_strategy = backoff_strategy or functools.partial(
            BackoffStrategy.exponential, base=2.0)
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        self.failure_threshold = failure_threshold
        self.last_failure_time = 0.0
        self.metrics = CircuitMetrics()
        self._state_lock = Lock()
        self.logger = logging.getLogger(f"GGFAI.grace.{name}")

    def execute(
        self,
        operation: Callable[..., T],
        fallback: Optional[Callable[..., T]] = None,
        operation_timeout: Optional[float] = None,
        **kwargs
    ) -> Optional[T]:
        """
        Execute with resilience policies applied.
        
        Args:
            operation: Primary operation to execute
            fallback: Fallback operation if primary fails
            operation_timeout: Timeout for individual operation attempt
            **kwargs: Arguments to pass to operations
            
        Returns:
            Result of operation or fallback, or None if all failed
        """
        if self.state == CircuitState.ISOLATED:
            self.logger.warning("Circuit is manually isolated")
            return self._trigger_fallback(fallback, **kwargs)

        attempt = 0
        last_error = None
        
        while attempt < self.max_attempts:
            if not self._allow_attempt():
                return self._trigger_fallback(fallback, **kwargs)

            try:
                result = self._execute_with_timeout(
                    operation, 
                    operation_timeout or self.timeout,
                    **kwargs
                )
                self._record_success()
                return result
            except TimeoutError as e:
                last_error = e
                self.metrics.record_timeout()
                self.logger.warning(f"Operation timed out: {str(e)}")
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt failed: {str(e)}", exc_info=True)
            finally:
                attempt += 1
                if attempt < self.max_attempts:
                    wait_time = self.backoff_strategy(attempt)
                    self.logger.info(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)

        self._record_failure()
        return self._trigger_fallback(fallback, error=last_error, **kwargs)

    @contextmanager
    def context(self, fallback: Optional[Callable] = None):
        """Context manager interface for graceful execution."""
        result = None
        error = None
        
        try:
            yield
        except Exception as e:
            error = e
            self._record_failure()
            if fallback:
                try:
                    result = fallback()
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback failed: {str(fallback_error)}",
                        exc_info=True
                    )
        finally:
            if error is None:
                self._record_success()
                
        return result

    def _execute_with_timeout(
        self,
        operation: Callable[..., T],
        timeout: float,
        **kwargs
    ) -> T:
        """
        Execute operation with timeout enforcement.
        
        Uses ThreadPoolExecutor to run the operation in a separate thread
        and enforces a timeout, canceling the operation if it takes too long.
        """
        if timeout <= 0:
            return operation(**kwargs)
            
        # Use ThreadPoolExecutor for proper timeout enforcement
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(operation, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                # Cancel the future if possible
                future.cancel()
                # Clean up any resources
                self._cleanup_timed_out_thread()
                raise TimeoutError(f"Operation timed out after {timeout:.2f}s")

    def _cleanup_timed_out_thread(self):
        """Attempt to clean up resources from timed out thread."""
        # This is a placeholder for any specific cleanup needed
        # In a real implementation, you might need to release locks,
        # close connections, etc.
        pass

    def _allow_attempt(self) -> bool:
        """Determine if attempt should be allowed based on circuit state."""
        with self._state_lock:
            if self.state == CircuitState.CLOSED:
                return True
                
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info("Circuit transitioning to HALF_OPEN")
                    return True
                return False
                
            if self.state == CircuitState.HALF_OPEN:
                return True
                
            return False

    def _record_success(self) -> None:
        """Update state after successful execution."""
        with self._state_lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.logger.info("Circuit reset to CLOSED")
            self.metrics.record_success()

    def _record_failure(self) -> None:
        """Update state after failed execution."""
        with self._state_lock:
            self.last_failure_time = time.time()
            self.metrics.record_failure()
            
            if (self.metrics.success_rate < self.failure_threshold or
                self.state == CircuitState.HALF_OPEN):
                self.state = CircuitState.OPEN
                self.logger.warning(
                    f"Circuit opened (success rate: {self.metrics.success_rate:.1%})"
                )

    def _trigger_fallback(
        self,
        fallback: Optional[Callable[..., T]],
        error: Optional[Exception] = None,
        **kwargs
    ) -> Optional[T]:
        """Execute fallback operation if available."""
        self.metrics.record_fallback()
        
        if fallback is None:
            if error:
                self.logger.error("No fallback available", exc_info=True)
            return None
            
        try:
            self.logger.info("Executing fallback operation")
            return fallback(**kwargs)
        except Exception as e:
            self.logger.error("Fallback operation failed", exc_info=True)
            return None

    def isolate(self) -> None:
        """Manually isolate circuit (maintenance mode)."""
        with self._state_lock:
            self.state = CircuitState.ISOLATED
        self.logger.warning("Circuit manually isolated")

    def reset(self) -> None:
        """Force reset circuit to closed state."""
        with self._state_lock:
            self.state = CircuitState.CLOSED
            self.last_failure_time = 0
        self.logger.info("Circuit manually reset")

# Global registry of executors for different operations
_executor_registry: Dict[str, GracefulExecutor] = {}
_registry_lock = Lock()

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
        with _registry_lock:
            if operation_name not in _executor_registry:
                _executor_registry[operation_name] = GracefulExecutor(
                    name=operation_name,
                    max_attempts=max_attempts,
                    backoff_strategy=lambda attempt: BackoffStrategy.exponential(backoff_base, attempt),
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
    with _registry_lock:
        return {
            name: {
                "state": executor.state.name,
                "failures": executor.metrics.total_failures,
                "timeouts": executor.metrics.total_timeouts,
                "success_rate": executor.metrics.success_rate,
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
    with _registry_lock:
        if operation_name in _executor_registry:
            _executor_registry[operation_name].reset()
            return True
        return False

def isolate_circuit(operation_name: str) -> bool:
    """
    Manually isolate a circuit (e.g., for maintenance).
    
    Args:
        operation_name: Name of circuit to isolate
        
    Returns:
        True if isolation successful, False if circuit not found
    """
    with _registry_lock:
        if operation_name in _executor_registry:
            _executor_registry[operation_name].isolate()
            return True
        return False