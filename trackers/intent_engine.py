# intent_engine.py - Context-Aware Error Handling
# written by DeepSeek Chat (honor call: The Sentinel)

from enum import Enum, auto
from pydantic import BaseModel
from typing import Optional, Dict, Any
import redis
import logging

class ErrorSeverity(Enum):
    WARNING = auto()
    CRITICAL = auto()
    FATAL = auto()

class ErrorContext(BaseModel):
    component: str
    severity: ErrorSeverity
    timestamp: float
    tags: Dict[str, Any]
    recovery_suggestion: Optional[str] = None

class ContextMonitor:
    def __init__(self, redis_host: str = "localhost"):
        self.redis = redis.Redis(host=redis_host, decode_responses=True)
        self.logger = logging.getLogger("GGFAI.context")
        self.emergency_triggers = {
            "emergency_alert": self._handle_emergency,
            "resource_exhaustion": self._handle_resource_crisis,
            "invalid_state": self._handle_invalid_state
        }

    def check_context(self, intent: Dict[str, Any]) -> Optional[ErrorContext]:
        """Analyze intent stream for emergency contexts."""
        if not intent.get('tags'):
            return None

        # Check for emergency triggers
        for trigger, handler in self.emergency_triggers.items():
            if trigger in intent['tags']:
                return handler(intent)

        # Check resource constraints
        if intent.get('required_resources'):
            unavailable = [
                res for res in intent['required_resources']
                if not self.redis.hexists("available_resources", res)
            ]
            if unavailable:
                return ErrorContext(
                    component="ResourceManager",
                    severity=ErrorSeverity.CRITICAL,
                    timestamp=time.time(),
                    tags={"missing_resources": unavailable},
                    recovery_suggestion=f"Release {unavailable} or degrade functionality"
                )

        return None

    def _handle_emergency(self, intent: Dict) -> ErrorContext:
        """Emergency interrupt handler."""
        self.logger.critical(f"EMERGENCY: {intent['tags']['emergency_alert']}")
        return ErrorContext(
            component="SafetyMonitor",
            severity=ErrorSeverity.FATAL,
            timestamp=time.time(),
            tags=intent['tags'],
            recovery_suggestion="Immediate human intervention required"
        )

    def _handle_resource_crisis(self, intent: Dict) -> ErrorContext:
        """Resource exhaustion handler."""
        current_load = {
            'cpu': float(self.redis.hget("system_metrics", "cpu")),
            'mem': float(self.redis.hget("system_metrics", "mem"))
        }
        return ErrorContext(
            component="ResourceManager",
            severity=ErrorSeverity.CRITICAL,
            timestamp=time.time(),
            tags={**intent['tags'], **current_load},
            recovery_suggestion="Activate garbage collection or kill non-critical processes"
        )

    def _handle_invalid_state(self, intent: Dict) -> ErrorContext:
        """State validation handler."""
        return ErrorContext(
            component="StateValidator",
            severity=ErrorSeverity.WARNING,
            timestamp=time.time(),
            tags=intent['tags'],
            recovery_suggestion="Rollback to last valid checkpoint"
        )