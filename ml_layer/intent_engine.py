# intent_engine.py - Hardened Intent Processing Core
# written by DeepSeek Chat (honor call: The Sentinel)

from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple, Callable
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor
import threading

from ..core.tag_registry import Tag
from ..core.run_with_grace import run_with_grace

# Constants
MAX_PROCESSING_TIME = 0.5  # seconds
THREAD_POOL_SIZE = 4

class ErrorSeverity(Enum):
    """Severity levels for intent processing errors."""
    WARNING = auto()
    CRITICAL = auto()
    FATAL = auto()

class IntentStage(Enum):
    """Processing stages for intent pipeline."""
    RAW_INPUT = auto()
    PARSED = auto()
    VALIDATED = auto()
    EXECUTABLE = auto()

class ErrorContext:
    """Context information for intent processing errors."""
    def __init__(self,
                component: str,
                severity: ErrorSeverity,
                tags: Dict[str, Any],
                recovery_suggestion: Optional[str] = None,
                stage: IntentStage = IntentStage.RAW_INPUT):
        """
        Initialize error context.
        
        Args:
            component: Component that generated the error
            severity: Error severity level
            tags: Additional context tags
            recovery_suggestion: Suggested recovery action
            stage: Processing stage where error occurred
        """
        self.component = component
        self.severity = severity
        self.timestamp = time.time()
        self.tags = tags
        self.recovery_suggestion = recovery_suggestion
        self.stage = stage
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component": self.component,
            "severity": self.severity.name,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "recovery_suggestion": self.recovery_suggestion,
            "stage": self.stage.name
        }

class IntentEngine:
    """
    Hardened intent processing system with:
    - Multi-stage validation pipeline
    - Error handling and recovery
    - Resource monitoring
    - Emergency protocols
    """
    
    def __init__(self, model_adapter=None):
        """
        Initialize intent engine.
        
        Args:
            model_adapter: Optional model adapter for ML processing
        """
        self.logger = logging.getLogger("GGFAI.intent_engine")
        self.executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
        self.model_adapter = model_adapter
        self.emergency_handlers = {
            "emergency_alert": self._handle_emergency,
            "resource_exhaustion": self._handle_resource_crisis,
            "invalid_state": self._handle_invalid_state
        }
        self._lock = threading.RLock()
        self.active_intents = {}  # intent_id -> status
        self.logger.info("Intent engine initialized")

    @run_with_grace(operation_name="process_intent", max_attempts=3)
    def process_intent(self, 
                      input_text: str, 
                      source: str = "user",
                      context: Dict[str, Any] = None) -> Tuple[Optional[Tag], Optional[ErrorContext]]:
        """
        Process raw input into structured intent.
        
        Args:
            input_text: Raw input text
            source: Source of the input
            context: Additional context information
            
        Returns:
            Tuple of (intent tag, error context)
        """
        start_time = time.time()
        intent_id = f"{int(start_time)}_{source}"
        
        try:
            self.logger.info(f"Processing intent from {source}: {input_text[:50]}...")
            
            # Track active intent
            with self._lock:
                self.active_intents[intent_id] = "processing"
            
            # Stage 1: Context Validation
            context = context or {}
            error_ctx = self._validate_context(context)
            if error_ctx and error_ctx.severity == ErrorSeverity.FATAL:
                return None, error_ctx

            # Stage 2: Resource Check
            if not self._check_resources(context):
                return None, ErrorContext(
                    component="ResourceManager",
                    severity=ErrorSeverity.CRITICAL,
                    tags={"input": input_text, "source": source},
                    recovery_suggestion="Scale resources or degrade features",
                    stage=IntentStage.VALIDATED
                )

            # Stage 3: Intent Processing
            intent_tag = self._execute_processing(input_text, source, context)
            
            # Check processing time
            processing_time = time.time() - start_time
            if processing_time > MAX_PROCESSING_TIME:
                self.logger.warning(f"Intent processing timeout: {processing_time:.2f}s > {MAX_PROCESSING_TIME}s")
                intent_tag.metadata["processing_time_warning"] = True

            # Update status
            with self._lock:
                self.active_intents[intent_id] = "completed"
                
            return intent_tag, None

        except Exception as e:
            self.logger.error(f"Processing failed: {e}", exc_info=True)
            
            # Update status
            with self._lock:
                self.active_intents[intent_id] = "failed"
                
            return None, ErrorContext(
                component="IntentEngine",
                severity=ErrorSeverity.CRITICAL,
                tags={"error": str(e), "input": input_text, "source": source},
                recovery_suggestion="Retry with degraded mode",
                stage=IntentStage.PARSED
            )

    def _validate_context(self, context: Dict) -> Optional[ErrorContext]:
        """
        Multi-layered context validation.
        
        Args:
            context: Context information
            
        Returns:
            Error context if validation fails, None otherwise
        """
        if not context:
            return None

        # Check for emergency triggers
        for trigger, handler in self.emergency_handlers.items():
            if trigger in context:
                return handler(context)

        return None

    def _check_resources(self, context: Dict) -> bool:
        """
        Check resource availability.
        
        Args:
            context: Context information
            
        Returns:
            True if resources available, False otherwise
        """
        # Simple implementation - can be extended with actual resource checks
        return True

    def _execute_processing(self, 
                           input_text: str, 
                           source: str,
                           context: Dict) -> Tag:
        """
        Core intent processing logic.
        
        Args:
            input_text: Raw input text
            source: Source of the input
            context: Additional context information
            
        Returns:
            Processed intent tag
        """
        # Use model adapter if available
        if self.model_adapter:
            try:
                intent_data = self.model_adapter.predict(input_text, context)
                return self._create_intent_tag(intent_data, source)
            except Exception as e:
                self.logger.error(f"Model adapter failed: {e}", exc_info=True)
                # Fall back to basic processing
        
        # Basic intent extraction (placeholder)
        # In a real implementation, this would use NLP techniques
        intent_name = self._extract_basic_intent(input_text)
        
        # Create intent tag
        return Tag(
            name=f"intent_{int(time.time())}",
            intent=intent_name,
            category="user_intent",
            subcategory=source,
            priority=0.5,  # Default medium priority
            metadata={
                "raw_input": input_text,
                "source": source,
                "timestamp": time.time(),
                "context": context
            }
        )

    def _extract_basic_intent(self, text: str) -> str:
        """
        Basic intent extraction from text.
        
        Args:
            text: Input text
            
        Returns:
            Extracted intent name
        """
        # Very basic keyword matching
        text = text.lower()
        
        if any(word in text for word in ["play", "music", "song", "listen"]):
            return "play_music"
        elif any(word in text for word in ["light", "dim", "bright"]):
            return "control_lights"
        elif any(word in text for word in ["temperature", "hot", "cold", "warm"]):
            return "adjust_temperature"
        elif any(word in text for word in ["remind", "reminder", "remember"]):
            return "set_reminder"
        elif any(word in text for word in ["timer", "alarm", "wake"]):
            return "set_timer"
        elif any(word in text for word in ["weather", "forecast", "rain"]):
            return "check_weather"
        elif any(word in text for word in ["news", "headline"]):
            return "get_news"
        elif any(word in text for word in ["search", "find", "look up"]):
            return "web_search"
        else:
            return "general_query"

    def _create_intent_tag(self, intent_data: Dict[str, Any], source: str) -> Tag:
        """
        Create intent tag from model output.
        
        Args:
            intent_data: Intent data from model
            source: Source of the input
            
        Returns:
            Intent tag
        """
        return Tag(
            name=f"intent_{int(time.time())}",
            intent=intent_data.get("intent", "unknown"),
            category="user_intent",
            subcategory=source,
            priority=intent_data.get("confidence", 0.5),
            metadata={
                "raw_input": intent_data.get("text", ""),
                "source": source,
                "timestamp": time.time(),
                "entities": intent_data.get("entities", {}),
                "confidence": intent_data.get("confidence", 0.5)
            }
        )

    def _handle_emergency(self, context: Dict) -> ErrorContext:
        """
        Handle emergency alerts.
        
        Args:
            context: Context information
            
        Returns:
            Error context
        """
        self.logger.critical(f"EMERGENCY: {context.get('emergency_alert')}")
        return ErrorContext(
            component="SafetyMonitor",
            severity=ErrorSeverity.FATAL,
            tags=context,
            recovery_suggestion="Immediate human intervention required",
            stage=IntentStage.RAW_INPUT
        )

    def _handle_resource_crisis(self, context: Dict) -> ErrorContext:
        """
        Handle resource exhaustion.
        
        Args:
            context: Context information
            
        Returns:
            Error context
        """
        return ErrorContext(
            component="ResourceManager",
            severity=ErrorSeverity.CRITICAL,
            tags=context,
            recovery_suggestion="Activate garbage collection protocol",
            stage=IntentStage.VALIDATED
        )

    def _handle_invalid_state(self, context: Dict) -> ErrorContext:
        """
        Handle invalid state.
        
        Args:
            context: Context information
            
        Returns:
            Error context
        """
        return ErrorContext(
            component="StateValidator",
            severity=ErrorSeverity.WARNING,
            tags=context,
            recovery_suggestion="Execute rollback protocol",
            stage=IntentStage.EXECUTABLE
        )

    def get_active_intents(self) -> Dict[str, str]:
        """
        Get active intents and their status.
        
        Returns:
            Dictionary of intent_id -> status
        """
        with self._lock:
            return self.active_intents.copy()

    def log_uncertain_predictions(self, text: str, confidence: float):
        """
        Log uncertain predictions for active learning.
        
        Args:
            text: Input text
            confidence: Prediction confidence
        """
        if confidence < 0.7:  # Flag for manual review
            try:
                with open("data/active_learning.txt", "a") as f:
                    f.write(f"{text}\n")
                self.logger.info(f"Logged uncertain prediction: {text[:50]}... ({confidence:.2f})")
            except Exception as e:
                self.logger.error(f"Failed to log uncertain prediction: {e}")

    def set_model_adapter(self, adapter):
        """
        Set model adapter for intent processing.
        
        Args:
            adapter: Model adapter instance
        """
        self.model_adapter = adapter
        self.logger.info(f"Set model adapter: {adapter.__class__.__name__}")

    def register_emergency_handler(self, trigger: str, handler: Callable):
        """
        Register custom emergency handler.
        
        Args:
            trigger: Trigger keyword
            handler: Handler function
        """
        self.emergency_handlers[trigger] = handler
        self.logger.info(f"Registered emergency handler for '{trigger}'")