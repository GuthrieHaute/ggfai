"""
ComponentInterface Module - Base Interface for GGF AI Framework Components

This module defines the base interface that all major framework components must implement,
ensuring consistent lifecycle management and dependency injection support.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger("GGFAI.core_framework.component")

class ComponentStatus(Enum):
    """Status of a framework component."""
    UNINITIALIZED = auto()  # Not yet initialized
    INITIALIZING = auto()    # In the process of initializing
    ACTIVE = auto()         # Running normally
    DEGRADED = auto()       # Running with reduced functionality
    ERROR = auto()          # In error state
    SHUTTING_DOWN = auto()  # In the process of shutting down
    TERMINATED = auto()     # Fully shut down

@dataclass
class ComponentMetadata:
    """Metadata about a framework component."""
    id: str
    type: str
    version: str = "1.0.0"
    status: ComponentStatus = ComponentStatus.UNINITIALIZED
    dependencies: Dict[str, str] = field(default_factory=dict)
    config_schema: Optional[Dict] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

class ComponentInterface(ABC):
    """Base interface that all framework components must implement."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the component with provided configuration.
        
        Args:
            config: Configuration dictionary for this component
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def start(self) -> bool:
        """
        Start the component's main functionality.
        
        Returns:
            bool: True if start successful, False otherwise
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the component's functionality gracefully.
        
        Returns:
            bool: True if stop successful, False otherwise
        """
        pass

    @abstractmethod
    def get_status(self) -> ComponentStatus:
        """
        Get current status of the component.
        
        Returns:
            ComponentStatus: Current status enum value
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> ComponentMetadata:
        """
        Get component metadata.
        
        Returns:
            ComponentMetadata: Component metadata object
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            dict: Dictionary of metrics
        """
        pass

    @abstractmethod
    def validate_health(self) -> bool:
        """
        Validate component's health status.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    @abstractmethod
    def handle_error(self, error: Exception) -> None:
        """
        Handle component-specific error conditions.
        
        Args:
            error: The exception that occurred
        """
        pass

class BaseComponent(ComponentInterface):
    """
    Base implementation of ComponentInterface providing common functionality.
    
    Components can inherit from this instead of implementing ComponentInterface
    directly if they want the default implementations.
    """

    def __init__(self, component_id: str, component_type: str):
        """
        Initialize the base component.
        
        Args:
            component_id: Unique identifier for this component instance
            component_type: Type of component (e.g., "EntryPoint", "MLComponent")
        """
        self._metadata = ComponentMetadata(
            id=component_id,
            type=component_type
        )
        self._status = ComponentStatus.UNINITIALIZED
        self._metrics = {}
        self._logger = logging.getLogger(f"GGFAI.{component_type}.{component_id}")

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Default initialization implementation."""
        try:
            self._status = ComponentStatus.INITIALIZING
            # Subclasses should override this with their specific initialization
            self._status = ComponentStatus.ACTIVE
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize component: {str(e)}")
            self._status = ComponentStatus.ERROR
            return False

    def start(self) -> bool:
        """Default start implementation."""
        if self._status != ComponentStatus.ACTIVE:
            self._logger.error("Cannot start component - not in ACTIVE state")
            return False
        return True

    def stop(self) -> bool:
        """Default stop implementation."""
        try:
            self._status = ComponentStatus.SHUTTING_DOWN
            # Subclasses should override this with their specific cleanup
            self._status = ComponentStatus.TERMINATED
            return True
        except Exception as e:
            self._logger.error(f"Error during component shutdown: {str(e)}")
            self._status = ComponentStatus.ERROR
            return False

    def get_status(self) -> ComponentStatus:
        """Get current component status."""
        return self._status

    def get_metadata(self) -> ComponentMetadata:
        """Get component metadata."""
        return self._metadata

    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics."""
        return self._metrics.copy()

    def validate_health(self) -> bool:
        """Default health validation."""
        return self._status in [ComponentStatus.ACTIVE, ComponentStatus.DEGRADED]

    def handle_error(self, error: Exception) -> None:
        """Default error handling."""
        self._logger.error(f"Component error: {str(error)}")
        self._status = ComponentStatus.ERROR
        # Subclasses should override with specific error handling