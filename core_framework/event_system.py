"""
EventSystem Module - Asynchronous event handling for GGF AI Framework

This module provides a centralized event system for decoupled inter-component
communication using asyncio and prioritized event queues.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import queue
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("GGFAI.core_framework.events")

class EventPriority(Enum):
    """Priority levels for event handling."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Event:
    """Base event class."""
    type: str
    source: str
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    _handled: bool = False

    def __lt__(self, other):
        """Compare events by priority for queue ordering."""
        if not isinstance(other, Event):
            return NotImplemented
        return self.priority.value > other.priority.value  # Higher priority first

@dataclass
class StateChangeEvent(Event):
    """Event indicating a component state change."""
    old_state: Any
    new_state: Any

@dataclass
class ErrorEvent(Event):
    """Event indicating an error condition."""
    error: Exception
    severity: str = "ERROR"

class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Handle an event asynchronously."""
        pass

    @property
    @abstractmethod
    def supported_events(self) -> Set[str]:
        """Get set of event types this handler supports."""
        pass

class EventSystem:
    """
    Centralized event handling system.
    
    Features:
    - Asynchronous event processing
    - Priority-based event queues
    - Event filtering and routing
    - Handler registration/unregistration
    - Error handling and recovery
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize the event system.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self._handlers: Dict[str, List[weakref.ref]] = {}
        self._queue = asyncio.PriorityQueue()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        self._event_buffer: List[Event] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Start event processing."""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Start processing queued events
        self._processing_task = asyncio.create_task(self._process_events())
        
        # Process any buffered events
        for event in self._event_buffer:
            await self.publish_event(event)
        self._event_buffer.clear()
        
        logger.info("Event system started")

    async def stop(self) -> None:
        """Stop event processing gracefully."""
        if not self._running:
            return

        self._running = False
        if self._processing_task:
            await self._processing_task
            self._processing_task = None
        
        self._executor.shutdown(wait=True)
        logger.info("Event system stopped")

    def register_handler(
        self,
        handler: EventHandler,
        event_types: Optional[List[str]] = None
    ) -> None:
        """
        Register an event handler.
        
        Args:
            handler: Handler implementing EventHandler interface
            event_types: Optional list of event types to handle (defaults to handler.supported_events)
        """
        with self._lock:
            handler_ref = weakref.ref(handler)
            types = event_types or list(handler.supported_events)
            
            for event_type in types:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []
                self._handlers[event_type].append(handler_ref)
            
            logger.debug(f"Registered handler for events: {types}")

    def unregister_handler(
        self,
        handler: EventHandler,
        event_types: Optional[List[str]] = None
    ) -> None:
        """
        Unregister an event handler.
        
        Args:
            handler: Handler to unregister
            event_types: Optional list of event types to unregister (defaults to all)
        """
        with self._lock:
            types = event_types or list(self._handlers.keys())
            
            for event_type in types:
                if event_type in self._handlers:
                    self._handlers[event_type] = [
                        h for h in self._handlers[event_type]
                        if h() is not None and h() is not handler
                    ]
                    
                    if not self._handlers[event_type]:
                        del self._handlers[event_type]
            
            logger.debug(f"Unregistered handler for events: {types}")

    async def publish_event(self, event: Event) -> None:
        """
        Publish an event for processing.
        
        Args:
            event: Event to publish
            
        If the event system isn't running, events will be buffered.
        """
        if not self._running:
            self._event_buffer.append(event)
            return
        
        await self._queue.put(event)
        logger.debug(f"Published event: {event.type}")

    def publish_event_sync(self, event: Event) -> None:
        """
        Synchronous version of publish_event for non-async contexts.
        
        Args:
            event: Event to publish
        """
        if self._loop is None:
            self._event_buffer.append(event)
            return
            
        asyncio.run_coroutine_threadsafe(
            self.publish_event(event),
            self._loop
        )

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await self._queue.get()
                await self._handle_event(event)
                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
                # Publish error event
                error_event = ErrorEvent(
                    type="event_processing_error",
                    source="event_system",
                    error=e
                )
                await self.publish_event(error_event)

    async def _handle_event(self, event: Event) -> None:
        """
        Handle a single event.
        
        Args:
            event: Event to handle
        """
        with self._lock:
            handlers = self._handlers.get(event.type, [])
            handlers = [h for h in handlers if h() is not None]
            self._handlers[event.type] = handlers
        
        if not handlers:
            logger.debug(f"No handlers for event type: {event.type}")
            return
        
        # Handle event in all registered handlers
        tasks = []
        for handler_ref in handlers:
            handler = handler_ref()
            if handler:
                task = asyncio.create_task(handler.handle_event(event))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            event._handled = True

    def get_handler_count(self, event_type: str) -> int:
        """
        Get number of handlers for an event type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            Number of registered handlers
        """
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            return len([h for h in handlers if h() is not None])

    def get_queue_size(self) -> int:
        """
        Get current size of event queue.
        
        Returns:
            Number of events in queue
        """
        return self._queue.qsize()

class EventHandlerMixin:
    """
    Mixin class to easily add event handling capabilities to components.
    
    Components can inherit from this mixin to easily implement event handling.
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        """
        Initialize the event handler mixin.
        
        Args:
            event_system: Optional event system to register with
        """
        self._event_system = event_system
        self._supported_events: Set[str] = set()
        
    def register_event_types(self, event_types: List[str]) -> None:
        """
        Register event types this component can handle.
        
        Args:
            event_types: List of event type strings
        """
        self._supported_events.update(event_types)
        if self._event_system:
            self._event_system.register_handler(self, event_types)
    
    def unregister_event_types(self, event_types: Optional[List[str]] = None) -> None:
        """
        Unregister event types.
        
        Args:
            event_types: Optional list of types to unregister (defaults to all)
        """
        types = event_types or list(self._supported_events)
        self._supported_events.difference_update(types)
        if self._event_system:
            self._event_system.unregister_handler(self, types)
    
    @property
    def supported_events(self) -> Set[str]:
        """Get set of supported event types."""
        return self._supported_events.copy()
    
    async def handle_event(self, event: Event) -> None:
        """
        Default event handler implementation.
        
        Args:
            event: Event to handle
            
        Components should override this method.
        """
        pass