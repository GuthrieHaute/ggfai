"""
DependencyInjector Module - Manages component dependencies and initialization

This module provides centralized dependency injection and component lifecycle management 
for the GGF AI Framework. It ensures components are initialized in the correct order
based on their dependencies, focusing on the core entry points and component system.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Type
import threading
from dataclasses import dataclass
from enum import Enum, auto
from .component_interface import ComponentInterface, ComponentStatus

logger = logging.getLogger("GGFAI.core_framework.injector")

class InjectorError(Exception):
    """Base exception for dependency injector errors."""
    pass

class DependencyError(InjectorError):
    """Exception for dependency resolution errors."""
    pass

class CircularDependencyError(DependencyError):
    """Exception for circular dependency detection."""
    pass

class ComponentRegistrationError(InjectorError):
    """Exception for component registration errors."""
    pass

@dataclass
class DependencyMetadata:
    """Metadata about a component dependency."""
    required_type: str
    component_id: Optional[str] = None
    is_optional: bool = False
    config: Dict[str, Any] = None

class DependencyInjector:
    """
    Centralized dependency injection system for the GGF AI Framework.
    
    Manages component lifecycle, dependency resolution, and initialization ordering
    for the framework's entry points and core components.
    """

    def __init__(self):
        """Initialize the dependency injector."""
        self._components: Dict[str, ComponentInterface] = {}
        self._component_types: Dict[str, Type[ComponentInterface]] = {}
        self._dependencies: Dict[str, List[DependencyMetadata]] = {}
        self._initialized: Set[str] = set()
        self._lock = threading.RLock()
        self._initialization_order: List[str] = []

    def register_component_type(self, type_name: str, component_class: Type[ComponentInterface]) -> None:
        """
        Register a component type for future instantiation.
        
        Args:
            type_name: String identifier for the component type
            component_class: The component class that implements ComponentInterface
            
        Raises:
            ComponentRegistrationError: If type already registered or invalid
        """
        with self._lock:
            if type_name in self._component_types:
                raise ComponentRegistrationError(f"Component type {type_name} already registered")
            
            if not issubclass(component_class, ComponentInterface):
                raise ComponentRegistrationError(
                    f"Component class {component_class.__name__} must implement ComponentInterface"
                )
            
            self._component_types[type_name] = component_class
            logger.info(f"Registered component type: {type_name}")

    def register_instance(
        self,
        component: ComponentInterface,
        dependencies: Optional[List[DependencyMetadata]] = None
    ) -> None:
        """
        Register an existing component instance.
        
        Args:
            component: Instance implementing ComponentInterface
            dependencies: Optional list of dependencies for this component
            
        Raises:
            ComponentRegistrationError: If component ID already registered
        """
        with self._lock:
            metadata = component.get_metadata()
            component_id = metadata.id
            
            if component_id in self._components:
                raise ComponentRegistrationError(f"Component ID {component_id} already registered")
            
            self._components[component_id] = component
            if dependencies:
                self._dependencies[component_id] = dependencies
            
            logger.info(f"Registered component instance: {component_id} ({metadata.type})")

    def create_component(
        self,
        type_name: str,
        component_id: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[DependencyMetadata]] = None
    ) -> ComponentInterface:
        """
        Create and register a new component instance.
        
        Args:
            type_name: Registered component type name
            component_id: Unique ID for the new instance
            config: Optional configuration for the component
            dependencies: Optional list of dependencies
            
        Returns:
            The created component instance
            
        Raises:
            ComponentRegistrationError: If creation fails
        """
        with self._lock:
            if component_id in self._components:
                raise ComponentRegistrationError(f"Component ID {component_id} already exists")
            
            if type_name not in self._component_types:
                raise ComponentRegistrationError(f"Unknown component type: {type_name}")
            
            try:
                component_class = self._component_types[type_name]
                component = component_class(component_id=component_id)
                
                if dependencies:
                    self._dependencies[component_id] = dependencies
                
                if config:
                    component.initialize(config)
                
                self._components[component_id] = component
                logger.info(f"Created component: {component_id} ({type_name})")
                return component
                
            except Exception as e:
                raise ComponentRegistrationError(f"Failed to create component: {str(e)}")

    def get_component(self, component_id: str) -> Optional[ComponentInterface]:
        """
        Get a registered component by ID.
        
        Args:
            component_id: ID of the component to retrieve
            
        Returns:
            The component instance or None if not found
        """
        return self._components.get(component_id)

    def get_components_by_type(self, type_name: str) -> List[ComponentInterface]:
        """
        Get all registered components of a specific type.
        
        Args:
            type_name: Type name to filter by
            
        Returns:
            List of matching component instances
        """
        return [
            component for component in self._components.values()
            if component.get_metadata().type == type_name
        ]

    def _resolve_dependencies(self, component_id: str, visited: Set[str]) -> List[str]:
        """
        Resolve initialization order for a component and its dependencies.
        
        Args:
            component_id: Component to resolve dependencies for
            visited: Set of already visited components (for cycle detection)
            
        Returns:
            List of component IDs in initialization order
            
        Raises:
            CircularDependencyError: If a dependency cycle is detected
        """
        if component_id in visited:
            raise CircularDependencyError(
                f"Circular dependency detected involving {component_id}"
            )
        
        visited.add(component_id)
        order = []
        
        # Process dependencies first
        if component_id in self._dependencies:
            for dep in self._dependencies[component_id]:
                if dep.component_id and dep.component_id not in self._initialized:
                    dep_order = self._resolve_dependencies(dep.component_id, visited.copy())
                    for dep_id in dep_order:
                        if dep_id not in order:
                            order.append(dep_id)
        
        # Add this component
        if component_id not in order:
            order.append(component_id)
        
        return order

    def initialize_components(self) -> bool:
        """
        Initialize all registered components in dependency order.
        
        Returns:
            bool: True if all components initialized successfully
            
        Raises:
            DependencyError: If initialization fails
        """
        with self._lock:
            try:
                # Resolve full initialization order
                remaining = set(self._components.keys()) - self._initialized
                order = []
                
                for component_id in remaining:
                    if component_id not in order:
                        component_order = self._resolve_dependencies(component_id, set())
                        for cid in component_order:
                            if cid not in order:
                                order.append(cid)
                
                # Initialize components
                for component_id in order:
                    if component_id not in self._initialized:
                        component = self._components[component_id]
                        
                        # Inject dependencies if needed
                        if component_id in self._dependencies:
                            for dep in self._dependencies[component_id]:
                                if dep.component_id:
                                    dependency = self._components[dep.component_id]
                                    # Component should implement method to receive dependency
                                    if hasattr(component, f"set_{dep.required_type}"):
                                        getattr(component, f"set_{dep.required_type}")(dependency)
                        
                        # Start the component
                        if not component.start():
                            raise DependencyError(
                                f"Failed to start component {component_id}"
                            )
                        
                        self._initialized.add(component_id)
                        logger.info(f"Initialized component: {component_id}")
                
                self._initialization_order = order
                return True
                
            except Exception as e:
                logger.error(f"Component initialization failed: {str(e)}")
                raise DependencyError(f"Initialization failed: {str(e)}")

    def shutdown_components(self) -> bool:
        """
        Gracefully shut down all components in reverse initialization order.
        
        Returns:
            bool: True if all components shut down successfully
        """
        with self._lock:
            success = True
            # Shutdown in reverse initialization order
            for component_id in reversed(self._initialization_order):
                try:
                    component = self._components[component_id]
                    if not component.stop():
                        logger.error(f"Failed to stop component: {component_id}")
                        success = False
                    else:
                        logger.info(f"Stopped component: {component_id}")
                except Exception as e:
                    logger.error(f"Error stopping component {component_id}: {str(e)}")
                    success = False
            
            self._initialized.clear()
            return success

    def get_initialization_order(self) -> List[str]:
        """
        Get the component initialization order.
        
        Returns:
            List of component IDs in initialization order
        """
        return self._initialization_order.copy()

    def validate_dependencies(self) -> bool:
        """
        Validate that all component dependencies are satisfied.
        
        Returns:
            bool: True if all dependencies are valid
        """
        with self._lock:
            for component_id, dependencies in self._dependencies.items():
                for dep in dependencies:
                    if dep.component_id:
                        if dep.component_id not in self._components:
                            if not dep.is_optional:
                                logger.error(
                                    f"Missing required dependency {dep.component_id} "
                                    f"for component {component_id}"
                                )
                                return False
            return True