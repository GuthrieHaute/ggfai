"""
Adaptive task scheduler with resource awareness and backpressure handling
"""
from typing import Dict, List, Optional, Set
import time
import logging
from threading import Lock
from dataclasses import dataclass, field
from enum import Enum
import heapq

from ..core.tag_registry import TagRegistry
from .hardware_shim import HardwareMonitor
from .proactive_anomaly_detection import ProactiveAnomalyDetector

logger = logging.getLogger("GGFAI.scheduler")

class TaskPriority(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1

@dataclass(order=True)
class ScheduledTask:
    priority: int
    timestamp: float
    task_id: str = field(compare=False)
    resource_demand: Dict[str, float] = field(compare=False)
    timeout: Optional[float] = field(compare=False, default=None)
    dependencies: Set[str] = field(compare=False, default_factory=set)
    _lock: Lock = field(init=False, repr=False, compare=False)
    
    def __post_init__(self):
        self._lock = Lock()
        
    def is_expired(self) -> bool:
        if self.timeout is None:
            return False
        return time.time() > self.timeout

class AdaptiveScheduler:
    def __init__(
        self,
        tag_registry: TagRegistry,
        hardware_monitor: Optional[HardwareMonitor] = None,
        anomaly_detector: Optional[ProactiveAnomalyDetector] = None
    ):
        self.tag_registry = tag_registry
        self.hardware_monitor = hardware_monitor or HardwareMonitor()
        self.anomaly_detector = anomaly_detector
        self._task_queue: List[ScheduledTask] = []  # Priority queue
        self._in_progress: Dict[str, ScheduledTask] = {}
        self._lock = Lock()
        self._resource_thresholds = {
            'cpu_high': 80.0,  # %
            'mem_high': 85.0,  # %
            'ready_tasks_limit': 50
        }
        
    def submit_task(
        self,
        task_id: str,
        priority: TaskPriority,
        resource_demand: Dict[str, float],
        timeout: Optional[float] = None,
        dependencies: Optional[Set[str]] = None
    ) -> bool:
        """Submit task for scheduling."""
        task = ScheduledTask(
            priority=priority.value,
            timestamp=time.time(),
            task_id=task_id,
            resource_demand=resource_demand,
            timeout=timeout,
            dependencies=dependencies or set()
        )
        
        with self._lock:
            # Check if system is overloaded
            if self._is_system_overloaded():
                if priority != TaskPriority.CRITICAL:
                    logger.warning(f"System overloaded, rejecting task {task_id}")
                    return False
                    
            # Add to queue and maintain heap invariant
            heapq.heappush(self._task_queue, task)
            logger.info(f"Task {task_id} queued with priority {priority.name}")
            return True
            
    def get_next_task(self) -> Optional[ScheduledTask]:
        """Get next task that is ready to execute."""
        with self._lock:
            while self._task_queue:
                # Get highest priority task
                task = heapq.heappop(self._task_queue)
                
                # Skip if expired
                if task.is_expired():
                    logger.warning(f"Task {task.task_id} expired before execution")
                    continue
                    
                # Skip if dependencies not met
                if not self._are_dependencies_met(task):
                    heapq.heappush(self._task_queue, task)  # Put back in queue
                    continue
                    
                # Skip if insufficient resources
                if not self._has_sufficient_resources(task):
                    heapq.heappush(self._task_queue, task)  # Put back in queue
                    return None
                    
                # Track in-progress task
                self._in_progress[task.task_id] = task
                return task
                
        return None
        
    def complete_task(self, task_id: str, success: bool = True) -> None:
        """Mark task as completed and update scheduler state."""
        with self._lock:
            if task_id in self._in_progress:
                task = self._in_progress.pop(task_id)
                logger.info(f"Task {task_id} completed with success={success}")
                
                # Log completion for analytics
                self.tag_registry.add_tag({
                    "type": "task_completion",
                    "task_id": task_id,
                    "success": success,
                    "duration": time.time() - task.timestamp,
                    "priority": TaskPriority(task.priority).name
                })
                
    def _is_system_overloaded(self) -> bool:
        """Check if system is experiencing resource pressure."""
        # Check hardware metrics
        cpu_load = self.hardware_monitor.get_cpu_load()
        mem_info = self.hardware_monitor.get_memory_info()
        
        if (cpu_load > self._resource_thresholds['cpu_high'] or
            mem_info['percent_used'] > self._resource_thresholds['mem_high']):
            return True
            
        # Check anomaly detector if available
        if self.anomaly_detector:
            alerts = self.anomaly_detector.detect_anomalies()
            if any('critical' in alert_type for alert_type in alerts):
                return True
                
        # Check queue size
        if len(self._task_queue) > self._resource_thresholds['ready_tasks_limit']:
            return True
            
        return False
        
    def _are_dependencies_met(self, task: ScheduledTask) -> bool:
        """Check if all task dependencies are completed."""
        return all(
            dep_id not in self._in_progress 
            for dep_id in task.dependencies
        )
        
    def _has_sufficient_resources(self, task: ScheduledTask) -> bool:
        """Check if required resources are available."""
        try:
            # Get current resource usage
            cpu_available = 100 - self.hardware_monitor.get_cpu_load()
            mem_available = self.hardware_monitor.get_available_memory()
            
            # Check against task demands
            if 'cpu' in task.resource_demand:
                if task.resource_demand['cpu'] > cpu_available:
                    return False
                    
            if 'memory' in task.resource_demand:
                if task.resource_demand['memory'] > mem_available:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking resources: {str(e)}")
            return False
            
    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about task queue."""
        with self._lock:
            priority_counts = {p.name: 0 for p in TaskPriority}
            for task in self._task_queue:
                priority_counts[TaskPriority(task.priority).name] += 1
                
            return {
                'total_queued': len(self._task_queue),
                'in_progress': len(self._in_progress),
                **priority_counts
            }
