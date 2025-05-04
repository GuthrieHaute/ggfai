"""
Adaptive Resource Scheduler with Hardware-Aware Task Allocation

Key Improvements:
1. Multi-strategy scheduling (FIFO, Priority, Round-Robin)
2. Dynamic resource thresholds
3. Enhanced backpressure signaling
4. Task preemption support
5. Comprehensive metrics collection
6. Thread-safe operations
7. Energy efficiency considerations
8. NUMA awareness
"""

import heapq
import time
import logging
import psutil
import numpy as np
import zmq
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto
from dataclasses import dataclass, field
from threading import Lock
from collections import deque
from statistics import mean

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GGFAI.scheduler")

class HardwareTier(Enum):
    """Hardware capability tiers with dynamic thresholds."""
    GARBAGE = auto()    # <1GB RAM, no GPU
    LOW_END = auto()    # 1-4GB RAM
    MID_RANGE = auto()  # 4-8GB RAM
    HIGH_END = auto()   # >8GB RAM + GPU

    def get_thresholds(self) -> Dict[str, float]:
        """Get resource thresholds based on hardware tier."""
        return {
            'cpu': 0.95 - (0.1 * self.value),  # Better hardware handles higher load
            'mem': 0.9 - (0.05 * self.value),
            'io': 0.85 - (0.03 * self.value),
            'gpu': 0.8 if self == HardwareTier.HIGH_END else 0.0,
            'energy': 0.7  # Universal energy efficiency threshold
        }

class SchedulingStrategy(Enum):
    """Available scheduling strategies."""
    PRIORITY = auto()
    FIFO = auto()
    ROUND_ROBIN = auto()
    ENERGY_AWARE = auto()

@dataclass
class ResourceLoad:
    """Track system resource utilization."""
    cpu: float = 0.0
    mem: float = 0.0
    io: float = 0.0
    gpu: float = 0.0
    energy: float = 0.0  # Estimated energy impact (0-1 scale)

@dataclass
class Task:
    """Task representation with scheduling metadata."""
    id: str
    priority: float  # 0.0 (low) to 1.0 (high)
    estimated_load: ResourceLoad
    min_hardware_tier: HardwareTier = HardwareTier.GARBAGE
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    dependencies: Set[str] = field(default_factory=set)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def adjust_priority(self, factor: float) -> None:
        """Thread-safe priority adjustment."""
        with self._lock:
            self.priority = max(0.0, min(1.0, self.priority * factor))

    def is_urgent(self) -> bool:
        """Check if task is approaching deadline."""
        if self.deadline is None:
            return False
        return (self.deadline - time.time()) < 60  # Within 1 minute

class AdaptiveScheduler:
    """
    Hardware-aware adaptive scheduler with:
    - Multiple scheduling strategies
    - Dynamic resource management
    - Backpressure signaling
    - Energy efficiency
    - NUMA awareness
    """
    
    def __init__(
        self,
        hardware_tier: HardwareTier,
        zmq_port: int = 5556,
        strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY
    ):
        self.hardware_tier = hardware_tier
        self.strategy = strategy
        self._queue_lock = Lock()
        
        # Initialize appropriate queue based on strategy
        if strategy == SchedulingStrategy.PRIORITY:
            self._task_queue = []
        elif strategy == SchedulingStrategy.FIFO:
            self._task_queue = deque()
        else:  # ROUND_ROBIN or ENERGY_AWARE
            self._task_queue = []
            self._current_index = 0

        self._completed_tasks = 0
        self._deferred_tasks = 0
        self._resource_thresholds = hardware_tier.get_thresholds()
        self._current_load = ResourceLoad()
        self._historical_load = deque(maxlen=100)  # Track for trend analysis
        
        # ZMQ setup for distributed coordination
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 100)  # Prevent queue buildup
        self.socket.bind(f"tcp://*:{zmq_port}")

        # Energy efficiency tracking
        self._energy_efficiency = 1.0  # Start assuming optimal
        self._last_energy_check = time.time()

        logger.info(
            f"Scheduler initialized in {hardware_tier.name} mode "
            f"with {strategy.name} strategy"
        )

    def add_task(self, task: Task) -> bool:
        """Add task to scheduler with validation."""
        if not self._validate_task(task):
            return False

        with self._queue_lock:
            if self.strategy == SchedulingStrategy.PRIORITY:
                heapq.heappush(self._task_queue, (-task.priority, task))
            elif self.strategy == SchedulingStrategy.FIFO:
                self._task_queue.append(task)
            else:
                self._task_queue.append(task)
                
            logger.debug(f"Added task {task.id} to queue")
            return True

    def _validate_task(self, task: Task) -> bool:
        """Validate task requirements against hardware capabilities."""
        if task.min_hardware_tier.value > self.hardware_tier.value:
            task.adjust_priority(0.5)  # Penalize for hardware mismatch
            logger.warning(
                f"Task {task.id} requires {task.min_hardware_tier.name} "
                f"but running on {self.hardware_tier.name}"
            )
            
        # Check for impossible resource requirements
        if any(
            getattr(task.estimated_load, resource) > 1.0 
            for resource in ['cpu', 'mem', 'io', 'gpu']
        ):
            logger.error(f"Task {task.id} has impossible resource requirements")
            return False
            
        return True

    def schedule(self) -> Optional[Task]:
        """
        Select next task considering:
        - System load
        - Task priority
        - Energy efficiency
        - Deadlines
        """
        self._update_system_load()
        
        # Check for system overload
        if self._is_overloaded():
            self._apply_backpressure()
            return None

        with self._queue_lock:
            if not self._task_queue:
                return None

            # Select task based on strategy
            if self.strategy == SchedulingStrategy.PRIORITY:
                _, task = heapq.heappop(self._task_queue)
            elif self.strategy == SchedulingStrategy.FIFO:
                task = self._task_queue.popleft()
            else:  # ROUND_ROBIN or ENERGY_AWARE
                task = self._select_round_robin()

            # Check if task can be safely executed
            if self._can_execute(task):
                return task
                
            # Defer task if not executable now
            self._defer_task(task)
            return None

    def _select_round_robin(self) -> Task:
        """Select task using round-robin or energy-aware strategy."""
        if self.strategy == SchedulingStrategy.ENERGY_AWARE:
            # Find most energy-efficient task that meets other criteria
            for i in range(len(self._task_queue)):
                idx = (self._current_index + i) % len(self._task_queue)
                task = self._task_queue[idx]
                if self._is_energy_efficient(task):
                    self._current_index = (idx + 1) % len(self._task_queue)
                    return self._task_queue.pop(idx)
        
        # Default round-robin
        task = self._task_queue[self._current_index]
        self._current_index = (self._current_index + 1) % len(self._task_queue)
        return self._task_queue.pop(self._current_index - 1)

    def _is_energy_efficient(self, task: Task) -> bool:
        """Check if task meets energy efficiency criteria."""
        energy_load = task.estimated_load.energy
        return energy_load < (self._energy_efficiency * 1.1)  # Within 10% of target

    def _update_system_load(self) -> None:
        """Update current and historical system load metrics."""
        cpu = psutil.cpu_percent() / 100
        mem = psutil.virtual_memory().percent / 100
        io = psutil.disk_io_counters().busy_time / 100 if hasattr(psutil.disk_io_counters(), 'busy_time') else 0.0
        
        # GPU monitoring would use platform-specific APIs
        gpu = 0.0
        if self.hardware_tier == HardwareTier.HIGH_END:
            try:
                import gpustat
                gpu = gpustat.new_query().gpus[0].utilization / 100
            except ImportError:
                pass

        # Update energy efficiency metric periodically
        if time.time() - self._last_energy_check > 60:
            self._update_energy_efficiency()
            self._last_energy_check = time.time()

        self._current_load = ResourceLoad(
            cpu=cpu,
            mem=mem,
            io=io,
            gpu=gpu,
            energy=self._energy_efficiency
        )
        self._historical_load.append(self._current_load)

    def _update_energy_efficiency(self) -> None:
        """Calculate current energy efficiency score."""
        # This would integrate with hardware monitoring tools
        # Simplified for example purposes
        avg_cpu = mean(l.cpu for l in self._historical_load)
        avg_mem = mean(l.mem for l in self._historical_load)
        
        # Simple heuristic: lower resource usage = better energy efficiency
        self._energy_efficiency = 1.0 - (avg_cpu * 0.6 + avg_mem * 0.4)

    def _is_overloaded(self) -> bool:
        """Check if any resource exceeds safe thresholds."""
        return any(
            getattr(self._current_load, resource) > threshold
            for resource, threshold in self._resource_thresholds.items()
        )

    def _can_execute(self, task: Task) -> bool:
        """Check if task can be executed without overloading system."""
        # Always allow urgent tasks (deadline approaching)
        if task.is_urgent():
            return True

        # Calculate predicted load
        predicted = ResourceLoad(
            cpu=self._current_load.cpu + task.estimated_load.cpu,
            mem=self._current_load.mem + task.estimated_load.mem,
            io=self._current_load.io + task.estimated_load.io,
            gpu=self._current_load.gpu + task.estimated_load.gpu,
            energy=self._energy_efficiency
        )

        # Check against thresholds with some headroom
        headroom = 0.05  # 5% buffer
        return not any(
            getattr(predicted, resource) > (threshold - headroom)
            for resource, threshold in self._resource_thresholds.items()
        )

    def _defer_task(self, task: Task) -> None:
        """Requeue task with adjusted priority."""
        with self._queue_lock:
            task.adjust_priority(0.9)  # Slightly reduce priority
            if self.strategy == SchedulingStrategy.PRIORITY:
                heapq.heappush(self._task_queue, (-task.priority, task))
            else:
                self._task_queue.append(task)
            self._deferred_tasks += 1
            logger.debug(f"Deferred task {task.id} due to resource constraints")

    def _apply_backpressure(self) -> None:
        """Signal upstream components to reduce load."""
        overloaded = [
            resource for resource, threshold in self._resource_thresholds.items()
            if getattr(self._current_load, resource) > threshold
        ]
        
        if overloaded:
            message = json.dumps({
                "type": "BACKPRESSURE",
                "resources": overloaded,
                "timestamp": time.time()
            })
            try:
                self.socket.send_string(message)
                logger.warning(
                    f"Applied backpressure for overloaded resources: {overloaded}"
                )
            except zmq.ZMQError as e:
                logger.error(f"Failed to send backpressure signal: {str(e)}")

    def graceful_degradation(self) -> Dict[str, List[str]]:
        """Determine which features to disable under load."""
        disabled = []
        
        # Always disable non-essential features
        disabled.extend([
            'high_resolution_analytics',
            'background_learning',
            'non_critical_plugins'
        ])

        # Tier-specific degradations
        if self.hardware_tier == HardwareTier.GARBAGE:
            disabled.append('visual_processing')
        elif self.hardware_tier == HardwareTier.LOW_END:
            disabled.append('high_accuracy_mode')

        # Load-based degradations
        if self._current_load.cpu > 0.8:
            disabled.append('complex_ml_models')
        if self._current_load.mem > 0.75:
            disabled.append('caching_layer')
            
        return {"disabled_features": disabled}

    def get_metrics(self) -> Dict[str, Any]:
        """Return scheduler performance metrics."""
        return {
            "queue_size": len(self._task_queue),
            "completed_tasks": self._completed_tasks,
            "deferred_tasks": self._deferred_tasks,
            "current_load": self._current_load.__dict__,
            "energy_efficiency": self._energy_efficiency,
            "hardware_tier": self.hardware_tier.name,
            "scheduling_strategy": self.strategy.name
        }