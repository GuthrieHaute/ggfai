from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

class IterationState(Enum):
    CONTINUE = "continue"
    PAUSE = "pause" 
    STOP = "stop"
    ERROR = "error"

@dataclass
class IterationMetrics:
    cycle_count: int
    success_rate: float
    error_rate: float
    resource_usage: float
    iteration_duration_ms: float

class IterationController:
    def __init__(self, config: Dict[str, Any]):
        self.max_cycles = config.get("max_cycles", 100)
        self.min_success_rate = config.get("min_success_rate", 0.8)
        self.max_error_rate = config.get("max_error_rate", 0.2)
        self.max_resource_usage = config.get("max_resource_usage", 0.9)
        self.current_cycle = 0
        self.metrics = []

    def should_continue(self, current_metrics: IterationMetrics) -> IterationState:
        """Determines if iteration should continue based on current metrics"""
        self.current_cycle += 1
        self.metrics.append(current_metrics)

        # Check stopping conditions
        if self.current_cycle >= self.max_cycles:
            return IterationState.STOP
            
        if current_metrics.error_rate > self.max_error_rate:
            return IterationState.ERROR

        if current_metrics.success_rate < self.min_success_rate:
            return IterationState.PAUSE

        if current_metrics.resource_usage > self.max_resource_usage:
            return IterationState.PAUSE

        return IterationState.CONTINUE

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Returns summary of iteration progress and metrics"""
        if not self.metrics:
            return {}

        return {
            "total_cycles": self.current_cycle,
            "avg_success_rate": sum(m.success_rate for m in self.metrics) / len(self.metrics),
            "avg_error_rate": sum(m.error_rate for m in self.metrics) / len(self.metrics),
            "avg_resource_usage": sum(m.resource_usage for m in self.metrics) / len(self.metrics),
            "avg_duration_ms": sum(m.iteration_duration_ms for m in self.metrics) / len(self.metrics)
        }

    def reset(self):
        """Resets the controller state"""
        self.current_cycle = 0
        self.metrics.clear()