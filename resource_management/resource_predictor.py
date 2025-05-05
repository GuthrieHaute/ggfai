"""
Resource demand prediction and profiling
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum
import psutil
import time
from ..core.tag_registry import TagRegistry

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"

class PredictionResult:
    def __init__(self, mean: float, std: float, confidence_interval: Tuple[float, float]):
        self.mean = mean
        self.std = std
        self.confidence_interval = confidence_interval

class ResourceProfile:
    def __init__(self):
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.disk_usage: List[float] = []
        self.network_usage: List[float] = []
        self.timestamps: List[float] = []

    def add_measurement(self, cpu: float, memory: float, disk: float = 0.0, network: float = 0.0):
        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)
        self.disk_usage.append(disk)
        self.network_usage.append(network)
        self.timestamps.append(time.time())

class ResourcePredictor:
    def __init__(self, tag_registry: TagRegistry):
        self.tag_registry = tag_registry
        self.window_size = 10
        
    def predict_resource_demand(
        self, 
        profile: ResourceProfile, 
        prediction_steps: int = 5,
        confidence_level: float = 0.95
    ) -> Dict[ResourceType, PredictionResult]:
        """Predict future resource demands using profile history."""
        predictions = {}
        
        # CPU prediction
        if profile.cpu_usage:
            cpu_mean = np.mean(profile.cpu_usage[-self.window_size:])
            cpu_std = np.std(profile.cpu_usage[-self.window_size:])
            cpu_ci = self._calculate_confidence_interval(
                cpu_mean, cpu_std, min(len(profile.cpu_usage), self.window_size), 
                confidence_level
            )
            predictions[ResourceType.CPU] = PredictionResult(cpu_mean, cpu_std, cpu_ci)
            
        # Memory prediction
        if profile.memory_usage:
            mem_mean = np.mean(profile.memory_usage[-self.window_size:])
            mem_std = np.std(profile.memory_usage[-self.window_size:])
            mem_ci = self._calculate_confidence_interval(
                mem_mean, mem_std, min(len(profile.memory_usage), self.window_size),
                confidence_level
            )
            predictions[ResourceType.MEMORY] = PredictionResult(mem_mean, mem_std, mem_ci)
            
        return predictions

    def _calculate_confidence_interval(
        self, 
        mean: float, 
        std: float, 
        n: int, 
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval using normal distribution."""
        from scipy import stats
        
        confidence = 1.0 - ((1.0 - confidence_level) / 2.0)
        z_value = stats.norm.ppf(confidence)
        margin = z_value * (std / np.sqrt(n))
        
        return (mean - margin, mean + margin)

    def get_current_profile(self) -> ResourceProfile:
        """Get current system resource profile."""
        profile = ResourceProfile()
        
        cpu = psutil.cpu_percent(interval=1) / 100.0
        memory = psutil.virtual_memory().percent / 100.0
        disk = psutil.disk_usage('/').percent / 100.0
        
        # Get network I/O stats
        net_io = psutil.net_io_counters()
        network = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # Convert to MB
        
        profile.add_measurement(cpu, memory, disk, network)
        return profile