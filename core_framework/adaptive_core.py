"""
Adaptive Core - Hardware-aware adaptive behavior system
"""

import logging
import time
from typing import Dict, List, Optional, Set
from .config_system import ConfigSystem, HardwareTier

logger = logging.getLogger("GGFAI.core_framework.adaptive")

class AdaptiveCore:
    """
    Hardware-aware adaptive behavior system that optimizes performance
    based on available system resources and capabilities.
    """
    
    def __init__(self):
        self.config = ConfigSystem()
        self.active_features: Set[str] = set()
        self.performance_metrics: Dict[str, float] = {}
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize adaptive core with hardware-aware defaults"""
        self.active_features = {"base_processing", "error_recovery"}
        
        # Add features based on hardware tier
        if self.config.hardware_profile.tier in {HardwareTier.MID, HardwareTier.HIGH}:
            self.active_features.update({
                "parallel_processing",
                "local_models",
                "basic_vision"
            })
            
        if self.config.hardware_profile.tier == HardwareTier.HIGH:
            self.active_features.update({
                "advanced_vision",
                "multi_model",
                "real_time_adapt"
            })
    
    def get_active_features(self) -> Set[str]:
        """Get currently active features"""
        return self.active_features.copy()
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics"""
        self.performance_metrics.update(metrics)
        self._adapt_to_metrics()
    
    def _adapt_to_metrics(self) -> None:
        """Adapt system behavior based on performance metrics"""
        cpu_usage = self.performance_metrics.get("cpu_usage", 0.0)
        memory_usage = self.performance_metrics.get("memory_usage", 0.0)
        
        # Adjust features based on resource usage
        if self.config.hardware_profile.tier == HardwareTier.HIGH:
            self._adapt_high_end(cpu_usage, memory_usage)
        elif self.config.hardware_profile.tier == HardwareTier.MID:
            self._adapt_mid_range(cpu_usage, memory_usage)
        else:
            self._adapt_low_end(cpu_usage, memory_usage)
    
    def _adapt_high_end(self, cpu_usage: float, memory_usage: float) -> None:
        """Adapt behavior for high-end systems"""
        # High-end systems can handle more features
        if cpu_usage < 70 and memory_usage < 70:
            # Enable advanced features if resources available
            self.active_features.update({
                "advanced_vision",
                "multi_model",
                "real_time_adapt"
            })
        elif cpu_usage > 90 or memory_usage > 90:
            # Temporarily disable most demanding features
            self.active_features.discard("advanced_vision")
            self.active_features.discard("real_time_adapt")
    
    def _adapt_mid_range(self, cpu_usage: float, memory_usage: float) -> None:
        """Adapt behavior for mid-range gaming PCs"""
        # Mid-range optimization focuses on gaming-PC-level features
        if cpu_usage < 60 and memory_usage < 60:
            # Enable core gaming PC features
            self.active_features.update({
                "parallel_processing",
                "local_models",
                "basic_vision"
            })
        elif cpu_usage > 80 or memory_usage > 80:
            # Scale back to essential features
            self.active_features.discard("basic_vision")
            self.active_features.discard("local_models")
    
    def _adapt_low_end(self, cpu_usage: float, memory_usage: float) -> None:
        """Adapt behavior for low-end systems"""
        # Keep only essential features
        if cpu_usage > 70 or memory_usage > 70:
            self.active_features = {"base_processing", "error_recovery"}
    
    def can_enable_feature(self, feature: str) -> bool:
        """Check if a feature can be enabled given hardware constraints"""
        if self.config.hardware_profile.tier == HardwareTier.LOW:
            # Low-end systems only support basic features
            return feature in {"base_processing", "error_recovery"}
            
        if self.config.hardware_profile.tier == HardwareTier.MID:
            # Mid-range systems support core gaming PC features
            return feature in {
                "base_processing", 
                "error_recovery",
                "parallel_processing",
                "local_models",
                "basic_vision"
            }
            
        # High-end systems support all features
        return True
    
    def get_resource_limits(self) -> Dict[str, float]:
        """Get hardware-appropriate resource limits"""
        if self.config.hardware_profile.tier == HardwareTier.HIGH:
            return {
                "max_cpu_percent": 90,
                "max_memory_percent": 90,
                "max_gpu_percent": 95,
                "max_batch_size": 32,
                "max_parallel_tasks": self.config.hardware_profile.cpu_threads - 1
            }
        elif self.config.hardware_profile.tier == HardwareTier.MID:
            return {
                "max_cpu_percent": 80,
                "max_memory_percent": 80,
                "max_gpu_percent": 85,
                "max_batch_size": 16,
                "max_parallel_tasks": max(2, self.config.hardware_profile.cpu_cores - 1)
            }
        else:
            return {
                "max_cpu_percent": 70,
                "max_memory_percent": 70,
                "max_gpu_percent": 0,  # Don't use GPU
                "max_batch_size": 8,
                "max_parallel_tasks": 2
            }
    
    def optimize_for_hardware(self, component_config: Dict) -> Dict:
        """Optimize component configuration for current hardware"""
        limits = self.get_resource_limits()
        
        # Apply hardware-appropriate limits
        component_config["max_cpu_percent"] = limits["max_cpu_percent"]
        component_config["max_memory_percent"] = limits["max_memory_percent"]
        component_config["max_gpu_percent"] = limits["max_gpu_percent"]
        component_config["batch_size"] = limits["max_batch_size"]
        component_config["parallel_tasks"] = limits["max_parallel_tasks"]
        
        # Enable/disable features based on hardware tier
        component_config["use_gpu"] = self.config.hardware_profile.tier != HardwareTier.LOW
        component_config["enable_advanced_features"] = self.config.hardware_profile.tier == HardwareTier.HIGH
        component_config["optimization_level"] = self.config.hardware_profile.tier.name.lower()
        
        return component_config