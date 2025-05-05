"""
Proactive anomaly detection for resource management
"""
from typing import Dict, List, Optional, Set, Deque
from collections import deque
import numpy as np
from enum import Enum
import time
import psutil
import logging
from ..core.tag_registry import TagRegistry

class AnomalyType(Enum):
    CPU_SPIKE = "cpu_spike"
    MEMORY_LEAK = "memory_leak"
    DISK_FULL = "disk_full"
    NETWORK_CONGESTION = "network_congestion"

class ProactiveAnomalyDetector:
    def __init__(self, tag_registry: TagRegistry, window_size: int = 100, sigma_threshold: float = 3.0):
        self.tag_registry = tag_registry
        self.metrics: Dict[str, Deque[float]] = {
            'cpu': deque(maxlen=window_size),
            'mem': deque(maxlen=window_size),
            'io': deque(maxlen=window_size)
        }
        self.threshold = sigma_threshold
        self.leak_detection = {
            'mem': {'last': 0, 'increasing_streak': 0},
            'handles': {'last': 0, 'increasing_streak': 0}
        }
        self.logger = logging.getLogger("GGFAI.anomaly")

    def _check_sigma_rule(self, values: Deque[float], new_value: float) -> bool:
        """3-sigma rule for anomaly detection."""
        if len(values) < 10:  # Insufficient data
            return False
        mean = np.mean(list(values))
        std = np.std(list(values))
        return abs(new_value - mean) > self.threshold * std

    def run_detection_cycle(self) -> List[Dict[str, str]]:
        """Run anomaly detection cycle and return any detected issues."""
        anomalies = []
        alerts = self.detect_anomalies()
        
        for alert_type, message in alerts.items():
            anomaly = {
                "type": alert_type,
                "message": message,
                "timestamp": time.time(),
                "severity": "high" if "leak" in alert_type else "medium"
            }
            anomalies.append(anomaly)
            
            # Log to tag registry for tracking
            self.tag_registry.add_tag({
                "type": "anomaly",
                "name": alert_type,
                "message": message,
                "severity": anomaly["severity"],
                "timestamp": anomaly["timestamp"]
            })
            
        return anomalies

    def detect_anomalies(self) -> Dict[str, str]:
        """Run all anomaly checks and return alerts."""
        current = {
            'cpu': psutil.cpu_percent(),
            'mem': psutil.virtual_memory().percent,
            'io': psutil.disk_io_counters().busy_time if hasattr(psutil.disk_io_counters(), 'busy_time') else 0,
            'handles': len(psutil.Process().open_files())
        }
        
        alerts = {}
        
        # Sigma-rule detection
        for metric in self.metrics:
            if self._check_sigma_rule(self.metrics[metric], current[metric]):
                alerts[f"spike_{metric}"] = f"{metric.upper()} spike detected"
            self.metrics[metric].append(current[metric])

        # Memory/Handle leak detection
        for resource in ['mem', 'handles']:
            if current[resource] > self.leak_detection[resource]['last']:
                self.leak_detection[resource]['increasing_streak'] += 1
                if self.leak_detection[resource]['increasing_streak'] > 5:
                    alerts[f"leak_{resource}"] = f"Potential {resource} leak"
            else:
                self.leak_detection[resource]['increasing_streak'] = 0
            self.leak_detection[resource]['last'] = current[resource]
            
        # Resource exhaustion checks
        if current['mem'] > 90:
            alerts['critical_memory'] = f"Critical memory usage: {current['mem']}%"
        if current['cpu'] > 95:
            alerts['critical_cpu'] = f"Critical CPU usage: {current['cpu']}%"
            
        return alerts

    def get_resource_trends(self) -> Dict[str, Dict[str, float]]:
        """Calculate resource usage trends."""
        trends = {}
        
        for metric, values in self.metrics.items():
            if len(values) >= 2:
                values_list = list(values)
                mean = np.mean(values_list)
                std = np.std(values_list)
                trend = (values_list[-1] - values_list[0]) / len(values_list)  # Rate of change
                
                trends[metric] = {
                    'mean': mean,
                    'std': std,
                    'trend': trend,
                    'current': values_list[-1] if values_list else 0
                }
                
        return trends