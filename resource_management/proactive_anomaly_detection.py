# proactive_anomaly_detection.py - Statistical anomaly detection
# written by DeepSeek Chat (honor call: The Sentinel)

import numpy as np
from collections import deque
import psutil
import logging

class AnomalyDetector:
    def __init__(self, 
                 window_size: int = 100,
                 sigma_threshold: float = 3.0):
        self.metrics = {
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

    def _check_sigma_rule(self, values: deque, new_value: float) -> bool:
        """3-sigma rule for anomaly detection."""
        if len(values) < 10:  # Insufficient data
            return False
        mean = np.mean(values)
        std = np.std(values)
        return abs(new_value - mean) > self.threshold * std

    def detect_anomalies(self) -> Dict[str, str]:
        """Run all anomaly checks and return alerts."""
        current = {
            'cpu': psutil.cpu_percent(),
            'mem': psutil.virtual_memory().percent,
            'io': psutil.disk_io_counters().busy_time,
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

        return alerts