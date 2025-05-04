"""
Hardware monitoring and abstraction layer
"""
from typing import Dict, Optional
import psutil
import time
import logging
from threading import Lock

logger = logging.getLogger("GGFAI.hardware")

class HardwareMonitor:
    def __init__(self, update_interval: float = 1.0):
        self._lock = Lock()
        self._update_interval = update_interval
        self._last_update = 0
        self._cache = {
            'cpu_load': 0.0,
            'memory_available': 0.0,
            'memory_total': psutil.virtual_memory().total,
            'disk_free': 0.0,
            'disk_total': psutil.disk_usage('/').total
        }
        
    def get_cpu_load(self) -> float:
        """Get current CPU load percentage."""
        self._maybe_update()
        with self._lock:
            return self._cache['cpu_load']
            
    def get_available_memory(self) -> float:
        """Get available memory in MB."""
        self._maybe_update()
        with self._lock:
            return self._cache['memory_available']
            
    def get_total_memory(self) -> float:
        """Get total system memory in MB."""
        with self._lock:
            return self._cache['memory_total'] / (1024 * 1024)
            
    def get_memory_info(self) -> Dict[str, float]:
        """Get detailed memory information."""
        self._maybe_update()
        with self._lock:
            return {
                'available': self._cache['memory_available'],
                'total': self._cache['memory_total'] / (1024 * 1024),
                'percent_used': (1 - self._cache['memory_available'] / 
                               self._cache['memory_total']) * 100
            }
            
    def get_disk_info(self) -> Dict[str, float]:
        """Get disk space information."""
        self._maybe_update()
        with self._lock:
            return {
                'free': self._cache['disk_free'] / (1024 * 1024 * 1024),  # GB
                'total': self._cache['disk_total'] / (1024 * 1024 * 1024),
                'percent_used': (1 - self._cache['disk_free'] / 
                               self._cache['disk_total']) * 100
            }
            
    def _maybe_update(self) -> None:
        """Update cached values if interval has elapsed."""
        current_time = time.time()
        if current_time - self._last_update >= self._update_interval:
            self._update_metrics()
            
    def _update_metrics(self) -> None:
        """Update all hardware metrics."""
        try:
            with self._lock:
                # CPU load (percent)
                self._cache['cpu_load'] = psutil.cpu_percent(interval=0.1)
                
                # Memory info (bytes)
                mem = psutil.virtual_memory()
                self._cache['memory_available'] = mem.available / (1024 * 1024)  # MB
                
                # Disk info (bytes)
                disk = psutil.disk_usage('/')
                self._cache['disk_free'] = disk.free
                
                self._last_update = time.time()
                
        except Exception as e:
            logger.error(f"Failed to update hardware metrics: {str(e)}")
            
    def detect_hardware_tier(self) -> str:
        """Detect hardware capabilities tier."""
        mem_gb = self._cache['memory_total'] / (1024 * 1024 * 1024)
        cpu_count = psutil.cpu_count()
        
        if mem_gb >= 16 and cpu_count >= 8:
            return "high_end"
        elif mem_gb >= 8 and cpu_count >= 4:
            return "mid_tier"
        elif mem_gb >= 4 and cpu_count >= 2:
            return "low_end"
        else:
            return "garbage_tier"