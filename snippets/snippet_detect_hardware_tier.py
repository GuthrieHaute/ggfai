# Filepath: snippets/snippet_detect_hardware_tier.py
import psutil
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

class HardwareTier(Enum):
    GARBAGE = auto()
    LOW_END = auto()
    MID_RANGE = auto()
    HIGH_END = auto()

def detect_hardware_tier() -> HardwareTier:
    """
    Detect the system's hardware tier based on available resources.
    
    Returns:
        HardwareTier enum value
    """
    try:
        # Get system resources
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
        
        # Determine tier based on thresholds
        if mem_gb < 1 or cpu_cores < 2:
            return HardwareTier.GARBAGE
        elif mem_gb < 4 or cpu_cores < 4:
            return HardwareTier.LOW_END
        elif mem_gb < 8 or cpu_cores < 8:
            return HardwareTier.MID_RANGE
        else:
            return HardwareTier.HIGH_END
            
    except Exception as e:
        logger.error(f"Error detecting hardware tier: {str(e)}")
        return HardwareTier.LOW_END  # Default to low-end if detection fails