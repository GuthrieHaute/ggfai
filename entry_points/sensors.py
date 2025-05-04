# Filepath: entry_points/sensors.py
import logging
import random
from typing import Optional
from datetime import datetime
from ..core.tag_registry import Tag
from ..resource_management.hardware_shim import detect_hardware_tier

logger = logging.getLogger(__name__)

def read_sensor_data(sensor_id: str) -> Optional[Tag]:
    """Generate realistic sensor data based on type and hardware tier."""
    try:
        hw_tier = detect_hardware_tier()
        base_accuracy = 0.9 if hw_tier.value > 1 else 0.7
        
        # Sensor type detection
        if "temp" in sensor_id:
            unit = "C"
            value = round(20 + random.uniform(-5, 5), 1)
        elif "humidity" in sensor_id:
            unit = "%"
            value = round(45 + random.uniform(-20, 20), 1)
        elif "light" in sensor_id:
            unit = "lux"
            value = random.randint(0, 1000)
        else:  # Generic sensor
            unit = ""
            value = round(random.uniform(0, 1), 3)

        return Tag(
            name=sensor_id,
            category="sensor",
            metadata={
                "value": value,
                "unit": unit,
                "accuracy": base_accuracy,
                "hw_tier": hw_tier.name,
                "timestamp": datetime.utcnow().isoformat()
            },
            priority=0.6 if "temp" in sensor_id else 0.4  # Higher priority for temp
        )
        
    except Exception as e:
        logger.error(f"Error reading {sensor_id}: {str(e)}", exc_info=True)
        return None