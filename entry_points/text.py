# Filepath: entry_points/sensors.py
import logging
from typing import Optional
from datetime import datetime
from ..core.tag_registry import Tag

logger = logging.getLogger(__name__)

def read_sensor_data(sensor_id: str) -> Optional[Tag]:
    """
    Simulate reading data from a sensor and return as a Tag object.
    
    Args:
        sensor_id: Identifier for the sensor
        
    Returns:
        Tag object containing sensor data, or None if reading fails
    """
    try:
        if not sensor_id.startswith("sensor_"):
            logger.warning(f"Invalid sensor ID format: {sensor_id}")
            return None
            
        # Generate dummy data based on sensor type
        if "temp" in sensor_id:
            value = 22.5  # Default temperature
            unit = "C"
        elif "humidity" in sensor_id:
            value = 45.0  # Default humidity
            unit = "%"
        else:
            value = 1.0   # Generic sensor value
            unit = ""
            
        return Tag(
            name=sensor_id,
            category="sensor",
            metadata={
                "value": value,
                "unit": unit,
                "timestamp": datetime.utcnow().isoformat()
            },
            priority=0.5
        )
        
    except Exception as e:
        logger.error(f"Error reading sensor {sensor_id}: {str(e)}")
        return None