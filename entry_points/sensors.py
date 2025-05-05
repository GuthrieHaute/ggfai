# Filepath: entry_points/sensors.py
"""
Sensor Integration Module for GGFAI Framework

This module provides a comprehensive interface for sensor data acquisition,
processing, and integration with the GGFAI tag system. It supports:
- Multiple sensor types (environmental, security, energy, etc.)
- Various integration methods (simulated, MQTT, REST API, direct hardware)
- Automatic sensor discovery and configuration
- Sensor data validation and error handling
- Context-aware sensor data processing
"""

import logging
import random
import json
import os
import time
import threading
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path
import importlib.util

from ..core.tag_registry import Tag, TagStatus
from ..resource_management.hardware_shim import detect_hardware_tier
from ..trackers.context_tracker import ContextTracker

# Configure logging
logger = logging.getLogger("GGFAI.sensors")

class SensorError(Exception):
    """Custom exception for sensor-related errors."""
    pass

class SensorManager:
    """
    Manages sensor discovery, configuration, and data collection.
    
    This class serves as the central coordinator for all sensor operations,
    handling sensor registration, scheduling, data validation, and integration
    with the GGFAI tag system.
    """
    
    def __init__(self, context_tracker: Optional[ContextTracker] = None):
        """
        Initialize the sensor manager.
        
        Args:
            context_tracker: Optional context tracker for storing sensor data
        """
        self.sensors: Dict[str, Dict] = {}
        self.sensor_values: Dict[str, Dict] = {}
        self.context_tracker = context_tracker
        self.config_path = Path(os.path.dirname(os.path.dirname(__file__))) / "config" / "sensors.json"
        self.config = self._load_config()
        self.hw_tier = detect_hardware_tier()
        self.running = False
        self.update_threads = {}
        self._lock = threading.RLock()
        
        # Initialize sensors from config
        self._initialize_sensors()
        logger.info(f"Sensor manager initialized with {len(self.sensors)} sensors")
    
    def _load_config(self) -> Dict:
        """Load sensor configuration from file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Sensor config not found at {self.config_path}, using defaults")
                return {
                    "sensors": {},
                    "locations": [],
                    "integrations": {"mqtt": {"enabled": False}}
                }
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded sensor configuration with {len(config.get('sensors', {}))} sensor types")
                return config
        except Exception as e:
            logger.error(f"Error loading sensor config: {str(e)}", exc_info=True)
            return {
                "sensors": {},
                "locations": [],
                "integrations": {"mqtt": {"enabled": False}}
            }
    
    def _initialize_sensors(self):
        """Initialize sensors based on configuration."""
        with self._lock:
            sensor_types = self.config.get("sensors", {})
            locations = self.config.get("locations", [])
            
            # Create sensor instances for each type and location
            for sensor_type, sensor_config in sensor_types.items():
                for location in locations:
                    for subtype in sensor_config.get("types", [""]):
                        sensor_id = f"{location}_{sensor_type}"
                        if subtype:
                            sensor_id += f"_{subtype}"
                        
                        self.sensors[sensor_id] = {
                            "type": sensor_type,
                            "subtype": subtype,
                            "location": location,
                            "units": sensor_config.get("units", ""),
                            "range": sensor_config.get("range", [0, 100]),
                            "precision": sensor_config.get("precision", 0.1),
                            "update_interval": sensor_config.get("update_interval_sec", 60),
                            "priority": sensor_config.get("priority", 0.5),
                            "category": sensor_config.get("category", "sensor"),
                            "last_update": None,
                            "enabled": True
                        }
            
            # Check for MQTT integration
            mqtt_config = self.config.get("integrations", {}).get("mqtt", {})
            if mqtt_config.get("enabled", False):
                self._setup_mqtt_integration(mqtt_config)
    
    def _setup_mqtt_integration(self, mqtt_config: Dict):
        """Set up MQTT integration if available."""
        try:
            if importlib.util.find_spec("paho.mqtt.client"):
                import paho.mqtt.client as mqtt
                
                def on_connect(client, userdata, flags, rc):
                    logger.info(f"Connected to MQTT broker with result code {rc}")
                    # Subscribe to all sensor topics
                    topic = f"{mqtt_config.get('topic_prefix', 'ggfai/sensors/')}#"
                    client.subscribe(topic)
                
                def on_message(client, userdata, msg):
                    try:
                        topic = msg.topic
                        payload = json.loads(msg.payload.decode())
                        # Extract sensor_id from topic
                        sensor_id = topic.split('/')[-1]
                        if sensor_id in self.sensors:
                            self._process_mqtt_data(sensor_id, payload)
                    except Exception as e:
                        logger.error(f"Error processing MQTT message: {str(e)}")
                
                client = mqtt.Client()
                client.on_connect = on_connect
                client.on_message = on_message
                
                # Set up connection in a separate thread to avoid blocking
                def connect_mqtt():
                    try:
                        client.connect(
                            mqtt_config.get("broker", "localhost"),
                            mqtt_config.get("port", 1883),
                            60
                        )
                        client.loop_start()
                    except Exception as e:
                        logger.error(f"Failed to connect to MQTT broker: {str(e)}")
                
                threading.Thread(target=connect_mqtt, daemon=True).start()
                logger.info("MQTT integration initialized")
            else:
                logger.warning("MQTT integration enabled but paho-mqtt not installed")
        except Exception as e:
            logger.error(f"Error setting up MQTT integration: {str(e)}")
    
    def _process_mqtt_data(self, sensor_id: str, data: Dict):
        """Process data received from MQTT."""
        if "value" not in data:
            logger.warning(f"MQTT data for {sensor_id} missing 'value' field")
            return
        
        with self._lock:
            if sensor_id in self.sensors:
                sensor_config = self.sensors[sensor_id]
                
                # Validate value against range
                value = data["value"]
                min_val, max_val = sensor_config["range"]
                if not min_val <= value <= max_val:
                    logger.warning(f"Value {value} for {sensor_id} outside valid range {min_val}-{max_val}")
                    # Clamp to valid range
                    value = max(min_val, min(value, max_val))
                
                # Update sensor value
                self.sensor_values[sensor_id] = {
                    "value": value,
                    "timestamp": datetime.utcnow(),
                    "source": "mqtt"
                }
                
                # Create and register tag
                self._create_sensor_tag(sensor_id)
    
    def start(self):
        """Start sensor data collection."""
        if self.running:
            return
        
        with self._lock:
            self.running = True
            
            # Start update threads for each sensor
            for sensor_id, sensor_config in self.sensors.items():
                if sensor_config["enabled"]:
                    self._start_sensor_update(sensor_id)
            
            logger.info("Sensor manager started")
    
    def stop(self):
        """Stop sensor data collection."""
        with self._lock:
            self.running = False
            # Wait for threads to finish
            for thread in self.update_threads.values():
                if thread.is_alive():
                    thread.join(timeout=1.0)
            self.update_threads.clear()
            logger.info("Sensor manager stopped")
    
    def _start_sensor_update(self, sensor_id: str):
        """Start update thread for a sensor."""
        if sensor_id in self.update_threads and self.update_threads[sensor_id].is_alive():
            return
        
        def update_loop():
            while self.running:
                try:
                    # Read sensor data
                    self.read_sensor(sensor_id)
                    
                    # Sleep until next update
                    interval = self.sensors[sensor_id]["update_interval"]
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in sensor update loop for {sensor_id}: {str(e)}")
                    time.sleep(5)  # Sleep on error to avoid tight loop
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        self.update_threads[sensor_id] = thread
    
    def read_sensor(self, sensor_id: str) -> Optional[Dict]:
        """
        Read data from a specific sensor.
        
        Args:
            sensor_id: Unique sensor identifier
            
        Returns:
            Sensor data dictionary or None if error
        """
        try:
            if sensor_id not in self.sensors:
                raise SensorError(f"Unknown sensor: {sensor_id}")
            
            sensor_config = self.sensors[sensor_id]
            
            # For now, generate simulated data
            # In a real implementation, this would interface with hardware or external APIs
            value = self._generate_simulated_data(sensor_id)
            
            # Store the value
            with self._lock:
                self.sensor_values[sensor_id] = {
                    "value": value,
                    "timestamp": datetime.utcnow(),
                    "source": "simulated"
                }
            
            # Create and register tag
            tag = self._create_sensor_tag(sensor_id)
            
            return {
                "sensor_id": sensor_id,
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "tag": tag.to_dict() if tag else None
            }
            
        except Exception as e:
            logger.error(f"Error reading sensor {sensor_id}: {str(e)}", exc_info=True)
            return None
    
    def _generate_simulated_data(self, sensor_id: str) -> Any:
        """Generate realistic simulated data for a sensor."""
        sensor_config = self.sensors[sensor_id]
        sensor_type = sensor_config["type"]
        min_val, max_val = sensor_config["range"]
        precision = sensor_config["precision"]
        
        # Add some randomness but maintain realistic patterns
        if sensor_type == "temperature":
            # Temperature varies around a baseline with time of day effects
            hour = datetime.now().hour
            time_factor = abs(12 - hour) / 12  # 0 at noon, 1 at midnight
            baseline = min_val + (max_val - min_val) * 0.6  # 60% of range
            variation = (max_val - min_val) * 0.2  # 20% variation
            value = baseline - (variation * time_factor) + random.uniform(-2, 2)
            
        elif sensor_type == "humidity":
            # Humidity inversely related to temperature
            value = min_val + (max_val - min_val) * (0.7 - random.uniform(-0.2, 0.2))
            
        elif sensor_type == "light":
            # Light level based on time of day
            hour = datetime.now().hour
            if 7 <= hour <= 19:  # Daytime
                value = min_val + (max_val - min_val) * (0.7 + random.uniform(-0.3, 0.3))
            else:  # Nighttime
                value = min_val + (max_val - min_val) * random.uniform(0, 0.1)
                
        elif sensor_type in ["motion", "door", "window"]:
            # Boolean sensors (0 or 1)
            # Higher chance of motion during day, doors/windows mostly closed
            if sensor_type == "motion":
                hour = datetime.now().hour
                if 8 <= hour <= 22:  # Active hours
                    value = 1 if random.random() < 0.3 else 0
                else:
                    value = 1 if random.random() < 0.05 else 0
            else:
                value = 1 if random.random() < 0.1 else 0
                
        elif sensor_type == "air_quality":
            # Air quality worse in evenings
            hour = datetime.now().hour
            if 17 <= hour <= 23:  # Evening
                value = min_val + (max_val - min_val) * (0.4 + random.uniform(0, 0.3))
            else:
                value = min_val + (max_val - min_val) * random.uniform(0, 0.3)
                
        elif sensor_type == "power":
            # Power consumption based on time of day
            hour = datetime.now().hour
            if 7 <= hour <= 9 or 17 <= hour <= 21:  # Peak hours
                value = min_val + (max_val - min_val) * (0.6 + random.uniform(0, 0.3))
            else:
                value = min_val + (max_val - min_val) * (0.2 + random.uniform(0, 0.2))
                
        elif sensor_type == "water":
            # Water usage spikes in morning and evening
            hour = datetime.now().hour
            if 6 <= hour <= 8 or 18 <= hour <= 21:  # Usage hours
                value = min_val + (max_val - min_val) * (0.5 + random.uniform(0, 0.4))
            else:
                value = min_val + (max_val - min_val) * random.uniform(0, 0.1)
                
        elif sensor_type == "presence":
            # Presence based on time of day
            hour = datetime.now().hour
            if 8 <= hour <= 22:  # Waking hours
                value = random.randint(1, 3)  # 1-3 people
            else:
                value = random.randint(0, 2)  # 0-2 people at night
                
        else:
            # Generic sensor
            value = min_val + (max_val - min_val) * random.random()
        
        # Apply precision and range constraints
        value = round(max(min_val, min(max_val, value)), int(-1 * (precision < 1) * (precision * 10).as_integer_ratio()[1].bit_length()))
        
        return value
    
    def _create_sensor_tag(self, sensor_id: str) -> Optional[Tag]:
        """Create and register a tag for sensor data."""
        try:
            if sensor_id not in self.sensors or sensor_id not in self.sensor_values:
                return None
            
            sensor_config = self.sensors[sensor_id]
            sensor_value = self.sensor_values[sensor_id]
            
            # Determine tag priority based on sensor type and value
            priority = sensor_config["priority"]
            
            # Increase priority for unusual values or rapid changes
            if sensor_id in self.sensor_values and "previous_value" in self.sensor_values[sensor_id]:
                prev_value = self.sensor_values[sensor_id]["previous_value"]
                current_value = sensor_value["value"]
                
                # Calculate normalized change
                value_range = sensor_config["range"][1] - sensor_config["range"][0]
                if value_range > 0:
                    change_pct = abs(current_value - prev_value) / value_range
                    if change_pct > 0.2:  # Significant change (>20% of range)
                        priority = min(1.0, priority + 0.2)
            
            # Create tag
            tag = Tag(
                name=f"sensor_{sensor_id}",
                intent="sensor_reading",
                category=sensor_config["category"],
                subcategory=sensor_config["type"],
                namespace="sensors",
                priority=priority,
                metadata={
                    "sensor_id": sensor_id,
                    "type": sensor_config["type"],
                    "subtype": sensor_config["subtype"],
                    "location": sensor_config["location"],
                    "value": sensor_value["value"],
                    "unit": sensor_config["units"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "accuracy": 0.9 if self.hw_tier.value > 1 else 0.7,
                    "hw_tier": self.hw_tier.name,
                    "source": sensor_value["source"]
                }
            )
            
            # Store previous value for change detection
            self.sensor_values[sensor_id]["previous_value"] = sensor_value["value"]
            
            # Register with context tracker if available
            if self.context_tracker:
                self.context_tracker.add_tag(tag)
            
            return tag
            
        except Exception as e:
            logger.error(f"Error creating sensor tag for {sensor_id}: {str(e)}", exc_info=True)
            return None
    
    def get_sensor_value(self, sensor_id: str) -> Optional[Dict]:
        """
        Get the current value of a sensor.
        
        Args:
            sensor_id: Sensor identifier
            
        Returns:
            Sensor value dictionary or None if not available
        """
        with self._lock:
            if sensor_id in self.sensor_values:
                return {
                    "value": self.sensor_values[sensor_id]["value"],
                    "timestamp": self.sensor_values[sensor_id]["timestamp"],
                    "unit": self.sensors[sensor_id]["units"] if sensor_id in self.sensors else ""
                }
            return None
    
    def get_sensors_by_type(self, sensor_type: str) -> List[str]:
        """
        Get all sensors of a specific type.
        
        Args:
            sensor_type: Type of sensor to find
            
        Returns:
            List of sensor IDs matching the type
        """
        with self._lock:
            return [
                sensor_id for sensor_id, config in self.sensors.items()
                if config["type"] == sensor_type
            ]
    
    def get_sensors_by_location(self, location: str) -> List[str]:
        """
        Get all sensors in a specific location.
        
        Args:
            location: Location to search for
            
        Returns:
            List of sensor IDs in the location
        """
        with self._lock:
            return [
                sensor_id for sensor_id, config in self.sensors.items()
                if config["location"] == location
            ]
    
    def get_all_sensor_values(self) -> Dict[str, Dict]:
        """
        Get current values for all sensors.
        
        Returns:
            Dictionary of sensor values by sensor ID
        """
        with self._lock:
            result = {}
            for sensor_id in self.sensor_values:
                if sensor_id in self.sensors:
                    result[sensor_id] = {
                        "value": self.sensor_values[sensor_id]["value"],
                        "timestamp": self.sensor_values[sensor_id]["timestamp"],
                        "unit": self.sensors[sensor_id]["units"],
                        "type": self.sensors[sensor_id]["type"],
                        "location": self.sensors[sensor_id]["location"]
                    }
            return result


# Global sensor manager instance
_sensor_manager = None

def get_sensor_manager(context_tracker=None) -> SensorManager:
    """
    Get or create the global sensor manager instance.
    
    Args:
        context_tracker: Optional context tracker for storing sensor data
        
    Returns:
        SensorManager instance
    """
    global _sensor_manager
    if _sensor_manager is None:
        _sensor_manager = SensorManager(context_tracker)
    return _sensor_manager

def read_sensor_data(sensor_id: str) -> Optional[Tag]:
    """
    Read data from a specific sensor and return as a Tag.
    
    This function maintains backward compatibility with the original API
    while leveraging the new SensorManager implementation.
    
    Args:
        sensor_id: Sensor identifier
        
    Returns:
        Tag with sensor data or None if error
    """
    try:
        manager = get_sensor_manager()
        result = manager.read_sensor(sensor_id)
        if result and "tag" in result:
            return Tag(**result["tag"])
        return None
    except Exception as e:
        logger.error(f"Error reading {sensor_id}: {str(e)}", exc_info=True)
        return None

def start_sensor_monitoring():
    """Start the sensor monitoring system."""
    manager = get_sensor_manager()
    manager.start()
    logger.info("Sensor monitoring started")

def stop_sensor_monitoring():
    """Stop the sensor monitoring system."""
    global _sensor_manager
    if _sensor_manager:
        _sensor_manager.stop()
        logger.info("Sensor monitoring stopped")