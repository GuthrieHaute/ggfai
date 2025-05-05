"""
Multimodal Fusion Module for GGFAI Framework.

This module provides capabilities for fusing information from multiple modalities
(voice, vision, text, sensors) to create a unified understanding of user intent
and environmental context.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
import threading
from dataclasses import dataclass, field
import json
from pathlib import Path

# Import GGFAI components
from core.tag_registry import Tag, TagRegistry
from trackers.context_tracker import ContextTracker
from trackers.intent_tracker import IntentTracker
from ml_layer.intent_engine import IntentEngine

# Configure logging
logger = logging.getLogger("GGFAI.multimodal")


@dataclass
class ModalityConfig:
    """Configuration for a single modality."""
    enabled: bool = True
    weight: float = 1.0  # Relative importance in fusion
    timeout: float = 2.0  # Seconds to wait for input
    required: bool = False  # Whether this modality is required


@dataclass
class MultimodalConfig:
    """Configuration for multimodal fusion."""
    voice: ModalityConfig = field(default_factory=ModalityConfig)
    vision: ModalityConfig = field(default_factory=lambda: ModalityConfig(weight=0.8))
    text: ModalityConfig = field(default_factory=ModalityConfig)
    sensors: ModalityConfig = field(default_factory=lambda: ModalityConfig(weight=0.5))
    
    # Fusion settings
    fusion_method: str = "weighted"  # weighted, max, attention
    context_window: int = 5  # Number of recent inputs to consider
    temporal_decay: float = 0.8  # Decay factor for older inputs
    
    # Resource management
    max_threads: int = 2
    
    # Cache settings
    cache_dir: str = "cache/multimodal"
    
    def __post_init__(self):
        """Initialize default values."""
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


class MultimodalFusion:
    """
    Multimodal fusion system for GGFAI Framework that combines information
    from multiple modalities to create a unified understanding.
    """
    
    def __init__(self, config: Optional[MultimodalConfig] = None, 
                 intent_engine: Optional[IntentEngine] = None):
        """
        Initialize multimodal fusion with configuration.
        
        Args:
            config: Multimodal fusion configuration
            intent_engine: Intent engine for processing fused inputs
        """
        self.config = config or MultimodalConfig()
        self.intent_engine = intent_engine
        self.logger = logging.getLogger("GGFAI.multimodal")
        
        # Initialize state
        self._lock = threading.RLock()
        self._recent_inputs = {
            "voice": [],
            "vision": [],
            "text": [],
            "sensors": []
        }
        
        # Initialize trackers
        self.context_tracker = ContextTracker()
        self.intent_tracker = IntentTracker()
        
        self.logger.info("Multimodal fusion initialized")
    
    def add_input(self, modality: str, input_data: Dict[str, Any]) -> bool:
        """
        Add input from a specific modality.
        
        Args:
            modality: The input modality (voice, vision, text, sensors)
            input_data: The input data
            
        Returns:
            True if successful, False otherwise
        """
        if modality not in self._recent_inputs:
            self.logger.warning(f"Unknown modality: {modality}")
            return False
        
        # Add timestamp if not present
        if "timestamp" not in input_data:
            input_data["timestamp"] = time.time()
        
        # Add source if not present
        if "source" not in input_data:
            input_data["source"] = modality
        
        # Add to recent inputs
        with self._lock:
            self._recent_inputs[modality].append(input_data)
            
            # Limit size of recent inputs
            if len(self._recent_inputs[modality]) > self.config.context_window:
                self._recent_inputs[modality].pop(0)
        
        self.logger.debug(f"Added {modality} input: {input_data.get('intent', 'unknown')}")
        return True
    
    def process_multimodal(self) -> Dict[str, Any]:
        """
        Process all recent inputs to create a fused understanding.
        
        Returns:
            Fused intent dictionary
        """
        with self._lock:
            # Get recent inputs for each modality
            voice_inputs = self._recent_inputs["voice"].copy()
            vision_inputs = self._recent_inputs["vision"].copy()
            text_inputs = self._recent_inputs["text"].copy()
            sensor_inputs = self._recent_inputs["sensors"].copy()
        
        # Check if we have required modalities
        if (self.config.voice.required and not voice_inputs or
            self.config.vision.required and not vision_inputs or
            self.config.text.required and not text_inputs or
            self.config.sensors.required and not sensor_inputs):
            
            self.logger.warning("Missing required modality inputs")
            return {
                "intent": "incomplete_input",
                "category": "system",
                "confidence": 0.5,
                "requires_clarification": True,
                "timestamp": time.time()
            }
        
        # Apply temporal decay to older inputs
        voice_inputs = self._apply_temporal_decay(voice_inputs)
        vision_inputs = self._apply_temporal_decay(vision_inputs)
        text_inputs = self._apply_temporal_decay(text_inputs)
        sensor_inputs = self._apply_temporal_decay(sensor_inputs)
        
        # Fuse inputs based on configured method
        if self.config.fusion_method == "weighted":
            fused_intent = self._weighted_fusion(
                voice_inputs, vision_inputs, text_inputs, sensor_inputs
            )
        elif self.config.fusion_method == "max":
            fused_intent = self._max_confidence_fusion(
                voice_inputs, vision_inputs, text_inputs, sensor_inputs
            )
        elif self.config.fusion_method == "attention":
            fused_intent = self._attention_fusion(
                voice_inputs, vision_inputs, text_inputs, sensor_inputs
            )
        else:
            self.logger.warning(f"Unknown fusion method: {self.config.fusion_method}")
            fused_intent = self._weighted_fusion(
                voice_inputs, vision_inputs, text_inputs, sensor_inputs
            )
        
        # Process with intent engine if available
        if self.intent_engine and fused_intent:
            try:
                # Create context for intent engine
                context = set(["multimodal_input"])
                
                # Add modality information to context
                if voice_inputs:
                    context.add("voice_input")
                if vision_inputs:
                    context.add("vision_input")
                if text_inputs:
                    context.add("text_input")
                if sensor_inputs:
                    context.add("sensor_input")
                
                # Create text description for intent engine
                description = self._create_multimodal_description(
                    fused_intent, voice_inputs, vision_inputs, text_inputs, sensor_inputs
                )
                
                # Process with intent engine
                processed_intent = self.intent_engine.process(description, context)
                
                # Merge the results, preserving multimodal metadata
                if processed_intent:
                    # Keep multimodal metadata
                    multimodal_info = {
                        k: v for k, v in fused_intent.items() 
                        if k in ["modalities", "voice_data", "vision_data", 
                                "text_data", "sensor_data", "source"]
                    }
                    
                    # Update with processed intent but preserve multimodal metadata
                    fused_intent = processed_intent
                    fused_intent.update(multimodal_info)
                    
            except Exception as e:
                self.logger.error(f"Intent engine processing failed: {e}")
        
        return fused_intent
    
    def _apply_temporal_decay(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply temporal decay to inputs based on age.
        
        Args:
            inputs: List of input dictionaries
            
        Returns:
            Inputs with confidence adjusted by temporal decay
        """
        if not inputs:
            return []
        
        current_time = time.time()
        result = []
        
        for input_data in inputs:
            # Create a copy to avoid modifying original
            input_copy = input_data.copy()
            
            # Calculate age in seconds
            age = current_time - input_copy.get("timestamp", current_time)
            
            # Apply decay factor based on age
            decay_factor = self.config.temporal_decay ** min(age, self.config.context_window)
            
            # Adjust confidence
            if "confidence" in input_copy:
                input_copy["confidence"] = input_copy["confidence"] * decay_factor
            
            # Add decay factor for reference
            input_copy["decay_factor"] = decay_factor
            
            result.append(input_copy)
        
        return result
    
    def _weighted_fusion(self, voice_inputs: List[Dict[str, Any]], 
                         vision_inputs: List[Dict[str, Any]],
                         text_inputs: List[Dict[str, Any]], 
                         sensor_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse inputs using weighted averaging.
        
        Args:
            voice_inputs: List of voice input dictionaries
            vision_inputs: List of vision input dictionaries
            text_inputs: List of text input dictionaries
            sensor_inputs: List of sensor input dictionaries
            
        Returns:
            Fused intent dictionary
        """
        # Get most recent input from each modality
        voice_input = voice_inputs[-1] if voice_inputs else None
        vision_input = vision_inputs[-1] if vision_inputs else None
        text_input = text_inputs[-1] if text_inputs else None
        sensor_input = sensor_inputs[-1] if sensor_inputs else None
        
        # Track which modalities are available
        modalities = []
        if voice_input:
            modalities.append("voice")
        if vision_input:
            modalities.append("vision")
        if text_input:
            modalities.append("text")
        if sensor_input:
            modalities.append("sensors")
        
        if not modalities:
            return {
                "intent": "no_input",
                "category": "system",
                "confidence": 0.0,
                "timestamp": time.time(),
                "modalities": []
            }
        
        # Initialize fused intent with default values
        fused_intent = {
            "intent": "unknown",
            "category": "unknown",
            "confidence": 0.0,
            "timestamp": time.time(),
            "modalities": modalities,
            "source": "multimodal"
        }
        
        # Calculate total weight
        total_weight = 0
        if voice_input and self.config.voice.enabled:
            total_weight += self.config.voice.weight
        if vision_input and self.config.vision.enabled:
            total_weight += self.config.vision.weight
        if text_input and self.config.text.enabled:
            total_weight += self.config.text.weight
        if sensor_input and self.config.sensors.enabled:
            total_weight += self.config.sensors.weight
        
        # Prevent division by zero
        if total_weight == 0:
            total_weight = 1
        
        # Collect all intents and categories with their weights
        intent_weights = {}
        category_weights = {}
        confidence_sum = 0
        
        # Process voice input
        if voice_input and self.config.voice.enabled:
            weight = self.config.voice.weight / total_weight
            
            # Update intent weights
            intent = voice_input.get("intent", "unknown")
            intent_weights[intent] = intent_weights.get(intent, 0) + weight
            
            # Update category weights
            category = voice_input.get("category", "unknown")
            category_weights[category] = category_weights.get(category, 0) + weight
            
            # Update confidence sum
            confidence_sum += voice_input.get("confidence", 0.5) * weight
            
            # Store voice data
            fused_intent["voice_data"] = {
                "intent": intent,
                "category": category,
                "confidence": voice_input.get("confidence", 0.5),
                "text": voice_input.get("text", "")
            }
        
        # Process vision input
        if vision_input and self.config.vision.enabled:
            weight = self.config.vision.weight / total_weight
            
            # Update intent weights
            intent = vision_input.get("intent", "unknown")
            intent_weights[intent] = intent_weights.get(intent, 0) + weight
            
            # Update category weights
            category = vision_input.get("category", "unknown")
            category_weights[category] = category_weights.get(category, 0) + weight
            
            # Update confidence sum
            confidence_sum += vision_input.get("confidence", 0.5) * weight
            
            # Store vision data
            fused_intent["vision_data"] = {
                "intent": intent,
                "category": category,
                "confidence": vision_input.get("confidence", 0.5),
                "objects": vision_input.get("primary_objects", []),
                "scene_type": vision_input.get("scene_type", "unknown")
            }
        
        # Process text input
        if text_input and self.config.text.enabled:
            weight = self.config.text.weight / total_weight
            
            # Update intent weights
            intent = text_input.get("intent", "unknown")
            intent_weights[intent] = intent_weights.get(intent, 0) + weight
            
            # Update category weights
            category = text_input.get("category", "unknown")
            category_weights[category] = category_weights.get(category, 0) + weight
            
            # Update confidence sum
            confidence_sum += text_input.get("confidence", 0.5) * weight
            
            # Store text data
            fused_intent["text_data"] = {
                "intent": intent,
                "category": category,
                "confidence": text_input.get("confidence", 0.5),
                "text": text_input.get("text", "")
            }
        
        # Process sensor input
        if sensor_input and self.config.sensors.enabled:
            weight = self.config.sensors.weight / total_weight
            
            # Update intent weights
            intent = sensor_input.get("intent", "unknown")
            intent_weights[intent] = intent_weights.get(intent, 0) + weight
            
            # Update category weights
            category = sensor_input.get("category", "unknown")
            category_weights[category] = category_weights.get(category, 0) + weight
            
            # Update confidence sum
            confidence_sum += sensor_input.get("confidence", 0.5) * weight
            
            # Store sensor data
            fused_intent["sensor_data"] = {
                "intent": intent,
                "category": category,
                "confidence": sensor_input.get("confidence", 0.5),
                "readings": sensor_input.get("readings", {})
            }
        
        # Select intent and category with highest weight
        if intent_weights:
            fused_intent["intent"] = max(intent_weights.items(), key=lambda x: x[1])[0]
        
        if category_weights:
            fused_intent["category"] = max(category_weights.items(), key=lambda x: x[1])[0]
        
        # Set confidence
        fused_intent["confidence"] = confidence_sum
        
        # Check for conflicting intents
        if len(intent_weights) > 1:
            # Sort intents by weight
            sorted_intents = sorted(intent_weights.items(), key=lambda x: x[1], reverse=True)
            
            # If top two intents are close, mark as ambiguous
            if len(sorted_intents) >= 2:
                top_weight = sorted_intents[0][1]
                second_weight = sorted_intents[1][1]
                
                if top_weight - second_weight < 0.3:  # Threshold for ambiguity
                    fused_intent["ambiguous"] = True
                    fused_intent["alternative_intents"] = [
                        {"intent": intent, "weight": weight}
                        for intent, weight in sorted_intents[:3]  # Top 3 alternatives
                    ]
        
        return fused_intent
    
    def _max_confidence_fusion(self, voice_inputs: List[Dict[str, Any]], 
                              vision_inputs: List[Dict[str, Any]],
                              text_inputs: List[Dict[str, Any]], 
                              sensor_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse inputs by selecting the one with highest confidence.
        
        Args:
            voice_inputs: List of voice input dictionaries
            vision_inputs: List of vision input dictionaries
            text_inputs: List of text input dictionaries
            sensor_inputs: List of sensor input dictionaries
            
        Returns:
            Fused intent dictionary
        """
        # Get most recent input from each modality
        voice_input = voice_inputs[-1] if voice_inputs else None
        vision_input = vision_inputs[-1] if vision_inputs else None
        text_input = text_inputs[-1] if text_inputs else None
        sensor_input = sensor_inputs[-1] if sensor_inputs else None
        
        # Track which modalities are available
        modalities = []
        if voice_input:
            modalities.append("voice")
        if vision_input:
            modalities.append("vision")
        if text_input:
            modalities.append("text")
        if sensor_input:
            modalities.append("sensors")
        
        if not modalities:
            return {
                "intent": "no_input",
                "category": "system",
                "confidence": 0.0,
                "timestamp": time.time(),
                "modalities": []
            }
        
        # Collect all inputs with their confidence
        inputs = []
        
        if voice_input and self.config.voice.enabled:
            inputs.append((
                voice_input, 
                voice_input.get("confidence", 0.5) * self.config.voice.weight
            ))
        
        if vision_input and self.config.vision.enabled:
            inputs.append((
                vision_input, 
                vision_input.get("confidence", 0.5) * self.config.vision.weight
            ))
        
        if text_input and self.config.text.enabled:
            inputs.append((
                text_input, 
                text_input.get("confidence", 0.5) * self.config.text.weight
            ))
        
        if sensor_input and self.config.sensors.enabled:
            inputs.append((
                sensor_input, 
                sensor_input.get("confidence", 0.5) * self.config.sensors.weight
            ))
        
        # Sort by confidence
        inputs.sort(key=lambda x: x[1], reverse=True)
        
        # Select input with highest confidence
        selected_input, selected_confidence = inputs[0]
        
        # Create fused intent
        fused_intent = {
            "intent": selected_input.get("intent", "unknown"),
            "category": selected_input.get("category", "unknown"),
            "confidence": selected_confidence,
            "timestamp": time.time(),
            "modalities": modalities,
            "source": "multimodal",
            "primary_modality": selected_input.get("source", "unknown")
        }
        
        # Store data from each modality
        if voice_input and self.config.voice.enabled:
            fused_intent["voice_data"] = {
                "intent": voice_input.get("intent", "unknown"),
                "category": voice_input.get("category", "unknown"),
                "confidence": voice_input.get("confidence", 0.5),
                "text": voice_input.get("text", "")
            }
        
        if vision_input and self.config.vision.enabled:
            fused_intent["vision_data"] = {
                "intent": vision_input.get("intent", "unknown"),
                "category": vision_input.get("category", "unknown"),
                "confidence": vision_input.get("confidence", 0.5),
                "objects": vision_input.get("primary_objects", []),
                "scene_type": vision_input.get("scene_type", "unknown")
            }
        
        if text_input and self.config.text.enabled:
            fused_intent["text_data"] = {
                "intent": text_input.get("intent", "unknown"),
                "category": text_input.get("category", "unknown"),
                "confidence": text_input.get("confidence", 0.5),
                "text": text_input.get("text", "")
            }
        
        if sensor_input and self.config.sensors.enabled:
            fused_intent["sensor_data"] = {
                "intent": sensor_input.get("intent", "unknown"),
                "category": sensor_input.get("category", "unknown"),
                "confidence": sensor_input.get("confidence", 0.5),
                "readings": sensor_input.get("readings", {})
            }
        
        # Add alternatives
        if len(inputs) > 1:
            fused_intent["alternatives"] = [
                {
                    "intent": input_data.get("intent", "unknown"),
                    "category": input_data.get("category", "unknown"),
                    "confidence": conf,
                    "source": input_data.get("source", "unknown")
                }
                for input_data, conf in inputs[1:3]  # Top 3 alternatives (excluding the selected one)
            ]
        
        return fused_intent
    
    def _attention_fusion(self, voice_inputs: List[Dict[str, Any]], 
                         vision_inputs: List[Dict[str, Any]],
                         text_inputs: List[Dict[str, Any]], 
                         sensor_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse inputs using attention mechanism.
        
        Args:
            voice_inputs: List of voice input dictionaries
            vision_inputs: List of vision input dictionaries
            text_inputs: List of text input dictionaries
            sensor_inputs: List of sensor input dictionaries
            
        Returns:
            Fused intent dictionary
        """
        # This is a simplified attention mechanism
        # In a full implementation, this would use a learned attention model
        
        # Get most recent input from each modality
        voice_input = voice_inputs[-1] if voice_inputs else None
        vision_input = vision_inputs[-1] if vision_inputs else None
        text_input = text_inputs[-1] if text_inputs else None
        sensor_input = sensor_inputs[-1] if sensor_inputs else None
        
        # Track which modalities are available
        modalities = []
        if voice_input:
            modalities.append("voice")
        if vision_input:
            modalities.append("vision")
        if text_input:
            modalities.append("text")
        if sensor_input:
            modalities.append("sensors")
        
        if not modalities:
            return {
                "intent": "no_input",
                "category": "system",
                "confidence": 0.0,
                "timestamp": time.time(),
                "modalities": []
            }
        
        # Initialize fused intent with default values
        fused_intent = {
            "intent": "unknown",
            "category": "unknown",
            "confidence": 0.0,
            "timestamp": time.time(),
            "modalities": modalities,
            "source": "multimodal"
        }
        
        # Calculate attention scores based on recency and confidence
        attention_scores = {}
        
        if voice_input and self.config.voice.enabled:
            # Base attention on confidence and recency
            recency = 1.0  # Most recent
            confidence = voice_input.get("confidence", 0.5)
            attention_scores["voice"] = recency * confidence * self.config.voice.weight
            
            # Store voice data
            fused_intent["voice_data"] = {
                "intent": voice_input.get("intent", "unknown"),
                "category": voice_input.get("category", "unknown"),
                "confidence": confidence,
                "text": voice_input.get("text", "")
            }
        
        if vision_input and self.config.vision.enabled:
            # Base attention on confidence and recency
            recency = 1.0  # Most recent
            confidence = vision_input.get("confidence", 0.5)
            attention_scores["vision"] = recency * confidence * self.config.vision.weight
            
            # Store vision data
            fused_intent["vision_data"] = {
                "intent": vision_input.get("intent", "unknown"),
                "category": vision_input.get("category", "unknown"),
                "confidence": confidence,
                "objects": vision_input.get("primary_objects", []),
                "scene_type": vision_input.get("scene_type", "unknown")
            }
        
        if text_input and self.config.text.enabled:
            # Base attention on confidence and recency
            recency = 1.0  # Most recent
            confidence = text_input.get("confidence", 0.5)
            attention_scores["text"] = recency * confidence * self.config.text.weight
            
            # Store text data
            fused_intent["text_data"] = {
                "intent": text_input.get("intent", "unknown"),
                "category": text_input.get("category", "unknown"),
                "confidence": confidence,
                "text": text_input.get("text", "")
            }
        
        if sensor_input and self.config.sensors.enabled:
            # Base attention on confidence and recency
            recency = 1.0  # Most recent
            confidence = sensor_input.get("confidence", 0.5)
            attention_scores["sensors"] = recency * confidence * self.config.sensors.weight
            
            # Store sensor data
            fused_intent["sensor_data"] = {
                "intent": sensor_input.get("intent", "unknown"),
                "category": sensor_input.get("category", "unknown"),
                "confidence": confidence,
                "readings": sensor_input.get("readings", {})
            }
        
        # Normalize attention scores
        total_attention = sum(attention_scores.values())
        if total_attention > 0:
            for modality in attention_scores:
                attention_scores[modality] /= total_attention
        
        # Store attention scores
        fused_intent["attention_scores"] = attention_scores
        
        # Determine primary modality
        if attention_scores:
            primary_modality = max(attention_scores.items(), key=lambda x: x[1])[0]
            fused_intent["primary_modality"] = primary_modality
            
            # Use intent and category from primary modality
            if primary_modality == "voice" and voice_input:
                fused_intent["intent"] = voice_input.get("intent", "unknown")
                fused_intent["category"] = voice_input.get("category", "unknown")
                fused_intent["confidence"] = voice_input.get("confidence", 0.5)
            elif primary_modality == "vision" and vision_input:
                fused_intent["intent"] = vision_input.get("intent", "unknown")
                fused_intent["category"] = vision_input.get("category", "unknown")
                fused_intent["confidence"] = vision_input.get("confidence", 0.5)
            elif primary_modality == "text" and text_input:
                fused_intent["intent"] = text_input.get("intent", "unknown")
                fused_intent["category"] = text_input.get("category", "unknown")
                fused_intent["confidence"] = text_input.get("confidence", 0.5)
            elif primary_modality == "sensors" and sensor_input:
                fused_intent["intent"] = sensor_input.get("intent", "unknown")
                fused_intent["category"] = sensor_input.get("category", "unknown")
                fused_intent["confidence"] = sensor_input.get("confidence", 0.5)
        
        return fused_intent
    
    def _create_multimodal_description(self, fused_intent: Dict[str, Any],
                                      voice_inputs: List[Dict[str, Any]], 
                                      vision_inputs: List[Dict[str, Any]],
                                      text_inputs: List[Dict[str, Any]], 
                                      sensor_inputs: List[Dict[str, Any]]) -> str:
        """
        Create a natural language description of the multimodal inputs.
        
        Args:
            fused_intent: The fused intent dictionary
            voice_inputs: List of voice input dictionaries
            vision_inputs: List of vision input dictionaries
            text_inputs: List of text input dictionaries
            sensor_inputs: List of sensor input dictionaries
            
        Returns:
            Text description of the multimodal inputs
        """
        description_parts = []
        
        # Add voice description
        if voice_inputs and "voice_data" in fused_intent:
            voice_text = fused_intent["voice_data"].get("text", "")
            if voice_text:
                description_parts.append(f"User said: \"{voice_text}\"")
        
        # Add vision description
        if vision_inputs and "vision_data" in fused_intent:
            vision_data = fused_intent["vision_data"]
            objects = vision_data.get("objects", [])
            scene_type = vision_data.get("scene_type", "unknown")
            
            if objects:
                object_classes = [obj.get("class", "unknown") for obj in objects[:3]]
                object_str = ", ".join(object_classes)
                description_parts.append(f"I can see: {object_str}")
            
            if scene_type != "unknown":
                description_parts.append(f"The scene appears to be a {scene_type} environment")
        
        # Add text description
        if text_inputs and "text_data" in fused_intent:
            text = fused_intent["text_data"].get("text", "")
            if text and (not voice_inputs or text != fused_intent.get("voice_data", {}).get("text", "")):
                description_parts.append(f"User typed: \"{text}\"")
        
        # Add sensor description
        if sensor_inputs and "sensor_data" in fused_intent:
            readings = fused_intent["sensor_data"].get("readings", {})
            if readings:
                sensor_parts = []
                for sensor, value in readings.items():
                    sensor_parts.append(f"{sensor}: {value}")
                
                if sensor_parts:
                    sensor_str = ", ".join(sensor_parts)
                    description_parts.append(f"Sensor readings: {sensor_str}")
        
        # Combine descriptions
        if description_parts:
            return " ".join(description_parts)
        else:
            return "No multimodal input available"
    
    def create_tag_from_intent(self, intent: Dict[str, Any]) -> Tag:
        """
        Create a Tag object from intent dictionary.
        
        Args:
            intent: Intent dictionary
            
        Returns:
            Tag object for tag_registry
        """
        return Tag(
            name=f"multimodal_intent_{int(time.time())}",
            intent=intent.get("intent", "unknown"),
            category=intent.get("category", "multimodal"),
            subcategory="multimodal_input",
            priority=intent.get("priority", 0.8),  # Higher priority for multimodal
            metadata={
                "modalities": intent.get("modalities", []),
                "primary_modality": intent.get("primary_modality", "unknown"),
                "confidence": intent.get("confidence", 1.0),
                "source": "multimodal",
                "timestamp": intent.get("timestamp", time.time())
            }
        )
    
    def clear_inputs(self, modality: Optional[str] = None) -> None:
        """
        Clear recent inputs for a specific modality or all modalities.
        
        Args:
            modality: The modality to clear, or None for all
        """
        with self._lock:
            if modality is None:
                # Clear all modalities
                for mod in self._recent_inputs:
                    self._recent_inputs[mod] = []
            elif modality in self._recent_inputs:
                # Clear specific modality
                self._recent_inputs[modality] = []
            else:
                self.logger.warning(f"Unknown modality: {modality}")
    
    def get_recent_inputs(self, modality: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recent inputs for a specific modality or all modalities.
        
        Args:
            modality: The modality to get, or None for all
            
        Returns:
            Dictionary of recent inputs by modality
        """
        with self._lock:
            if modality is None:
                # Return all modalities
                return {
                    mod: inputs.copy() 
                    for mod, inputs in self._recent_inputs.items()
                }
            elif modality in self._recent_inputs:
                # Return specific modality
                return {modality: self._recent_inputs[modality].copy()}
            else:
                self.logger.warning(f"Unknown modality: {modality}")
                return {}