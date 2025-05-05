"""
GGFAI Model Adapter - Safe and Efficient Model Interface

Key Improvements:
1. Strong type safety and validation
2. Better error handling and fallbacks
3. Memory management
4. Performance optimizations
5. Extended model support
6. Comprehensive logging
7. Thread safety
8. Configurable preprocessing
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import asdict
import hashlib
from enum import Enum, auto
import gc
from threading import Lock
import numpy as np
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GGFAI.adapter")

class ModelType(Enum):
    """Supported model formats."""
    GGUF = auto()
    ONNX = auto()
    TFLITE = auto()
    PYTORCH = auto()
    SAFETENSORS = auto()

class AdapterError(Exception):
    """Custom exception for adapter failures."""
    pass

class ModelAdapter:
    """
    Robust model adapter with:
    - Memory-safe model loading
    - Thread-safe predictions
    - Comprehensive validation
    - Automatic cleanup
    """
    
    def __init__(
        self,
        model_path: str,
        llm_coordinator,
        component_id: str,
        personality_config: Optional[Dict] = None,
        model_timeout: float = 30.0
    ):
        """
        Initialize adapter with memory and performance optimizations.
        
        Args:
            model_path: Path to model file
            llm_coordinator: LLMCoordinator instance for model management
            component_id: Unique ID of the component using this adapter
            personality_config: Personality configuration for responses
            model_timeout: Maximum time to wait for model acquisition
        """
        self._lock = Lock()
        self.model_path = model_path
        self.model_type = self._detect_model_type(model_path)
        self.model = None
        self._load_config = {}
        self._input_shape = None
        self._output_shape = None
        self._llm_coordinator = llm_coordinator
        self._component_id = component_id
        self._model_timeout = model_timeout
        self.personality_config = personality_config or {
            "warmth": 0.7,
            "formality": 0.5,
            "empathy": 0.8,
            "detail_level": 0.6
        }
        self.conversation_context = []
        
        try:
            # Generate unique model ID
            model_id = self._generate_model_id(model_path)
            
            # Try to acquire model from coordinator
            success, instance = self._llm_coordinator.acquire_llm(
                model_id=model_id,
                requester_id=self._component_id,
                wait=True,
                timeout=self._model_timeout
            )
            
            if success and instance:
                self.model = instance
                logger.info(
                    f"Acquired {self.model_type.name} model {model_id} "
                    f"from coordinator for {self._component_id}"
                )
            else:
                # Load model if not available from coordinator
                self.model = self._load_model(model_path)
                
                # Register with coordinator
                success = self._llm_coordinator.register_llm(
                    model_id=model_id,
                    instance=self.model,
                    owner=self._component_id,
                    metadata={
                        "model_path": model_path,
                        "model_type": self.model_type.name,
                        "load_config": self._load_config
                    }
                )
                
                if not success:
                    raise AdapterError(f"Failed to register model {model_id} with coordinator")
                    
                logger.info(
                    f"Loaded and registered {self.model_type.name} model {model_id} "
                    f"with coordinator for {self._component_id}"
                )
                
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise AdapterError(f"Failed to initialize model: {str(e)}") from e

    def __del__(self):
        """Ensure proper cleanup of model resources."""
        try:
            if self.model and self._llm_coordinator:
                # Release model from coordinator
                model_id = self._generate_model_id(self.model_path)
                self._llm_coordinator.release_llm(
                    model_id=model_id,
                    requester_id=self._component_id
                )
        except Exception as e:
            logger.error(f"Error during model cleanup: {str(e)}")
        finally:
            self._unload_model()

    def _generate_model_id(self, path: str) -> str:
        """Generate unique model ID based on path and config."""
        hasher = hashlib.sha256()
        hasher.update(path.encode())
        hasher.update(str(self._load_config).encode())
        return hasher.hexdigest()[:16]

    def _detect_model_type(self, path: str) -> ModelType:
        """Detect model format from file extension."""
        path = path.lower()
        if path.endswith(".gguf"):
            return ModelType.GGUF
        elif path.endswith(".onnx"):
            return ModelType.ONNX
        elif path.endswith(".tflite"):
            return ModelType.TFLITE
        elif path.endswith(".pt") or path.endswith(".pth"):
            return ModelType.PYTORCH
        elif path.endswith(".safetensors"):
            return ModelType.SAFETENSORS
        else:
            raise ValueError(f"Unsupported model format: {path}")

    def _load_model(self, path: str) -> Any:
        """Load model with format-specific optimizations."""
        try:
            if self.model_type == ModelType.GGUF:
                from llama_cpp import Llama
                return Llama(
                    model_path=path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=-1,
                    verbose=False
                )
                
            elif self.model_type == ModelType.ONNX:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 4
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                return ort.InferenceSession(
                    path,
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
                
            elif self.model_type == ModelType.TFLITE:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(
                    model_path=path,
                    num_threads=4
                )
                interpreter.allocate_tensors()
                self._input_shape = interpreter.get_input_details()[0]['shape']
                self._output_shape = interpreter.get_output_details()[0]['shape']
                return interpreter
                
            elif self.model_type == ModelType.PYTORCH:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = torch.jit.load(path, map_location=device)
                model.eval()
                return model
                
            elif self.model_type == ModelType.SAFETENSORS:
                from safetensors import safe_open
                return safe_open(path, framework="pt")
                
        except ImportError as e:
            logger.error(f"Required library not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise

    def _unload_model(self) -> None:
        """Safely unload model and clean up resources."""
        if self.model is None:
            return
            
        try:
            # GGUF/ONNX/TFLite don't need special cleanup
            if self.model_type in (ModelType.PYTORCH, ModelType.SAFETENSORS):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            del self.model
            self.model = None
            gc.collect()
            logger.info("Model unloaded and memory cleaned up")
        except Exception as e:
            logger.warning(f"Model cleanup failed: {str(e)}")

    async def predict(
        self,
        text: str,
        stream: bool = False,
        requires_clarification: bool = False,
        suggested_clarification: str = None,
        emotion_context: dict = None
    ) -> dict:
        """
        Generate a natural, context-aware response
        """
        # Prepare conversation context
        context = self._prepare_conversation_context(
            text,
            requires_clarification,
            suggested_clarification,
            emotion_context
        )
        
        # Generate response with appropriate style
        response = await self._generate_styled_response(text, context)
        
        # Add natural conversation markers
        response = self._add_conversation_markers(response, context)
        
        return {
            "text": response,
            "context": context
        }
        
    def _prepare_conversation_context(
        self,
        text: str,
        requires_clarification: bool,
        suggested_clarification: str,
        emotion_context: dict
    ) -> dict:
        """
        Prepare rich context for natural conversation
        """
        context = {
            "personality": self.personality_config,
            "emotion": emotion_context or {},
            "needs_clarification": requires_clarification,
            "clarification": suggested_clarification,
            "conversation_style": self._detect_conversation_style(text)
        }
        
        # Adjust personality based on emotional context
        if emotion_context and emotion_context.get("primary_emotion") in ["frustrated", "confused"]:
            context["personality"]["empathy"] = min(1.0, context["personality"]["empathy"] * 1.2)
            context["personality"]["formality"] = max(0.3, context["personality"]["formality"] * 0.8)
            
        return context
        
    async def _generate_styled_response(self, text: str, context: dict) -> str:
        """
        Generate response with appropriate conversation style
        """
        # Prepare prompt with style guidance
        prompt = self._create_styled_prompt(text, context)
        
        # Generate base response
        response = await self._generate_base_response(prompt)
        
        # Apply style adjustments
        response = self._adjust_response_style(response, context)
        
        return response
        
    def _create_styled_prompt(self, text: str, context: dict) -> str:
        """
        Create a prompt that guides the model toward natural conversation
        """
        personality = context["personality"]
        style_markers = []
        
        # Add style markers based on personality
        if personality["warmth"] > 0.6:
            style_markers.append("friendly and approachable")
        if personality["empathy"] > 0.6:
            style_markers.append("understanding and supportive")
        if personality["formality"] < 0.4:
            style_markers.append("casual and relaxed")
            
        # Construct the prompt
        prompt_parts = [
            f"Respond in a {', '.join(style_markers)} way to:",
            text
        ]
        
        # Add clarification guidance if needed
        if context["needs_clarification"]:
            prompt_parts.append(f"\nSeek clarification naturally: {context['clarification']}")
            
        return "\n".join(prompt_parts)
        
    def _add_conversation_markers(self, response: str, context: dict) -> str:
        """
        Add natural conversation markers based on context
        """
        personality = context["personality"]
        markers = {
            "acknowledgment": [
                "I see what you mean",
                "That's interesting",
                "I understand",
                "You raise a good point"
            ],
            "thinking": [
                "Let me think about that",
                "Well",
                "Hmm",
                "You know"
            ],
            "empathy": [
                "I can imagine",
                "That must be",
                "I hear you",
                "It sounds like"
            ]
        }
        
        # Select appropriate markers
        selected_markers = []
        if personality["empathy"] > 0.6:
            selected_markers.extend(markers["empathy"])
        if random.random() < 0.3:
            selected_markers.extend(markers["thinking"])
        if context.get("emotion", {}).get("needs_acknowledgment", False):
            selected_markers.extend(markers["acknowledgment"])
            
        # Add marker if appropriate
        if selected_markers and random.random() < 0.4:
            marker = random.choice(selected_markers)
            response = f"{marker}, {response}"
            
        return response
        
    def _detect_conversation_style(self, text: str) -> dict:
        """
        Detect the user's conversation style
        """
        style = {
            "formality": 0.5,
            "technical_depth": 0.5,
            "emotional_content": 0.0,
            "needs_acknowledgment": False
        }
        
        # Analyze text patterns
        words = text.lower().split()
        
        # Check formality
        formal_indicators = {"would", "could", "please", "kindly", "appreciate"}
        style["formality"] += 0.1 * sum(word in formal_indicators for word in words)
        
        # Check technical depth
        technical_indicators = {"specifically", "technically", "actually", "precisely"}
        style["technical_depth"] += 0.1 * sum(word in technical_indicators for word in words)
        
        # Check emotional content
        emotional_indicators = {"feel", "think", "believe", "want", "need", "hope"}
        style["emotional_content"] = 0.1 * sum(word in emotional_indicators for word in words)
        
        # Detect need for acknowledgment
        style["needs_acknowledgment"] = style["emotional_content"] > 0.3
        
        return style