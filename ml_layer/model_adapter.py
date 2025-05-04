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
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize adapter with memory and performance optimizations.
        
        Args:
            model_path: Path to model file
            kwargs: Model-specific initialization options
        """
        self._lock = Lock()
        self.model_path = model_path
        self.model_type = self._detect_model_type(model_path)
        self.model = None
        self._load_config = kwargs
        self._input_shape = None
        self._output_shape = None
        
        try:
            self.model = self._load_model(model_path, **kwargs)
            logger.info(
                f"Loaded {self.model_type.name} model from {model_path} "
                f"with config: {kwargs}"
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise AdapterError(f"Failed to load model: {str(e)}") from e

    def __del__(self):
        """Ensure proper cleanup of model resources."""
        self._unload_model()

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

    def _load_model(self, path: str, **kwargs) -> Any:
        """Load model with format-specific optimizations."""
        try:
            if self.model_type == ModelType.GGUF:
                from llama_cpp import Llama
                return Llama(
                    model_path=path,
                    n_ctx=kwargs.get("n_ctx", 2048),
                    n_threads=kwargs.get("n_threads", 4),
                    n_gpu_layers=kwargs.get("n_gpu_layers", -1),
                    verbose=kwargs.get("verbose", False)
                )
                
            elif self.model_type == ModelType.ONNX:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = kwargs.get("n_threads", 4)
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                return ort.InferenceSession(
                    path,
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
                
            elif self.model_type == ModelType.TFLITE:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(
                    model_path=path,
                    num_threads=kwargs.get("n_threads", 4)
                )
                interpreter.allocate_tensors()
                self._input_shape = interpreter.get_input_details()[0]['shape']
                self._output_shape = interpreter.get_output_details()[0]['shape']
                return interpreter
                
            elif self.model_type == ModelType.PYTORCH:
                import torch
                device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
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

    def predict(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Thread-safe prediction with automatic preprocessing.
        
        Args:
            input_data: Input to the model (text, tensor, etc.)
            kwargs: Model-specific inference parameters
            
        Returns:
            Dictionary with standardized output format
        """
        with self._lock:
            if self.model is None:
                raise AdapterError("Model not loaded")
                
            try:
                # Preprocess input
                processed_input = self._preprocess(input_data)
                
                # Run inference
                raw_output = self._infer(processed_input, **kwargs)
                
                # Convert to standard format
                return self._standardize_output(raw_output, str(input_data))
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}", exc_info=True)
                return {
                    "error": str(e),
                    "fallback": True,
                    "model_type": self.model_type.name
                }

    def _preprocess(self, input_data: Any) -> Any:
        """Convert input to model-appropriate format."""
        try:
            if isinstance(input_data, str):
                if self.model_type == ModelType.GGUF:
                    return input_data.encode("utf-8")
                return input_data
                
            elif isinstance(input_data, (np.ndarray, list, tuple)):
                if self.model_type == ModelType.TFLITE and self._input_shape:
                    return np.array(input_data).reshape(self._input_shape)
                return np.array(input_data)
                
            return input_data
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise AdapterError(f"Input processing error: {str(e)}") from e

    def _infer(self, input_data: Any, **kwargs) -> Any:
        """Run model inference with format-specific handling."""
        try:
            if self.model_type == ModelType.GGUF:
                return self.model.create_completion(input_data, **kwargs)
                
            elif self.model_type == ModelType.ONNX:
                input_name = self.model.get_inputs()[0].name
                return self.model.run(None, {input_name: input_data})[0]
                
            elif self.model_type == ModelType.TFLITE:
                input_details = self.model.get_input_details()
                self.model.set_tensor(input_details[0]['index'], input_data)
                self.model.invoke()
                output_details = self.model.get_output_details()
                return self.model.get_tensor(output_details[0]['index'])
                
            elif self.model_type == ModelType.PYTORCH:
                import torch
                with torch.no_grad():
                    if isinstance(input_data, str):
                        # Handle text input
                        return self.model(input_data, **kwargs)
                    else:
                        # Handle tensor input
                        tensor = torch.from_numpy(input_data) if not isinstance(input_data, torch.Tensor) else input_data
                        return self.model(tensor, **kwargs).detach().cpu().numpy()
                        
            elif self.model_type == ModelType.SAFETENSORS:
                # Handle safetensors input (typically embeddings)
                return {"embeddings": input_data}
                
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise AdapterError(f"Inference error: {str(e)}") from e

    def _standardize_output(self, raw_output: Any, source_text: str = "") -> Dict[str, Any]:
        """
        Convert raw model output to standardized GGFAI format.
        
        Standard fields:
        - intent: Primary action/classification
        - confidence: Score (0-1)
        - features: Extracted attributes
        - embeddings: Vector representations
        - metadata: Additional model-specific data
        """
        try:
            if not isinstance(raw_output, dict):
                # Convert non-dict outputs
                if isinstance(raw_output, (np.ndarray, list)):
                    return {
                        "embeddings": raw_output.tolist() if hasattr(raw_output, 'tolist') else raw_output,
                        "source": source_text[:1000]  # Truncate long text
                    }
                return {"raw_output": raw_output, "source": source_text[:1000]}
            
            # Handle structured outputs
            output = {"source": source_text[:1000]}
            
            if "label" in raw_output:  # Classification
                output.update({
                    "intent": raw_output["label"],
                    "confidence": float(raw_output.get("confidence", 0.7))
                })
            elif "action" in raw_output:  # LLM structured
                output.update({
                    "intent": raw_output["action"],
                    "metadata": {k: v for k, v in raw_output.items() if k != "action"}
                })
            elif "features" in raw_output:  # Feature extraction
                output.update({
                    "features": raw_output["features"],
                    "metadata": {k: v for k, v in raw_output.items() if k != "features"}
                })
            else:  # Fallback to raw output
                output["metadata"] = raw_output
                
            return output
            
        except Exception as e:
            logger.error(f"Output standardization failed: {str(e)}")
            return {
                "error": "Output processing failed",
                "raw_output": str(raw_output)[:1000],  # Truncate if large
                "fallback": True
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and capabilities."""
        return {
            "model_type": self.model_type.name,
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "loaded": self.model is not None,
            "config": self._load_config
        }