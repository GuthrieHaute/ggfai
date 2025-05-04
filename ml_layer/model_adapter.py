# GGFAI Model Adapter Template
# written by DeepSeek Chat (honor call: The Universal Translator)

from typing import Dict, Any
from dataclasses import asdict
from ..core.tag_registry import Tag
import logging

class ModelAdapter:
    """Converts any model's output to GGFAI-standard tags"""
    
    def __init__(self, model_path: str):
        self.logger = logging.getLogger("GGFAI.adapter")
        self.model = self._load_model(model_path)
        self.logger.info(f"Loaded adapter for: {model_path}")

    def _load_model(self, path: str):
        """Auto-detects model type and loads with optimal settings"""
        if path.endswith(".gguf"):
            from llama_cpp import Llama
            return Llama(model_path=path, n_ctx=2048, n_threads=4)
        elif path.endswith(".onnx"):
            import onnxruntime
            return onnxruntime.InferenceSession(path)
        elif path.endswith(".tflite"):
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            raise ValueError(f"Unsupported model format: {path}")

    def to_ggfai_tag(self, raw_output: Dict[str, Any], source_text: str = "") -> Tag:
        """
        Universal conversion method. Example mappings:
        
        | Model Type      | Raw Output              | GGFAI Tag Field  |
        |-----------------|-------------------------|------------------|
        | Intent Classifier | {"label": "play_music"} | intent           |
        | Feature Extractor| {"dimmable": True}      | metadata.features|
        | LLM             | {"action": "dim_lights"}| intent           |
        """
        try:
            # Intent Classification
            if "label" in raw_output:
                return Tag(
                    name=f"intent_{raw_output['label']}",
                    intent=raw_output["label"],
                    priority=raw_output.get("confidence", 0.7),
                    metadata={"source": source_text}
                )
            
            # Feature Extraction
            elif "features" in raw_output:
                return Tag(
                    name=f"feat_{hash(str(raw_output))}",
                    category="device_features",
                    metadata=raw_output
                )
            
            # LLM Structured Output
            elif "action" in raw_output:
                return Tag(
                    name=f"llm_{raw_output['action']}",
                    intent=raw_output["action"],
                    metadata={k:v for k,v in raw_output.items() if k != "action"}
                )
                
            # Fallback: Embeddings
            else:
                return Tag(
                    name="embedding",
                    metadata={"vector": raw_output}
                )
                
        except Exception as e:
            self.logger.error(f"Adapter failed: {str(e)}")
            return Tag(name="adapter_error", intent="unknown")

    def predict(self, input_data, **kwargs) -> Tag:
        """Standardized prediction interface"""
        # Preprocess
        model_input = self._preprocess(input_data)
        
        # Infer
        if hasattr(self.model, "create_completion"):  # GGUF
            raw = self.model.create_completion(model_input, **kwargs)
        elif hasattr(self.model, "run"):  # ONNX
            raw = self.model.run(None, {self.model.get_inputs()[0].name: model_input})
        else:  # TFLite
            self.model.set_tensor(input_details[0]['index'], model_input)
            self.model.invoke()
            raw = self.model.get_tensor(output_details[0]['index'])
        
        # Convert
        return self.to_ggfai_tag(raw, source_text=str(input_data)[:100])

    def _preprocess(self, input_data):
        """Model-specific input formatting"""
        if isinstance(input_data, str):
            return input_data.encode("utf-8")
        return input_data