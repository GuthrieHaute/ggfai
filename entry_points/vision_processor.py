"""
Vision Processing Module for GGFAI Framework.

This module provides computer vision capabilities for the GGFAI Framework,
enabling visual perception, object detection, scene understanding, and
integration with the intent engine for multimodal intelligence.
"""

import logging
import time
import threading
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import concurrent.futures
from dataclasses import dataclass
import queue

import cv2
import numpy as np

# Import GGFAI components
from core.tag_registry import Tag, TagRegistry
from trackers.context_tracker import ContextTracker

# Configure logging
logger = logging.getLogger("GGFAI.vision")

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some vision features will be limited.")


class VisionModelType(Enum):
    """Types of vision models supported by the system."""
    OBJECT_DETECTION = "object_detection"
    SCENE_CLASSIFICATION = "scene_classification"
    FACE_RECOGNITION = "face_recognition"
    OCR = "ocr"
    ACTIVITY_RECOGNITION = "activity_recognition"
    DEPTH_ESTIMATION = "depth_estimation"


@dataclass
class VisionConfig:
    """Configuration for vision processing."""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 15
    
    # Processing settings
    detection_interval: float = 0.5  # Seconds between detections
    confidence_threshold: float = 0.4
    max_objects: int = 20
    
    # Model settings
    model_type: VisionModelType = VisionModelType.OBJECT_DETECTION
    model_path: Optional[str] = None
    use_gpu: bool = TORCH_AVAILABLE
    
    # Resource management
    max_threads: int = 2
    frame_buffer_size: int = 5
    
    # Cache settings
    cache_dir: str = "cache/vision"
    max_cache_size_mb: int = 100
    max_cache_age_days: int = 7
    
    # API keys for cloud services
    api_keys: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.api_keys is None:
            self.api_keys = {}


class VisionProcessor:
    """
    Vision processing system for GGFAI Framework with object detection,
    scene understanding, and integration with the intent engine.
    """
    
    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize vision processor with configuration.
        
        Args:
            config: Vision processing configuration
        """
        self.config = config or VisionConfig()
        self.logger = logging.getLogger("GGFAI.vision")
        
        # Initialize state
        self._lock = threading.RLock()
        self._running = False
        self._camera = None
        self._frame_buffer = queue.Queue(maxsize=self.config.frame_buffer_size)
        self._last_frame = None
        self._last_detection_time = 0
        self._detected_objects = []
        
        # Initialize thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_threads
        )
        
        # Initialize cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self._init_model()
        
        self.logger.info("Vision processor initialized")
    
    def _init_model(self):
        """Initialize vision model based on configuration."""
        self.model = None
        self.model_loaded = False
        
        try:
            if self.config.model_type == VisionModelType.OBJECT_DETECTION:
                self._init_object_detection_model()
            elif self.config.model_type == VisionModelType.SCENE_CLASSIFICATION:
                self._init_scene_classification_model()
            elif self.config.model_type == VisionModelType.FACE_RECOGNITION:
                self._init_face_recognition_model()
            elif self.config.model_type == VisionModelType.OCR:
                self._init_ocr_model()
            else:
                self.logger.warning(f"Unsupported model type: {self.config.model_type}")
                return
            
            self.model_loaded = True
            self.logger.info(f"Vision model {self.config.model_type.value} initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vision model: {e}")
            self.model_loaded = False
    
    def _init_object_detection_model(self):
        """Initialize object detection model."""
        # Use YOLO from OpenCV DNN module for broad compatibility
        try:
            # If custom model path is provided, use it
            if self.config.model_path:
                model_path = self.config.model_path
                config_path = model_path.replace('.weights', '.cfg')
            else:
                # Use pre-trained YOLOv4 model
                model_dir = Path(__file__).parent.parent / "ml_layer" / "models" / "vision"
                model_path = str(model_dir / "yolov4-tiny.weights")
                config_path = str(model_dir / "yolov4-tiny.cfg")
                
                # Download model if not exists
                if not os.path.exists(model_path) or not os.path.exists(config_path):
                    self.logger.info("Downloading YOLOv4-tiny model...")
                    self._download_yolo_model(model_dir)
            
            # Load COCO class names
            classes_path = str(Path(__file__).parent.parent / "ml_layer" / "models" / "vision" / "coco.names")
            if not os.path.exists(classes_path):
                self._download_coco_names(classes_path)
            
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Load model
            self.model = cv2.dnn.readNetFromDarknet(config_path, model_path)
            
            # Use GPU if available and configured
            if self.config.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.logger.info("Using CUDA for object detection")
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.logger.info("Using CPU for object detection")
            
            # Get output layer names
            self.layer_names = self.model.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
            
        except Exception as e:
            self.logger.error(f"Failed to initialize object detection model: {e}")
            raise
    
    def _download_yolo_model(self, model_dir):
        """Download YOLOv4-tiny model files."""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download weights
        weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
        weights_path = model_dir / "yolov4-tiny.weights"
        self._download_file(weights_url, weights_path)
        
        # Download config
        cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
        cfg_path = model_dir / "yolov4-tiny.cfg"
        self._download_file(cfg_url, cfg_path)
    
    def _download_coco_names(self, classes_path):
        """Download COCO class names."""
        url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
        self._download_file(url, classes_path)
    
    def _download_file(self, url, path):
        """Download a file from URL to path."""
        import requests
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Downloaded {url} to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            raise
    
    def _init_scene_classification_model(self):
        """Initialize scene classification model."""
        # Placeholder for scene classification model
        self.logger.warning("Scene classification not implemented yet")
    
    def _init_face_recognition_model(self):
        """Initialize face recognition model."""
        # Placeholder for face recognition model
        self.logger.warning("Face recognition not implemented yet")
    
    def _init_ocr_model(self):
        """Initialize OCR model."""
        # Placeholder for OCR model
        self.logger.warning("OCR not implemented yet")
    
    def start_camera(self) -> bool:
        """
        Start the camera capture process.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if self._running:
                self.logger.warning("Camera already running")
                return True
            
            try:
                self._camera = cv2.VideoCapture(self.config.camera_index)
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                self._camera.set(cv2.CAP_PROP_FPS, self.config.fps)
                
                if not self._camera.isOpened():
                    self.logger.error("Failed to open camera")
                    return False
                
                self._running = True
                
                # Start capture thread
                self._capture_thread = threading.Thread(
                    target=self._capture_loop,
                    daemon=True
                )
                self._capture_thread.start()
                
                self.logger.info(f"Camera started (index: {self.config.camera_index})")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to start camera: {e}")
                if self._camera:
                    self._camera.release()
                    self._camera = None
                return False
    
    def stop_camera(self) -> bool:
        """
        Stop the camera capture process.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if not self._running:
                self.logger.warning("Camera not running")
                return True
            
            try:
                self._running = False
                
                # Wait for capture thread to end
                if hasattr(self, '_capture_thread') and self._capture_thread.is_alive():
                    self._capture_thread.join(timeout=2.0)
                
                if self._camera:
                    self._camera.release()
                    self._camera = None
                
                self.logger.info("Camera stopped")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to stop camera: {e}")
                return False
    
    def _capture_loop(self):
        """Camera capture loop running in a separate thread."""
        last_fps_time = time.time()
        frame_count = 0
        
        while self._running:
            try:
                # Capture frame
                ret, frame = self._camera.read()
                
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Update frame buffer
                try:
                    # Use non-blocking put with a timeout
                    self._frame_buffer.put(frame, block=True, timeout=0.1)
                except queue.Full:
                    # If buffer is full, remove oldest frame
                    try:
                        self._frame_buffer.get_nowait()
                        self._frame_buffer.put(frame, block=False)
                    except:
                        pass
                
                # Update last frame
                with self._lock:
                    self._last_frame = frame
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - last_fps_time
                
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    last_fps_time = current_time
                    self.logger.debug(f"Camera FPS: {fps:.2f}")
                
                # Process frame if needed
                if current_time - self._last_detection_time >= self.config.detection_interval:
                    self.executor.submit(self._process_frame, frame.copy())
                    self._last_detection_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame):
        """
        Process a frame for object detection.
        
        Args:
            frame: The frame to process
        """
        if not self.model_loaded:
            return
        
        try:
            if self.config.model_type == VisionModelType.OBJECT_DETECTION:
                self._detect_objects(frame)
            elif self.config.model_type == VisionModelType.SCENE_CLASSIFICATION:
                self._classify_scene(frame)
            elif self.config.model_type == VisionModelType.FACE_RECOGNITION:
                self._recognize_faces(frame)
            elif self.config.model_type == VisionModelType.OCR:
                self._perform_ocr(frame)
                
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
    
    def _detect_objects(self, frame):
        """
        Detect objects in a frame using YOLO.
        
        Args:
            frame: The frame to process
        """
        try:
            height, width, _ = frame.shape
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                frame, 
                1/255.0, 
                (416, 416), 
                swapRB=True, 
                crop=False
            )
            
            # Set input and forward pass
            self.model.setInput(blob)
            outputs = self.model.forward(self.output_layers)
            
            # Process outputs
            class_ids = []
            confidences = []
            boxes = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.config.confidence_threshold:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                confidences, 
                self.config.confidence_threshold, 
                0.4  # NMS threshold
            )
            
            # Prepare results
            objects = []
            
            for i in indices:
                if isinstance(i, list):  # OpenCV 4.5.4 and earlier
                    i = i[0]
                
                box = boxes[i]
                x, y, w, h = box
                
                # Get class name
                class_id = class_ids[i]
                class_name = self.classes[class_id] if class_id < len(self.classes) else f"unknown_{class_id}"
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Calculate relative position
                rel_x = center_x / width
                rel_y = center_y / height
                
                # Calculate relative size
                rel_size = (w * h) / (width * height)
                
                objects.append({
                    "class": class_name,
                    "confidence": confidences[i],
                    "box": [x, y, w, h],
                    "center": [center_x, center_y],
                    "relative_position": [rel_x, rel_y],
                    "relative_size": rel_size,
                    "timestamp": time.time()
                })
            
            # Limit number of objects
            objects = sorted(objects, key=lambda x: x["confidence"], reverse=True)
            objects = objects[:self.config.max_objects]
            
            # Update detected objects
            with self._lock:
                self._detected_objects = objects
            
            self.logger.debug(f"Detected {len(objects)} objects")
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
    
    def _classify_scene(self, frame):
        """
        Classify the scene in a frame.
        
        Args:
            frame: The frame to process
        """
        # Placeholder for scene classification
        self.logger.debug("Scene classification not implemented yet")
    
    def _recognize_faces(self, frame):
        """
        Recognize faces in a frame.
        
        Args:
            frame: The frame to process
        """
        # Placeholder for face recognition
        self.logger.debug("Face recognition not implemented yet")
    
    def _perform_ocr(self, frame):
        """
        Perform OCR on a frame.
        
        Args:
            frame: The frame to process
        """
        # Placeholder for OCR
        self.logger.debug("OCR not implemented yet")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent camera frame.
        
        Returns:
            The frame or None if not available
        """
        with self._lock:
            return self._last_frame.copy() if self._last_frame is not None else None
    
    def get_detected_objects(self) -> List[Dict[str, Any]]:
        """
        Get the list of detected objects.
        
        Returns:
            List of detected objects with class, confidence, and position
        """
        with self._lock:
            return self._detected_objects.copy()
    
    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame with object detection annotations.
        
        Returns:
            Annotated frame or None if not available
        """
        frame = self.get_current_frame()
        if frame is None:
            return None
        
        objects = self.get_detected_objects()
        
        # Draw bounding boxes and labels
        for obj in objects:
            x, y, w, h = obj["box"]
            class_name = obj["class"]
            confidence = obj["confidence"]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                frame, 
                label, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        return frame
    
    def save_annotated_frame(self, path: str) -> bool:
        """
        Save the current annotated frame to a file.
        
        Args:
            path: Path to save the frame
            
        Returns:
            True if successful, False otherwise
        """
        frame = self.get_annotated_frame()
        if frame is None:
            return False
        
        try:
            cv2.imwrite(path, frame)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save frame: {e}")
            return False
    
    def process_to_intent(self, intent_engine=None) -> Dict[str, Any]:
        """
        Process current visual scene to intent format.
        
        Args:
            intent_engine: Optional IntentEngine instance for advanced processing
            
        Returns:
            Intent dictionary with visual information
        """
        objects = self.get_detected_objects()
        
        # Base intent structure
        intent = {
            "source": "vision",
            "timestamp": time.time(),
            "objects": objects,
            "intent": "visual_observation",
            "category": "perception",
            "confidence": 1.0 if objects else 0.5
        }
        
        # Determine primary objects (largest and most confident)
        if objects:
            # Sort by confidence and size
            primary_objects = sorted(
                objects, 
                key=lambda x: x["confidence"] * x["relative_size"], 
                reverse=True
            )[:3]
            
            # Add primary objects to intent
            intent["primary_objects"] = primary_objects
            
            # Extract classes for easy access
            intent["classes"] = [obj["class"] for obj in objects]
            intent["primary_classes"] = [obj["class"] for obj in primary_objects]
            
            # Determine scene type based on objects
            scene_type = self._determine_scene_type(objects)
            intent["scene_type"] = scene_type
        
        # Use intent engine if available for advanced processing
        if intent_engine and objects:
            try:
                # Create context for intent engine
                context = set(["visual_input", "object_detection"])
                
                # Create text description for intent engine
                description = self._create_scene_description(objects)
                
                # Process with intent engine
                processed_intent = intent_engine.process(description, context)
                
                # Merge the results, preserving visual metadata
                if processed_intent:
                    # Keep visual metadata
                    visual_info = {
                        k: v for k, v in intent.items() 
                        if k in ["objects", "primary_objects", "classes", 
                                "primary_classes", "scene_type", "source"]
                    }
                    
                    # Update with processed intent but preserve visual metadata
                    intent = processed_intent
                    intent.update(visual_info)
                    
            except Exception as e:
                self.logger.error(f"Intent engine processing failed: {e}")
        
        return intent
    
    def _determine_scene_type(self, objects: List[Dict[str, Any]]) -> str:
        """
        Determine the type of scene based on detected objects.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Scene type string
        """
        # Extract classes
        classes = [obj["class"] for obj in objects]
        
        # Define scene categories
        indoor_objects = {"chair", "sofa", "bed", "dining table", "toilet", "tv", "laptop", "book"}
        outdoor_objects = {"car", "bicycle", "motorcycle", "bus", "truck", "traffic light", "stop sign"}
        nature_objects = {"tree", "plant", "flower", "bird", "dog", "cat"}
        kitchen_objects = {"refrigerator", "microwave", "oven", "sink", "cup", "fork", "knife", "spoon", "bowl"}
        
        # Count objects in each category
        indoor_count = sum(1 for c in classes if c in indoor_objects)
        outdoor_count = sum(1 for c in classes if c in outdoor_objects)
        nature_count = sum(1 for c in classes if c in nature_objects)
        kitchen_count = sum(1 for c in classes if c in kitchen_objects)
        
        # Determine scene type
        if kitchen_count >= 2:
            return "kitchen"
        elif indoor_count > outdoor_count and indoor_count > nature_count:
            return "indoor"
        elif outdoor_count > indoor_count and outdoor_count > nature_count:
            return "outdoor"
        elif nature_count > indoor_count and nature_count > outdoor_count:
            return "nature"
        else:
            return "general"
    
    def _create_scene_description(self, objects: List[Dict[str, Any]]) -> str:
        """
        Create a natural language description of the scene.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Text description of the scene
        """
        if not objects:
            return "I don't see anything notable in the scene."
        
        # Count objects by class
        class_counts = {}
        for obj in objects:
            class_name = obj["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create description
        description_parts = []
        
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            if count == 1:
                description_parts.append(f"a {class_name}")
            else:
                description_parts.append(f"{count} {class_name}s")
        
        if len(description_parts) == 1:
            description = f"I see {description_parts[0]}."
        elif len(description_parts) == 2:
            description = f"I see {description_parts[0]} and {description_parts[1]}."
        else:
            description = "I see " + ", ".join(description_parts[:-1]) + f", and {description_parts[-1]}."
        
        return description
    
    def create_tag_from_intent(self, intent: Dict[str, Any]) -> Tag:
        """
        Create a Tag object from intent dictionary.
        
        Args:
            intent: Intent dictionary
            
        Returns:
            Tag object for tag_registry
        """
        return Tag(
            name=f"vision_intent_{int(time.time())}",
            intent=intent.get("intent", "visual_observation"),
            category=intent.get("category", "perception"),
            subcategory="vision_input",
            priority=intent.get("priority", 0.7),
            metadata={
                "objects": intent.get("primary_objects", []),
                "scene_type": intent.get("scene_type", "unknown"),
                "confidence": intent.get("confidence", 1.0),
                "source": "vision",
                "timestamp": intent.get("timestamp", time.time())
            }
        )
    
    def cleanup_cache(self) -> int:
        """
        Clean up old cache files.
        
        Returns:
            Number of files removed
        """
        try:
            removed = 0
            total_size = 0
            files = []
            
            # Get all cache files with their stats
            for path in self.cache_dir.glob("*.*"):
                if path.is_file():
                    stats = path.stat()
                    age_days = (time.time() - stats.st_mtime) / (24 * 3600)
                    size_mb = stats.st_size / (1024 * 1024)
                    total_size += size_mb
                    files.append((path, age_days, size_mb))
            
            # Remove old files
            for path, age_days, _ in files:
                if age_days > self.config.max_cache_age_days:
                    path.unlink()
                    removed += 1
            
            # If still over size limit, remove more files
            if total_size > self.config.max_cache_size_mb:
                # Re-check remaining files
                remaining = [f for f in files if f[0].exists()]
                remaining.sort(key=lambda x: x[1], reverse=True)  # Oldest first
                
                for path, _, size_mb in remaining:
                    if total_size <= self.config.max_cache_size_mb:
                        break
                    path.unlink()
                    total_size -= size_mb
                    removed += 1
            
            self.logger.info(f"Removed {removed} cached files")
            return removed
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
            return 0
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        self.stop_camera()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)