# Filepath: entry_points/video_processor.py
"""
Video Processing Module for GGFAI Framework

This module provides computer vision capabilities to the GGFAI framework,
enabling object detection, tracking, and scene understanding through
integration with YOLO models via the ModelAdapter.

Key features:
- Camera stream capture and processing
- Object detection using YOLO models
- Visual perception tag generation
- Integration with context tracking system
"""

import logging
import json
import os
import time
import threading
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path
import importlib.util
import queue

from ..core.tag_registry import Tag, TagStatus
from ..resource_management.hardware_shim import detect_hardware_tier
from ..trackers.context_tracker import ContextTracker
from ..ml_layer.model_adapter import ModelAdapter

# Configure logging
logger = logging.getLogger("GGFAI.video_processor")

# Check for OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not installed. Video processing will be limited to simulated data.")
    OPENCV_AVAILABLE = False

class VideoProcessorError(Exception):
    """Custom exception for video processing errors."""
    pass

class VideoProcessor:
    """
    Manages video capture, processing, and object detection.
    
    This class handles camera streams, applies computer vision models,
    and generates visual perception tags for the GGFAI framework.
    """
    
    def __init__(
        self,
        llm_coordinator,
        context_tracker: Optional[ContextTracker] = None,
        component_id: Optional[str] = None
    ):
        """
        Initialize the video processor.
        
        Args:
            llm_coordinator: LLMCoordinator instance for model management
            context_tracker: Optional context tracker for storing visual perception data
            component_id: Optional unique ID for this component, will be auto-generated if not provided
        """
        self.cameras: Dict[str, Dict] = {}
        self.camera_streams: Dict[str, Any] = {}
        self.detection_results: Dict[str, List[Dict]] = {}
        self.context_tracker = context_tracker
        self.config_path = Path(os.path.dirname(os.path.dirname(__file__))) / "config" / "cameras.json"
        self.config = self._load_config()
        self.hw_tier = detect_hardware_tier()
        self.running = False
        self.processing_threads = {}
        self._lock = threading.RLock()
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self._llm_coordinator = llm_coordinator
        self._component_id = component_id or f"video_processor_{int(time.time())}"
        
        # Initialize model adapter for YOLO
        self.model = None
        self._initialize_model()
        
        # Initialize cameras from config
        self._initialize_cameras()
        logger.info(f"Video processor initialized with {len(self.cameras)} cameras")
    
    def _load_config(self) -> Dict:
        """Load camera configuration from file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Camera config not found at {self.config_path}, using defaults")
                return {
                    "cameras": {},
                    "model": {
                        "path": "yolov8n.pt",
                        "confidence": 0.5,
                        "device": "cpu"
                    },
                    "processing": {
                        "frame_interval": 5,
                        "resolution": [640, 480]
                    }
                }
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded camera configuration with {len(config.get('cameras', {}))} cameras")
                return config
        except Exception as e:
            logger.error(f"Error loading camera config: {str(e)}", exc_info=True)
            return {
                "cameras": {},
                "model": {
                    "path": "yolov8n.pt",
                    "confidence": 0.5,
                    "device": "cpu"
                },
                "processing": {
                    "frame_interval": 5,
                    "resolution": [640, 480]
                }
            }
    
    def _initialize_model(self):
        """Initialize YOLO model via ModelAdapter."""
        try:
            model_config = self.config.get("model", {})
            model_path = model_config.get("path", "yolov8n.pt")
            
            # Create model adapter using LLMCoordinator
            self.model = ModelAdapter(
                model_path=model_path,
                llm_coordinator=self._llm_coordinator,
                component_id=self._component_id,
                model_timeout=10.0
            )
            
            logger.info(f"YOLO model initialized: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {str(e)}")
            self.model = None
    
    def _initialize_cameras(self):
        """Initialize cameras based on configuration."""
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available, using simulated cameras only")
            
        with self._lock:
            camera_configs = self.config.get("cameras", {})
            
            # If no cameras configured, add a simulated one
            if not camera_configs:
                camera_configs = {
                    "simulated": {
                        "type": "simulated",
                        "location": "living_room",
                        "resolution": [640, 480],
                        "fps": 5,
                        "enabled": True
                    }
                }
            
            # Create camera instances
            for camera_id, camera_config in camera_configs.items():
                self.cameras[camera_id] = {
                    "type": camera_config.get("type", "usb"),
                    "location": camera_config.get("location", "unknown"),
                    "resolution": camera_config.get("resolution", [640, 480]),
                    "fps": camera_config.get("fps", 5),
                    "source": camera_config.get("source", 0),  # Default to first camera
                    "enabled": camera_config.get("enabled", True),
                    "last_frame_time": None,
                    "frame_count": 0,
                    "detection_count": 0
                }
    
    def start(self):
        """Start video processing."""
        if self.running:
            return
        
        if not OPENCV_AVAILABLE and not any(c.get("type") == "simulated" for c in self.cameras.values()):
            logger.error("Cannot start video processing: OpenCV not available and no simulated cameras")
            return
        
        with self._lock:
            self.running = True
            
            # Start worker thread for processing frames
            self._start_processing_worker()
            
            # Start camera threads
            for camera_id, camera_config in self.cameras.items():
                if camera_config["enabled"]:
                    self._start_camera(camera_id)
            
            logger.info("Video processor started")
    
    def stop(self):
        """Stop video processing."""
        with self._lock:
            self.running = False
            
            # Release camera resources
            for camera_id, stream in self.camera_streams.items():
                if stream and hasattr(stream, "release"):
                    stream.release()
            
            self.camera_streams.clear()
            
            # Wait for threads to finish
            for thread in self.processing_threads.values():
                if thread.is_alive():
                    thread.join(timeout=1.0)
            
            self.processing_threads.clear()
            logger.info("Video processor stopped")
    
    def _start_camera(self, camera_id: str):
        """Start capture thread for a camera."""
        if camera_id in self.processing_threads and self.processing_threads[camera_id].is_alive():
            return
        
        camera_config = self.cameras[camera_id]
        
        def capture_loop():
            try:
                # Initialize camera
                if camera_config["type"] == "simulated":
                    # No actual capture device needed for simulated camera
                    stream = None
                    logger.info(f"Started simulated camera: {camera_id}")
                elif OPENCV_AVAILABLE:
                    # Open camera with OpenCV
                    stream = cv2.VideoCapture(camera_config["source"])
                    width, height = camera_config["resolution"]
                    stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    stream.set(cv2.CAP_PROP_FPS, camera_config["fps"])
                    
                    if not stream.isOpened():
                        logger.error(f"Failed to open camera: {camera_id}")
                        return
                    
                    logger.info(f"Started camera: {camera_id}")
                    self.camera_streams[camera_id] = stream
                else:
                    logger.error(f"Cannot start camera {camera_id}: OpenCV not available")
                    return
                
                # Frame processing loop
                frame_interval = self.config.get("processing", {}).get("frame_interval", 5)
                frame_count = 0
                
                while self.running:
                    try:
                        # Get frame
                        if camera_config["type"] == "simulated":
                            frame = self._generate_simulated_frame(camera_id)
                            success = True
                        else:
                            success, frame = stream.read()
                        
                        if not success:
                            logger.warning(f"Failed to read frame from camera: {camera_id}")
                            time.sleep(1)
                            continue
                        
                        # Process every nth frame to reduce CPU usage
                        frame_count += 1
                        if frame_count % frame_interval == 0:
                            # Add to processing queue
                            try:
                                self.frame_queue.put({
                                    "camera_id": camera_id,
                                    "frame": frame,
                                    "timestamp": datetime.utcnow()
                                }, block=False)
                            except queue.Full:
                                # Skip frame if queue is full
                                pass
                        
                        # Update stats
                        with self._lock:
                            self.cameras[camera_id]["frame_count"] += 1
                            self.cameras[camera_id]["last_frame_time"] = datetime.utcnow()
                        
                        # Sleep to maintain frame rate
                        time.sleep(1.0 / camera_config["fps"])
                        
                    except Exception as e:
                        logger.error(f"Error in camera loop for {camera_id}: {str(e)}")
                        time.sleep(1)
            
            finally:
                # Clean up resources
                if stream and hasattr(stream, "release"):
                    stream.release()
        
        thread = threading.Thread(target=capture_loop, daemon=True)
        thread.start()
        self.processing_threads[camera_id] = thread
    
    def _start_processing_worker(self):
        """Start worker thread for processing frames."""
        def processing_loop():
            while self.running:
                try:
                    # Get frame from queue
                    try:
                        frame_data = self.frame_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    
                    camera_id = frame_data["camera_id"]
                    frame = frame_data["frame"]
                    timestamp = frame_data["timestamp"]
                    
                    # Process frame with YOLO
                    detections = self._process_frame(camera_id, frame)
                    
                    # Create tags for detections
                    if detections:
                        self._create_detection_tags(camera_id, detections, timestamp)
                    
                    # Update detection count
                    with self._lock:
                        self.cameras[camera_id]["detection_count"] += len(detections)
                    
                    # Mark task as done
                    self.frame_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Error in processing worker: {str(e)}")
                    time.sleep(0.1)
        
        thread = threading.Thread(target=processing_loop, daemon=True)
        thread.start()
        self.processing_threads["processor"] = thread
    
    def _generate_simulated_frame(self, camera_id: str) -> Any:
        """Generate a simulated frame for testing."""
        if not OPENCV_AVAILABLE:
            # Return None if OpenCV is not available
            return None
        
        camera_config = self.cameras[camera_id]
        width, height = camera_config["resolution"]
        
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add camera ID
        cv2.putText(frame, f"Camera: {camera_id}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Simulate objects (randomly)
        import random
        import numpy as np
        
        # Occasionally add simulated objects
        if random.random() < 0.3:  # 30% chance of object
            # Draw a "person"
            x1 = random.randint(50, width - 100)
            y1 = random.randint(100, height - 200)
            x2 = x1 + random.randint(50, 100)
            y2 = y1 + random.randint(100, 200)
            
            # Draw rectangle for person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def _process_frame(self, camera_id: str, frame) -> List[Dict]:
        """
        Process a frame with YOLO model.
        
        Args:
            camera_id: Camera identifier
            frame: Image frame to process
            
        Returns:
            List of detection results
        """
        if self.model is None:
            return []
        
        try:
            # Get confidence threshold from config
            confidence = self.config.get("model", {}).get("confidence", 0.5)
            
            # Process with YOLO
            if hasattr(self.model, "predict"):
                # Direct Ultralytics YOLO
                results = self.model.predict(frame, conf=confidence, verbose=False)
                
                # Convert results to standard format
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        name = result.names[cls]
                        
                        detections.append({
                            "class": name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })
            else:
                # Use ModelAdapter
                input_data = {
                    "image": frame,
                    "confidence_threshold": confidence
                }
                
                result = self.model.predict(input_data)
                
                # Extract detections from result
                if "detections" in result:
                    detections = result["detections"]
                else:
                    detections = []
            
            # Store results
            with self._lock:
                self.detection_results[camera_id] = detections
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame from {camera_id}: {str(e)}")
            return []
    
    def _create_detection_tags(self, camera_id: str, detections: List[Dict], timestamp: datetime):
        """
        Create tags for object detections.
        
        Args:
            camera_id: Camera identifier
            detections: List of detection results
            timestamp: Time of detection
        """
        if not self.context_tracker:
            return
        
        try:
            camera_config = self.cameras[camera_id]
            location = camera_config["location"]
            
            # Create a visual perception tag
            perception_tag = Tag(
                name=f"visual_perception_{camera_id}_{timestamp.strftime('%Y%m%d%H%M%S')}",
                intent="visual_perception",
                category="perception",
                subcategory="visual",
                namespace="cameras",
                priority=0.7,  # Higher priority for visual perception
                metadata={
                    "camera_id": camera_id,
                    "location": location,
                    "timestamp": timestamp.isoformat(),
                    "detection_count": len(detections),
                    "detections": detections,
                    "hw_tier": self.hw_tier.name
                }
            )
            
            # Register with context tracker
            self.context_tracker.add_tag(perception_tag)
            
            # Create individual object tags for important classes
            important_classes = ["person", "car", "dog", "cat", "door", "window"]
            
            for detection in detections:
                obj_class = detection["class"]
                confidence = detection["confidence"]
                
                if obj_class in important_classes and confidence > 0.6:
                    object_tag = Tag(
                        name=f"object_{obj_class}_{camera_id}_{timestamp.strftime('%Y%m%d%H%M%S')}",
                        intent="object_detection",
                        category="object",
                        subcategory=obj_class,
                        namespace="cameras",
                        priority=0.8 if obj_class == "person" else 0.6,  # Higher priority for people
                        metadata={
                            "class": obj_class,
                            "confidence": confidence,
                            "camera_id": camera_id,
                            "location": location,
                            "timestamp": timestamp.isoformat(),
                            "bbox": detection["bbox"]
                        }
                    )
                    
                    # Register with context tracker
                    self.context_tracker.add_tag(object_tag)
            
        except Exception as e:
            logger.error(f"Error creating detection tags for {camera_id}: {str(e)}")
    
    def get_latest_detections(self, camera_id: str = None) -> Dict[str, List[Dict]]:
        """
        Get the latest detection results.
        
        Args:
            camera_id: Optional camera ID to filter results
            
        Returns:
            Dictionary of detection results by camera ID
        """
        with self._lock:
            if camera_id:
                return {camera_id: self.detection_results.get(camera_id, [])}
            else:
                return self.detection_results.copy()
    
    def get_camera_stats(self, camera_id: str = None) -> Dict[str, Dict]:
        """
        Get camera statistics.
        
        Args:
            camera_id: Optional camera ID to filter results
            
        Returns:
            Dictionary of camera statistics
        """
        with self._lock:
            if camera_id:
                if camera_id in self.cameras:
                    return {camera_id: self.cameras[camera_id].copy()}
                return {}
            else:
                return {k: v.copy() for k, v in self.cameras.items()}


# Global video processor instance
_video_processor = None

def get_video_processor(llm_coordinator, context_tracker=None) -> VideoProcessor:
    """
    Get or create the global video processor instance.
    
    Args:
        llm_coordinator: LLMCoordinator instance for model management
        context_tracker: Optional context tracker for storing visual perception data
        
    Returns:
        VideoProcessor instance
    """
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor(llm_coordinator, context_tracker)
    return _video_processor

def start_video_processing(llm_coordinator):
    """Start the video processing system."""
    processor = get_video_processor(llm_coordinator)
    processor.start()
    logger.info("Video processing started")

def stop_video_processing():
    """Stop the video processing system."""
    global _video_processor
    if _video_processor:
        _video_processor.stop()
        logger.info("Video processing stopped")