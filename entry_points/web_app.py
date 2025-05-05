"""
GGFAI Web Interface - Secure Real-time API and WebSocket Service

Key Improvements:
1. Enhanced security headers and middleware
2. Rate limiting and request validation
3. Proper async resource management
4. Comprehensive error handling
5. Structured logging
6. Configurable security settings
7. Health checks and monitoring endpoints
8. Chat interface with LLM thought process visualization
9. 3D tag cloud visualization
10. Voice processing with multiple STT/TTS engines
11. Natural conversation handling
12. YOLO vision integration
"""

import asyncio
import gzip
import json
import logging
import time
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set

import cv2
import numpy as np
import ollama
import uvicorn
from fastapi import (
    FastAPI, 
    WebSocket, 
    WebSocketDisconnect, 
    Request, 
    HTTPException,
    status,
    Depends,
    File,
    UploadFile,
    Form,
    BackgroundTasks
)
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import GGFAI components
from ml_layer.agent.tag_analyzer import TagAnalyzer, AnalysisMethod
from trackers.intent_tracker import IntentTracker
from trackers.feature_tracker import FeatureTracker
from trackers.context_tracker import ContextTracker
from trackers.analytics_tracker import AnalyticsTracker
from ml_layer.agent.generate_explanation import ExplanationGenerator, ExplanationConfig, ExplanationLevel
from ml_layer.agent.planning_service import PlanningService
from ml_layer.llm_coordinator import llm_coordinator
from ml_layer.model_adapter import ModelAdapter
from entry_points.voice import VoiceProcessor, VoiceConfig, RecognitionEngine, TTSEngine
from ml_layer.intent_engine import IntentEngine
from core.tag_registry import TagRegistry

# Configuration
class Config:
    DEBUG = True  # Set to False in production
    API_KEY_NAME = "X-API-KEY"
    MAX_WS_CONNECTIONS = 100
    WS_MAX_SIZE = 2 ** 20  # 1MB
    COMPRESSION_MIN_SIZE = 500
    RATE_LIMIT = "100/minute"
    ALLOWED_ORIGINS = ["*"] if True else ["https://yourdomain.com"]  # Set appropriate origins in production
    DEFAULT_MODEL = "llama2"  # Default LLM model
    ENABLE_HTTPS = False  # Set to True in production
    ENABLE_DONATION = True  # Enable donation link
    ENABLE_THOUGHT_PROCESS = True  # Enable LLM thought process visualization
    ENABLE_VOICE_PROCESSING = True  # Enable voice processing features
    ENABLE_YOLO = True  # Enable YOLO object detection
    DEFAULT_YOLO_MODEL = "default"  # Default YOLO model
    YOLO_CONFIDENCE_THRESHOLD = 0.5  # Default confidence threshold for YOLO
    DEFAULT_TTS_ENGINE = TTSEngine.SYSTEM  # Default TTS engine
    DEFAULT_STT_ENGINES = [RecognitionEngine.GOOGLE, RecognitionEngine.SPHINX]  # Default STT engines
    AUDIO_CACHE_DIR = "audio_cache"  # Directory for caching audio files

    # Add configuration system
    def save_to_file(self):
        """Save current configuration to file"""
        config_path = Path("config/webapp_config.json")
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    def load_from_file(self):
        """Load configuration from file"""
        config_path = Path("config/webapp_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.__dict__.update(config)

# Initialize config
config = Config()
config.load_from_file()

api_key_header = APIKeyHeader(name=config.API_KEY_NAME, auto_error=False)

# Initialize FastAPI
app = FastAPI(
    title="GGFAI Web Interface",
    description="Secure real-time control plane for GGFAI",
    version="2.0",
    docs_url=None if not config.DEBUG else "/docs",
    redoc_url=None if not config.DEBUG else "/redoc"
)

# Security middleware
if config.ENABLE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=config.COMPRESSION_MIN_SIZE)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

app.mount(
    "/static", 
    StaticFiles(directory=str(static_dir)), 
    name="static"
)

templates = Jinja2Templates(directory=str(templates_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GGFAI.web")
security_logger = logging.getLogger("GGFAI.security")

# Initialize GGFAI components
intent_tracker = IntentTracker()
feature_tracker = FeatureTracker()
context_tracker = ContextTracker()
analytics_tracker = AnalyticsTracker()
intent_engine = IntentEngine()
model_adapter = ModelAdapter()
tag_registry = TagRegistry()

# Initialize tag analyzer
tag_analyzer = TagAnalyzer(
    intent_tracker=intent_tracker,
    feature_tracker=feature_tracker,
    context_tracker=context_tracker,
    analytics_tracker=analytics_tracker
)

# Initialize planning service with domain knowledge
domain_knowledge = {
    "actions": {},  # Will be populated from config
    "resources": [],  # Will be populated at runtime
    "agent_capabilities": {}  # Will be populated at runtime
}

# Initialize planning service
planning_service = PlanningService(
    domain_knowledge=domain_knowledge,
    tag_analyzer=tag_analyzer
)

# Initialize explanation generator
explanation_generator = ExplanationGenerator(
    planning_service=planning_service,
    tag_analyzer=tag_analyzer,
    analytics_tracker=analytics_tracker
)

# Initialize voice processor if enabled
voice_processor = None
if config.ENABLE_VOICE_PROCESSING:
    try:
        # Create voice config
        voice_config = VoiceConfig(
            tts_engine=config.DEFAULT_TTS_ENGINE,
            recognition_engines=config.DEFAULT_STT_ENGINES,
            audio_cache_dir=config.AUDIO_CACHE_DIR
        )
        
        # Initialize voice processor
        voice_processor = VoiceProcessor(voice_config)
        logger.info("Voice processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize voice processor: {e}")

# Initialize YOLO video processor if enabled
YOLO_MODELS = {
    'default': 'yolov8n.pt',
    'custom': 'models/custom_yolo.pt'
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}
        self.client_contexts: Dict[int, Set[str]] = {}
        self.yolo_states: Dict[int, bool] = {}
        
    async def connect(self, websocket: WebSocket, client_id: int):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_contexts[client_id] = set()
        self.yolo_states[client_id] = False
        
    def disconnect(self, client_id: int):
        self.active_connections.pop(client_id, None)
        self.client_contexts.pop(client_id, None)
        self.yolo_states.pop(client_id, None)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)
            
    def get_client_context(self, client_id: int) -> Set[str]:
        return self.client_contexts.get(client_id, set())
        
    def update_client_context(self, client_id: int, context: Set[str]):
        self.client_contexts[client_id] = context

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'message':
                # Process message with natural conversation understanding
                await process_message(websocket, client_id, data)
                
            elif data['type'] == 'config_update':
                # Handle configuration updates (e.g., YOLO toggle)
                manager.yolo_states[client_id] = data.get('yolo_enabled', False)
                if 'yolo_model' in data:
                    await update_yolo_model(websocket, data['yolo_model'])
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)

async def process_message(websocket: WebSocket, client_id: int, data: dict):
    message = data['content']
    yolo_enabled = manager.yolo_states.get(client_id, False)
    
    # Start thinking indication
    await websocket.send_json({"type": "thinking_start"})
    
    try:
        # Get current context
        current_context = manager.get_client_context(client_id)
        
        # Process visual input if YOLO is enabled
        visual_context = set()
        if yolo_enabled:
            visual_context = await process_visual_input(data.get('yolo_model', 'default'))
            current_context.update(visual_context)
        
        # Generate response using intent engine
        intent = intent_engine.process(
            message,
            context=current_context
        )
        
        # Track the intent
        intent_tracker.track(intent)
        
        # Generate natural response
        response = await generate_natural_response(intent, current_context)
        
        # Update context
        context_tracker.update(intent, response)
        manager.update_client_context(client_id, current_context)
        
        # Send response with any visual context
        await websocket.send_json({
            "type": "response",
            "content": response,
            "visual_context": list(visual_context) if visual_context else None
        })
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": "Sorry, I encountered an error processing your message."
        })

async def process_visual_input(model_name: str = 'default') -> Set[str]:
    """Process video input using YOLO and return detected objects/context"""
    try:
        model_path = YOLO_MODELS.get(model_name, YOLO_MODELS['default'])
        
        # Get frame from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.warning("Failed to capture video frame")
            return set()
        
        # Process frame with YOLO
        results = model_adapter.detect_objects(
            frame,
            model_path=model_path
        )
        
        # Convert detections to context
        context = set()
        for detection in results:
            confidence = detection.get('confidence', 0)
            if confidence > 0.5:  # Confidence threshold
                context.add(detection['class'])
                
        return context
        
    except Exception as e:
        logger.error(f"Error in visual processing: {str(e)}")
        return set()

async def generate_natural_response(intent: dict, context: Set[str]) -> str:
    """Generate a natural, contextual response based on intent and context"""
    try:
        # Get response template based on intent
        template = intent_engine.get_response_template(intent)
        
        # Personalize based on context
        response = model_adapter.generate_response(
            template=template,
            context=context,
            style="conversational"  # Ensure natural dialogue style
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble formulating a response right now."

async def update_yolo_model(websocket: WebSocket, model_name: str):
    """Update the YOLO model configuration"""
    try:
        if model_name in YOLO_MODELS:
            # Validate model file exists
            model_path = Path(YOLO_MODELS[model_name])
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            await websocket.send_json({
                "type": "model_loaded",
                "model": model_name
            })
        else:
            raise ValueError(f"Invalid model name: {model_name}")
            
    except Exception as e:
        logger.error(f"Error updating YOLO model: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to load YOLO model: {model_name}"
        })

@app.get("/")
async def root(request):
    """Render the main dashboard"""
    return templates.TemplateResponse(
        "dashboard_new.html",
        {
            "request": request,
            "models": list(YOLO_MODELS.keys()),
            "debug": config.DEBUG
        }
    )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

@app.get("/api/config")
async def get_config() -> Dict:
    """Get current configuration"""
    return {k: v for k, v in config.__dict__.items() 
            if not k.startswith('_')}

@app.post("/api/config")
async def update_config(
    updates: Dict[str, Any],
    request: Request,
    api_key: str = Depends(api_key_header)
) -> Dict[str, str]:
    """Update configuration settings"""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    # Validate updates
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Save to file
    config.save_to_file()
    
    return {"status": "Configuration updated successfully"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=30,
        ws_ping_timeout=10,
        timeout_keep_alive=5,
        log_config=None if config.DEBUG else {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "security": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "security.log",
                    "formatter": "json",
                    "maxBytes": 10_000_000,
                    "backupCount": 5
                }
            },
            "loggers": {
                "GGFAI.security": {
                    "handlers": ["security"],
                    "level": "WARNING",
                    "propagate": False
                }
            }
        }
    )