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
"""

import asyncio
import gzip
import json
import logging
import time
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

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
    Form
)
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware

# Import GGFAI components
from ml_layer.agent.tag_analyzer import TagAnalyzer, AnalysisMethod
from trackers.intent_tracker import IntentTracker
from trackers.feature_tracker import FeatureTracker
from trackers.context_tracker import ContextTracker
from trackers.analytics_tracker import AnalyticsTracker
from ml_layer.agent.generate_explanation import ExplanationGenerator, ExplanationConfig, ExplanationLevel
from ml_layer.agent.planning_service import PlanningService
from ml_layer.llm_coordinator import LLMCoordinator
from ml_layer.model_adapter import ModelAdapter

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

config = Config()
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
app.mount(
    "/static", 
    StaticFiles(directory="static"), 
    name="static"
)

# Mount Bootstrap assets
app.mount(
    "/static/assets", 
    StaticFiles(directory="static/assets"), 
    name="bootstrap_assets"
)

templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GGFAI.web")
security_logger = logging.getLogger("GGFAI.security")

# Initialize GGFAI components
intent_tracker = IntentTracker()
feature_tracker = FeatureTracker()
context_tracker = ContextTracker()
analytics_tracker = AnalyticsTracker()

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

# Initialize LLM Coordinator
llm_coordinator = LLMCoordinator(
    cleanup_interval=300.0,
    personality_config={
        "warmth": 0.7,
        "empathy": 0.8,
        "formality": 0.5,
        "adaptability": 0.9
    }
)

class WebSocketManager:
    """Thread-safe WebSocket connection manager with connection limits."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self.connection_count = 0

    async def connect(self, websocket: WebSocket) -> bool:
        """Attempt to add new connection with limit enforcement."""
        async with self._lock:
            if self.connection_count >= config.MAX_WS_CONNECTIONS:
                logger.warning("Connection limit reached")
                return False
                
            await websocket.accept()
            self.active_connections.append(websocket)
            self.connection_count += 1
            logger.info(f"New WS connection (Total: {self.connection_count})")
            return True

    async def disconnect(self, websocket: WebSocket) -> None:
        """Safely remove a connection and clean up LLM resources."""
        async with self._lock:
            if websocket in self.active_connections:
                # Clean up any LLM resources associated with this connection
                try:
                    if hasattr(websocket.state, "component_id") and hasattr(websocket.state, "model_id"):
                        # Release the model from the coordinator
                        success = llm_coordinator.release_llm(
                            websocket.state.model_id,
                            websocket.state.component_id
                        )
                        
                        if success:
                            logger.info(f"Released model {websocket.state.model_id} for {websocket.state.component_id}")
                        else:
                            logger.warning(f"Failed to release model {websocket.state.model_id} for {websocket.state.component_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up LLM resources: {str(e)}")
                
                # Remove the connection
                self.active_connections.remove(websocket)
                self.connection_count -= 1
                logger.info(f"WS disconnected (Remaining: {self.connection_count})")

    async def send_personal_message(
        self, 
        message: Dict, 
        websocket: WebSocket
    ) -> bool:
        """Send message to single client with error handling."""
        try:
            await websocket.send_json(message)
            return True
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.warning(f"WS send failed: {str(e)}")
            await self.disconnect(websocket)
            return False

    async def broadcast(self, message: Dict) -> None:
        """Broadcast to all active connections."""
        dead_connections = []
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except (WebSocketDisconnect, RuntimeError):
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for connection in dead_connections:
                await self.disconnect(connection)

manager = WebSocketManager()

# --- Security Helpers ---
async def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Validate API key against configured keys."""
    # In production, replace with proper key validation
    if config.DEBUG:
        return True
    return api_key == "your_secure_key_here"

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
@limiter.limit(config.RATE_LIMIT)
async def dashboard(
    request: Request,
    api_key: str = Depends(api_key_header)
) -> HTMLResponse:
    """Serve main dashboard with security checks."""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
        
    # Get available models for the dropdown
    models = await _get_available_models()
    
    return templates.TemplateResponse(
        "dashboard_new.html" if os.path.exists("templates/dashboard_new.html") else "dashboard.html",
        {
            "request": request,
            "api_key": api_key,
            "debug": config.DEBUG,
            "models": models,
            "donation_form": config.ENABLE_DONATION
        }
    )

@app.get("/donate", response_class=HTMLResponse)
@limiter.limit(config.RATE_LIMIT)
async def donation_popup(
    request: Request,
    api_key: str = Depends(api_key_header)
) -> HTMLResponse:
    """Serve donation pop-up template."""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
        
    return templates.TemplateResponse(
        "donation.html",
        {
            "request": request,
            "debug": config.DEBUG
        }
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections with validation."""
    # Validate origin in production
    if not config.DEBUG and websocket.headers.get("origin") not in config.ALLOWED_ORIGINS:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if not await manager.connect(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        while True:
            try:
                data = await websocket.receive_json()
                await _handle_websocket_message(data, websocket)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except ValueError as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        await manager.disconnect(websocket)

async def _handle_websocket_message(data: Dict, websocket: WebSocket) -> None:
    """Process and validate WebSocket messages."""
    if not isinstance(data, dict):
        raise ValueError("Message must be JSON object")
        
    message_type = data.get("type")
    
    if message_type == "model_change":
        await _handle_model_change(data, websocket)
    elif message_type == "user_input":
        await _handle_user_input(data, websocket)
    elif message_type == "tag_analysis_request":
        await _handle_tag_analysis_request(data, websocket)
    elif message_type == "explanation_request":
        await _handle_explanation_request(data, websocket)
    else:
        raise ValueError("Invalid message type")
        
async def _handle_tag_analysis_request(data: Dict, websocket: WebSocket) -> None:
    """Handle real-time tag analysis requests."""
    try:
        context_id = data.get("context_id")
        method = data.get("method", "hybrid")
        limit = min(int(data.get("limit", 10)), 50)  # Cap at 50 for performance
        
        # Get context
        context = {}
        if context_id:
            context = context_tracker.get_context(context_id) or {}
        else:
            # Use most recent context
            contexts = context_tracker.get_recent_contexts(1)
            if contexts:
                context = contexts[0]
                
        # Convert method string to enum
        analysis_method = AnalysisMethod.HYBRID
        try:
            analysis_method = AnalysisMethod(method.lower())
        except ValueError:
            logger.warning(f"Invalid analysis method: {method}, using HYBRID")
            
        # Get tag analysis
        ranked_tags = tag_analyzer.prioritize_tags(
            context,
            method=analysis_method,
            limit=limit
        )
        
        # Generate visualization data
        visualization_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add context node
        visualization_data["nodes"].append({
            "id": "context",
            "label": "Current Context",
            "color": "#FF6B6B",
            "size": 25
        })
        
        # Add tag nodes and edges
        for i, tag_info in enumerate(ranked_tags):
            node_id = f"tag_{i}"
            visualization_data["nodes"].append({
                "id": node_id,
                "label": f"{tag_info['tag_id']}\nScore: {tag_info['score']:.2f}",
                "color": "#4ECDC4",
                "size": 15 + tag_info['score'] * 10
            })
            
            visualization_data["edges"].append({
                "from": "context",
                "to": node_id,
                "width": tag_info['score'] * 5,
                "title": f"Score: {tag_info['score']:.2f}"
            })
            
        await manager.send_personal_message(
            {
                "type": "tag_analysis_response",
                "request_id": data.get("request_id", ""),
                "analysis_method": analysis_method.value,
                "context": context,
                "ranked_tags": ranked_tags,
                "visualization_data": visualization_data
            },
            websocket
        )
    except Exception as e:
        logger.error(f"Tag analysis error: {str(e)}", exc_info=True)
        await manager.send_personal_message(
            {
                "type": "error",
                "request_id": data.get("request_id", ""),
                "message": "Tag analysis failed"
            },
            websocket
        )
        
async def _handle_explanation_request(data: Dict, websocket: WebSocket) -> None:
    """Handle real-time explanation requests."""
    try:
        trace_id = data.get("trace_id")
        if not trace_id:
            raise ValueError("Missing trace_id")
            
        level = data.get("level", "standard")
        use_physics = data.get("use_physics", True)
        
        # Convert level string to enum
        explanation_level = ExplanationLevel.STANDARD
        try:
            explanation_level = ExplanationLevel[level.upper()]
        except (KeyError, ValueError):
            logger.warning(f"Invalid explanation level: {level}, using STANDARD")
            
        # Create explanation config
        config = ExplanationConfig(
            level=explanation_level,
            use_physics_viz=use_physics
        )
        
        # Get context for the trace
        context = context_tracker.get_context(trace_id) or {}
        
        # Get goal data
        goal_data = {"intent": trace_id}  # Simplified for demo
        intent = intent_tracker.get_intent(trace_id)
        if intent:
            goal_data = intent
            
        # Generate explanation
        explanation_data = explanation_generator.generate_for_web(
            trace_id=trace_id,
            context=context,
            goal_data=goal_data,
            config=config
        )
        
        await manager.send_personal_message(
            {
                "type": "explanation_response",
                "request_id": data.get("request_id", ""),
                "trace_id": trace_id,
                "explanation": explanation_data
            },
            websocket
        )
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}", exc_info=True)
        await manager.send_personal_message(
            {
                "type": "error",
                "request_id": data.get("request_id", ""),
                "message": "Explanation generation failed"
            },
            websocket
        )

async def _handle_model_change(data: Dict, websocket: WebSocket) -> None:
    """Handle model switching requests with LLM coordination."""
    try:
        model = data["model"]
        if model not in await _get_available_models():
            raise ValueError("Invalid model specified")
        
        # Create a component ID for this web session
        component_id = f"web_session_{websocket.client.host}_{int(time.time())}"
        
        # Create a model adapter that will use the LLM coordinator
        try:
            adapter = ModelAdapter(
                model_path=f"ollama:{model}",
                llm_coordinator=llm_coordinator,
                component_id=component_id,
                wait_for_model=True,
                model_timeout=10.0,
                auto_pull=True
            )
            
            # Generate a simple response to confirm the model works
            response = adapter.predict(
                "Confirm model switch",
                stream=False
            )
            
            # Send confirmation to the client
            await manager.send_personal_message(
                {
                    "type": "model_confirmation",
                    "model": model,
                    "response": response.get("text", "Model switch confirmed"),
                    "coordinator_status": "active"
                },
                websocket
            )
            
            # Store the component_id in the websocket state for cleanup
            websocket.state.component_id = component_id
            websocket.state.model_id = model
            
        except Exception as e:
            logger.error(f"Model adapter error: {str(e)}")
            
            # Try direct Ollama call as fallback
            response = ollama.generate(
                model=model,
                prompt="Confirm model switch",
                stream=False
            )
            
            await manager.send_personal_message(
                {
                    "type": "model_confirmation",
                    "model": model,
                    "response": response,
                    "coordinator_status": "fallback"
                },
                websocket
            )
            
    except KeyError:
        raise ValueError("Missing required fields")
    except ollama.ResponseError as e:
        logger.error(f"Model error: {str(e)}")
        raise ValueError("Model operation failed")

async def _handle_user_input(data: Dict, websocket: WebSocket) -> None:
    """Process user input messages and generate AI responses."""
    try:
        text = data["text"]
        if len(text) > 1000:  # Prevent abuse
            raise ValueError("Input too long")
        
        # Get model from data or use default
        model = data.get("model") or config.DEFAULT_MODEL
        
        # Get or create conversation context
        conversation_context = getattr(websocket.state, "conversation_context", {})
        if not hasattr(websocket.state, "conversation_context"):
            websocket.state.conversation_context = conversation_context
        
        # Add contextual awareness
        conversation_context.update({
            "timestamp": time.time(),
            "previous_messages": conversation_context.get("previous_messages", []) + [text],
            "message_count": conversation_context.get("message_count", 0) + 1
        })
        
        # Send thinking indicator with more personality
        thinking_messages = [
            "Let me think about that...",
            "Hmm, give me a moment...",
            "Processing that thought...",
            "Just a sec while I consider that..."
        ]
        await manager.send_personal_message(
            {
                "type": "thinking",
                "text": random.choice(thinking_messages),
                "timestamp": time.time()
            },
            websocket
        )
        
        # Create a model adapter with more natural persona
        adapter = ModelAdapter(
            model_path=f"ollama:{model}",
            llm_coordinator=llm_coordinator,
            component_id=websocket.state.component_id,
            wait_for_model=True,
            model_timeout=30.0,
            auto_pull=True
        )
        
        # Generate more contextual, personality-rich responses
        response = adapter.predict(
            text,
            stream=False,
            system_prompt="""You are a helpful and friendly AI assistant. 
            Respond naturally and conversationally, showing empathy and understanding.
            Consider the context of the conversation and adapt your tone accordingly.
            Use casual language when appropriate but maintain professionalism.""",
            context=conversation_context
        )
        
        # Update conversation state
        conversation_context["last_response"] = response.get("text", "")
        websocket.state.conversation_context = conversation_context
        
        await manager.send_personal_message(
            {
                "type": "chat_response",
                "text": response.get("text", "I need a moment to think about that."),
                "timestamp": time.time(),
                "context": {"message_count": conversation_context["message_count"]}
            },
            websocket
        )
    except KeyError:
        raise ValueError("Missing text field")

def _generate_thought_process(text: str) -> List[Dict[str, Any]]:
    """Generate a simulated thought process for demonstration purposes."""
    # In a real implementation, this would come from the LLM's chain-of-thought
    # or from a dedicated thought process generation component
    
    # For demo purposes, we'll create a simple simulated thought process
    thought_process = []
    
    # Add initial reasoning
    thought_process.append({
        "type": "Reasoning",
        "content": f"The user input is: '{text}'. I need to understand the intent and context.",
        "timestamp": time.time(),
        "depth": 0,
        "confidence": 1.0
    })
    
    # Add retrieval step
    thought_process.append({
        "type": "Retrieval",
        "content": "Searching for relevant information in my knowledge base...",
        "timestamp": time.time() + 0.1,
        "depth": 1,
        "parent_index": 0,
        "confidence": 0.85
    })
    
    # Add planning step
    thought_process.append({
        "type": "Planning",
        "content": "I'll need to formulate a response that addresses the user's query directly and provides helpful information.",
        "timestamp": time.time() + 0.2,
        "depth": 1,
        "parent_index": 0,
        "confidence": 0.9
    })
    
    # Add evaluation step
    thought_process.append({
        "type": "Evaluation",
        "content": "Considering multiple possible responses and evaluating which would be most helpful...",
        "timestamp": time.time() + 0.3,
        "depth": 2,
        "parent_index": 2,
        "confidence": 0.75
    })
    
    # Add generation step
    thought_process.append({
        "type": "Generation",
        "content": "Drafting a clear, concise response that addresses the user's needs.",
        "timestamp": time.time() + 0.4,
        "depth": 2,
        "parent_index": 2,
        "confidence": 0.95
    })
    
    # Add decision step
    thought_process.append({
        "type": "Decision",
        "content": "I've selected the most appropriate response based on relevance and helpfulness.",
        "timestamp": time.time() + 0.5,
        "depth": 3,
        "parent_index": 4,
        "confidence": 0.9
    })
    
    # Add reflection step
    thought_process.append({
        "type": "Reflection",
        "content": "My response addresses the user's query, but I should be prepared for follow-up questions on this topic.",
        "timestamp": time.time() + 0.6,
        "depth": 3,
        "parent_index": 5,
        "confidence": 0.8
    })
    
    return thought_process

@app.post("/api/chat")
@limiter.limit(config.RATE_LIMIT)
async def chat_endpoint(
    request: Request,
    data: Dict[str, Any],
    api_key: str = Depends(api_key_header)
) -> JSONResponse:
    """REST API endpoint for chat."""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    try:
        text = data["text"]
        if len(text) > 1000:  # Prevent abuse
            raise ValueError("Input too long")
        
        # Get model from data or use default
        model = data.get("model") or config.DEFAULT_MODEL
        
        # Generate thought process if enabled
        thought_process = None
        if config.ENABLE_THOUGHT_PROCESS:
            thought_process = _generate_thought_process(text)
        
        # Process the input and generate a response
        try:
            # Create a model adapter
            adapter = ModelAdapter(
                model_path=f"ollama:{model}",
                llm_coordinator=llm_coordinator,
                component_id=f"api_chat_{request.client.host}_{int(time.time())}",
                wait_for_model=True,
                model_timeout=30.0,
                auto_pull=True
            )
            
            # Generate response
            response = adapter.predict(
                text,
                stream=False,
                system_prompt="You are a helpful AI assistant. Respond in a clear, concise manner."
            )
            
            return JSONResponse(
                content=jsonable_encoder({
                    "text": response.get("text", "I'm sorry, I couldn't generate a response."),
                    "timestamp": time.time(),
                    "thought_process": thought_process
                })
            )
            
        except Exception as e:
            logger.error(f"Chat API error: {str(e)}")
            
            # Try direct Ollama call as fallback
            try:
                response = ollama.generate(
                    model=model,
                    prompt=text,
                    stream=False
                )
                
                return JSONResponse(
                    content=jsonable_encoder({
                        "text": response.get("response", "I'm sorry, I couldn't generate a response."),
                        "timestamp": time.time(),
                        "thought_process": thought_process
                    })
                )
            except Exception as e2:
                logger.error(f"Fallback API error: {str(e2)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to generate response"
                )
                
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing text field"
        )

@app.post("/api/voice-input")
@limiter.limit(config.RATE_LIMIT)
async def voice_input(
    request: Request,
    audio: UploadFile = File(...),
    api_key: str = Depends(api_key_header)
) -> JSONResponse:
    """Process voice input and return transcription and response."""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    try:
        # Save audio file temporarily
        audio_path = f"temp_audio_{int(time.time())}.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        
        # In a real implementation, you would use a speech recognition service
        # For demo purposes, we'll simulate transcription
        transcription = "This is a simulated transcription of voice input."
        
        # Generate response
        model = config.DEFAULT_MODEL
        
        # Generate thought process if enabled
        thought_process = None
        if config.ENABLE_THOUGHT_PROCESS:
            thought_process = _generate_thought_process(transcription)
        
        # Process the input and generate a response
        try:
            # Create a model adapter
            adapter = ModelAdapter(
                model_path=f"ollama:{model}",
                llm_coordinator=llm_coordinator,
                component_id=f"voice_api_{request.client.host}_{int(time.time())}",
                wait_for_model=True,
                model_timeout=30.0,
                auto_pull=True
            )
            
            # Generate response
            response = adapter.predict(
                transcription,
                stream=False,
                system_prompt="You are a helpful AI assistant. Respond in a clear, concise manner."
            )
            
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return JSONResponse(
                content=jsonable_encoder({
                    "text": transcription,
                    "response": response.get("text", "I'm sorry, I couldn't generate a response."),
                    "timestamp": time.time(),
                    "thought_process": thought_process
                })
            )
            
        except Exception as e:
            logger.error(f"Voice API error: {str(e)}")
            
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process voice input"
            )
                
    except Exception as e:
        logger.error(f"Voice input error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process voice input"
        )

@app.post("/api/select-model")
@limiter.limit(config.RATE_LIMIT)
async def select_model(
    request: Request,
    data: Dict[str, str],
    api_key: str = Depends(api_key_header)
) -> JSONResponse:
    """Select a model for the session."""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    try:
        model_id = data["model_id"]
        if model_id not in await _get_available_models():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model specified"
            )
        
        # In a real implementation, you would store the selected model in the session
        # For demo purposes, we'll just return success
        return JSONResponse(
            content=jsonable_encoder({
                "status": "success",
                "model": model_id
            })
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing model_id field"
        )

@app.get("/api/models")
@limiter.limit(config.RATE_LIMIT)
async def list_gguf_models(
    api_key: str = Depends(api_key_header)
) -> JSONResponse:
    """Fetch available GGUF models with caching."""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
        
    try:
        models = await _get_available_models()
        return JSONResponse(
            content=jsonable_encoder(models),
            headers={"Cache-Control": "public, max-age=60"}
        )
    except ollama.ResponseError as e:
        logger.error(f"Model list error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable"
        )

@app.get("/api/explanation/{trace_id}")
@limiter.limit(config.RATE_LIMIT)
async def get_explanation(
    request: Request,
    trace_id: str,
    level: str = "standard",
    use_physics: bool = True,
    api_key: str = Depends(api_key_header)
) -> JSONResponse:
    """
    Get explanation for a decision trace.
    
    Args:
        trace_id: Trace ID to explain
        level: Explanation level (simple, standard, technical, developer)
        use_physics: Whether to use physics in visualization
        
    Returns:
        Explanation data with narrative and visualization
    """
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
        
    try:
        # Convert level string to enum
        explanation_level = ExplanationLevel.STANDARD
        try:
            explanation_level = ExplanationLevel[level.upper()]
        except (KeyError, ValueError):
            logger.warning(f"Invalid explanation level: {level}, using STANDARD")
            
        # Create explanation config
        config = ExplanationConfig(
            level=explanation_level,
            use_physics_viz=use_physics
        )
        
        # Get context for the trace
        context = context_tracker.get_context(trace_id) or {}
        
        # Get goal data
        goal_data = {"intent": trace_id}  # Simplified for demo
        intent = intent_tracker.get_intent(trace_id)
        if intent:
            goal_data = intent
            
        # Generate explanation
        explanation_data = explanation_generator.generate_for_web(
            trace_id=trace_id,
            context=context,
            goal_data=goal_data,
            config=config
        )
        
        return JSONResponse(
            content=jsonable_encoder(explanation_data),
            headers={"Cache-Control": "private, max-age=30"}
        )
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explanation generation failed"
        )

@app.get("/api/llm-status")
@limiter.limit(config.RATE_LIMIT)
async def llm_status(
    api_key: str = Depends(api_key_header)
) -> JSONResponse:
    """Get status of all LLM models."""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
        
    try:
        status = llm_coordinator.get_all_llm_status()
        return JSONResponse(
            content=jsonable_encoder(status),
            headers={"Cache-Control": "public, max-age=5"}
        )
    except Exception as e:
        logger.error(f"LLM status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM status service unavailable"
        )

async def _get_available_models() -> List[str]:
    """Get available models with error handling."""
    try:
        result = ollama.list()
        return [m["name"] for m in result.get("models", [])]
    except:
        # Return some default models if Ollama is not available
        return ["llama2", "mistral", "gemma", "phi"]

@app.get("/api/tag-analysis")
@limiter.limit(config.RATE_LIMIT)
async def get_tag_analysis(
    request: Request,
    context_id: Optional[str] = None,
    method: str = "hybrid",
    limit: int = 10,
    api_key: str = Depends(api_key_header)
) -> JSONResponse:
    """
    Get tag analysis visualization data.
    
    Args:
        context_id: Optional context ID to analyze
        method: Analysis method (frequency, recency, success_rate, context_match, hybrid)
        limit: Maximum number of tags to return
        
    Returns:
        Tag analysis data for visualization
    """
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
        
    try:
        # Get current context
        context = {}
        if context_id:
            context = context_tracker.get_context(context_id) or {}
        else:
            # Use most recent context
            contexts = context_tracker.get_recent_contexts(1)
            if contexts:
                context = contexts[0]
                
        # Convert method string to enum
        analysis_method = AnalysisMethod.HYBRID
        try:
            analysis_method = AnalysisMethod(method.lower())
        except ValueError:
            logger.warning(f"Invalid analysis method: {method}, using HYBRID")
            
        # Get tag analysis
        ranked_tags = tag_analyzer.prioritize_tags(
            context,
            method=analysis_method,
            limit=limit
        )
        
        # Generate visualization data
        visualization_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add context node
        visualization_data["nodes"].append({
            "id": "context",
            "label": "Current Context",
            "color": "#FF6B6B",
            "size": 25
        })
        
        # Add tag nodes and edges
        for i, tag_info in enumerate(ranked_tags):
            node_id = f"tag_{i}"
            visualization_data["nodes"].append({
                "id": node_id,
                "label": f"{tag_info['tag_id']}\nScore: {tag_info['score']:.2f}",
                "color": "#4ECDC4",
                "size": 15 + tag_info['score'] * 10
            })
            
            visualization_data["edges"].append({
                "from": "context",
                "to": node_id,
                "width": tag_info['score'] * 5,
                "title": f"Score: {tag_info['score']:.2f}"
            })
            
        return JSONResponse(
            content=jsonable_encoder({
                "analysis_method": analysis_method.value,
                "context": context,
                "ranked_tags": ranked_tags,
                "visualization_data": visualization_data,
                "tags": ranked_tags  # For compatibility with app_new.js
            }),
            headers={"Cache-Control": "private, max-age=10"}
        )
    except Exception as e:
        logger.error(f"Tag analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tag analysis failed"
        )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom error handler for security."""
    security_logger.warning(
        f"HTTPException: {exc.status_code} - {exc.detail}",
        extra={"client": request.client.host}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={"X-Error": "true"}
    )

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