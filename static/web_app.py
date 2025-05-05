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
8. Added donation pop-up link and template
"""

import asyncio
import gzip
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import ollama
import uvicorn
from fastapi import (
    FastAPI, 
    WebSocket, 
    WebSocketDisconnect, 
    Request, 
    HTTPException,
    status
)
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware

# Configuration
class Config:
    DEBUG = False
    API_KEY_NAME = "X-API-KEY"
    MAX_WS_CONNECTIONS = 100
    WS_MAX_SIZE = 2 ** 20  # 1MB
    COMPRESSION_MIN_SIZE = 500
    RATE_LIMIT = "100/minute"

config = Config()
api_key_header = APIKeyHeader(name=config.API_KEY_NAME, auto_error=False)

# Initialize FastAPI
app = FastAPI(
    title="GGFAI Web Interface",
    description="Secure real-time control plane for GGFAI",
    version="2.0",
    docs_url=None if not config.DEBUG else "/docs",
    redoc_url=None
)

# Security middleware
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"] if not config.DEBUG else ["*"],
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
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GGFAI.web")
security_logger = logging.getLogger("GGFAI.security")

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
                        # Import here to avoid circular imports
                        from ml_layer.llm_coordinator import llm_coordinator
                        
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
    """Serve main dashboard with security checks and donation link."""
    if not await validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
        
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "minified": not config.DEBUG,
            "donation_form": True  # Enable donation link
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
            "minified": not config.DEBUG
        }
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections with validation."""
    # Validate origin in production
    if not config.DEBUG and websocket.headers.get("origin") not in ["https://yourdomain.com"]:
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
        await _handle_user_input(data)
    else:
        raise ValueError("Invalid message type")

async def _handle_model_change(data: Dict, websocket: WebSocket) -> None:
    """Handle model switching requests with LLM coordination."""
    try:
        # Import here to avoid circular imports
        from ml_layer.llm_coordinator import llm_coordinator
        from ml_layer.model_adapter import ModelAdapter
        
        model = data["model"]
        if model not in await _get_available_models():
            raise ValueError("Invalid model specified")
        
        # Create a component ID for this web session
        component_id = f"web_session_{websocket.client.host}_{int(time.time())}"
        
        # Create a model adapter that will use the LLM coordinator
        try:
            adapter = ModelAdapter(
                model_path=f"ollama:{model}",
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

async def _handle_user_input(data: Dict) -> None:
    """Process user input messages."""
    try:
        text = data["text"]
        if len(text) > 1000:  # Prevent abuse
            raise ValueError("Input too long")
            
        await manager.broadcast({
            "type": "intent_update",
            "input": text[:1000],  # Enforce limit
            "timestamp": time.time()
        })
    except KeyError:
        raise ValueError("Missing text field")

@app.get("/api/models")
@limiter.limit(config.RATE_LIMIT)
async def list_gguf_models() -> JSONResponse:
    """Fetch available GGUF models with caching."""
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

async def _get_available_models() -> List[str]:
    """Get available models with error handling."""
    result = ollama.list()
    return [m["name"] for m in result.get("models", [])]

@app.get("/api/llm-status")
@limiter.limit(config.RATE_LIMIT)
async def llm_status() -> JSONResponse:
    """Get status of all LLM models."""
    try:
        # Import here to avoid circular imports
        from ml_layer.llm_coordinator import llm_coordinator
        
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