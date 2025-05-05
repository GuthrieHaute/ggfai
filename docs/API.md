# GGFAI Framework API Documentation

## Component Interfaces

### Entry Points

#### Voice Interface (`entry_points/voice.py`)
```python
class VoiceProcessor:
    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process raw audio data and convert to text/intent"""
    
    def start_listening(self) -> None:
        """Start continuous audio capture"""
    
    def stop_listening(self) -> None:
        """Stop audio capture"""
```

#### Video Processing (`entry_points/video_processor.py`)
```python
class VisionProcessor:
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process a single video frame"""
    
    def start_capture(self, device_id: int = 0) -> None:
        """Start video capture from specified device"""
```

#### Web Interface (`entry_points/web_app.py`)
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """REST endpoint for chat messages"""
```

### Core Framework

#### Intent Engine (`ml_layer/intent_engine.py`)
```python
class IntentEngine:
    def process_intent(self, text: str, context: Dict[str, Any]) -> Intent:
        """Process text input and return structured intent"""
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze emotional tone of text"""
```

#### Tag Registry (`core/tag_registry.py`)
```python
class TagRegistry:
    def register_tag(self, tag: Tag) -> None:
        """Register a new tag"""
    
    def query_tags(self, filter_criteria: Dict[str, Any]) -> List[Tag]:
        """Query tags matching criteria"""
```

## Data Structures

### Tag
```python
class Tag(BaseModel):
    id: str
    type: TagType
    value: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    source: str
```

### Intent
```python
class Intent(BaseModel):
    type: str
    confidence: float
    entities: List[Entity]
    context: Dict[str, Any]
    raw_text: str
```

### Context
```python
class Context(BaseModel):
    tags: List[Tag]
    state: Dict[str, Any]
    history: List[HistoryItem]
```

## Extension Points

### Custom Entry Points
To create a new entry point:

1. Create a new class implementing `ComponentInterface`
2. Register with the framework using `@register_component`
3. Implement required methods:
   - `initialize()`
   - `process_input()`
   - `cleanup()`

Example:
```python
@register_component
class CustomEntryPoint(ComponentInterface):
    def initialize(self) -> None:
        # Setup code
        
    def process_input(self, data: Any) -> Dict[str, Any]:
        # Process input data
        
    def cleanup(self) -> None:
        # Cleanup code
```

### Custom Models
To add a new ML model:

1. Implement the `ModelAdapter` interface
2. Register the model in `config/models.json`
3. Implement model-specific processing logic

Example:
```python
class CustomModelAdapter(ModelAdapter):
    def load_model(self) -> None:
        # Load model
        
    def predict(self, input_data: Any) -> Any:
        # Make predictions
        
    def cleanup(self) -> None:
        # Cleanup resources
```

## Configuration

### Model Configuration (`config/models.json`)
```json
{
    "model_id": {
        "type": "custom",
        "path": "path/to/model",
        "adapter": "CustomModelAdapter",
        "parameters": {
            "param1": "value1"
        }
    }
}
```

### Device Configuration (`config/devices.json`)
```json
{
    "device_id": {
        "type": "camera",
        "index": 0,
        "resolution": [640, 480],
        "fps": 30
    }
}
```

## Event System

### Publishing Events
```python
event_system.publish("topic", {
    "type": "event_type",
    "data": event_data,
    "timestamp": datetime.now()
})
```

### Subscribing to Events
```python
@event_system.subscribe("topic")
def handle_event(event: Dict[str, Any]) -> None:
    # Handle event
```

## Best Practices

1. **Error Handling**
   - Use appropriate exception types
   - Always cleanup resources
   - Implement graceful degradation

2. **Performance**
   - Cache expensive computations
   - Use async where appropriate
   - Implement resource limits

3. **Security**
   - Validate all inputs
   - Use secure defaults
   - Implement rate limiting

4. **Testing**
   - Write unit tests for components
   - Use integration tests
   - Mock external dependencies

## Development Setup

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Running Tests**
```bash
pytest tests/
```

3. **Code Style**
```bash
black .
flake8
mypy .
```

## Debugging

1. **Logging**
   - Use the built-in logger
   - Set appropriate log levels
   - Include context in log messages

2. **Monitoring**
   - Check resource usage
   - Monitor event queues
   - Track model performance

3. **Troubleshooting**
   - Check logs in `logs/`
   - Verify configurations
   - Test components in isolation