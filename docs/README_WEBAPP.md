# GGFAI Framework Web Interface

This document provides instructions for setting up and using the enhanced GGFAI Framework web interface.

## Overview

The GGFAI Framework web interface provides a modern, interactive dashboard for interacting with the AI system. It features:

- **Chat Interface**: Natural language interaction with the AI assistant
- **Tag Visualization**: Interactive 3D and network visualizations of the tag system
- **LLM Thought Process**: Visualization of the AI's reasoning process
- **System Monitoring**: Real-time monitoring of system resources and LLM status

## Setup Instructions

### Prerequisites

- Python 3.8+
- Ollama (for LLM support)
- Required Python packages (see `requirements.txt`)

### Installation

1. Make sure you have the latest files:
   - `templates/dashboard_new.html` (renamed to `dashboard.html`)
   - `static/app_new.js` (renamed to `app.js`)
   - `entry_points/web_app_new.py` (renamed to `web_app.py`)
   - `templates/donation.html`

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama service (if not already running):
   ```bash
   ollama serve
   ```

4. Pull required models:
   ```bash
   ollama pull llama2
   # Optional: Pull additional models
   ollama pull mistral
   ollama pull gemma
   ```

5. Start the web server:
   ```bash
   python entry_points/web_app.py
   ```

6. Access the dashboard at `http://localhost:8000`

## Using the Web Interface

### Chat Interface

- Type messages in the chat input box and press Enter or click the send button
- The AI will respond with natural language
- View the AI's thought process in the "LLM Thought Process" tab

### Tag Visualization

- The "Tag Visualization" tab shows two visualizations:
  - 3D Tag Cloud: Interactive 3D visualization of tags
  - Tag Network: Network graph showing relationships between tags
- Click on tags to view explanations
- Use the settings modal to customize the visualization

### LLM Thought Process

- The "LLM Thought Process" tab shows the AI's reasoning process
- Click on thought nodes to view details
- Export thought processes for analysis

### System Monitoring

- The "System" tab shows resource usage and LLM status
- View system logs for debugging

## Configuration

The web interface can be configured by modifying the `Config` class in `web_app.py`:

```python
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
```

## Troubleshooting

### WebSocket Connection Issues

If you experience WebSocket connection issues:

1. Check that the server is running
2. Ensure your browser supports WebSockets
3. Check for CORS issues if accessing from a different domain

### LLM Issues

If LLM models are not working:

1. Ensure Ollama is running (`ollama serve`)
2. Check that required models are pulled (`ollama list`)
3. Check server logs for errors

### Visualization Issues

If visualizations are not rendering:

1. Check browser console for JavaScript errors
2. Ensure required libraries are loaded
3. Try refreshing the page

## Development

To extend the web interface:

1. Add new API endpoints in `web_app.py`
2. Update the frontend in `app.js`
3. Modify the HTML templates as needed

## Security Considerations

For production deployment:

1. Set `DEBUG = False` in the `Config` class
2. Implement proper API key validation
3. Set `ENABLE_HTTPS = True` and configure SSL certificates
4. Set appropriate `ALLOWED_ORIGINS` for CORS protection

## License

This software is provided under the MIT License. See the LICENSE file for details.