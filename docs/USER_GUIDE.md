# GGFAI Framework User Guide

## Overview

The GGFAI Framework is a powerful AI assistant system that provides natural interaction through multiple interfaces. This guide will help you get started and make the most of its features.

## Getting Started

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Webcam (optional, for video features)
- Microphone (optional, for voice features)

### Installation

1. Install prerequisites:
```bash
# Windows
python -m pip install --upgrade pip
pip install virtualenv

# Linux/macOS
python3 -m pip install --upgrade pip
pip3 install virtualenv
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama (for LLM support):
   - Visit [ollama.ai](https://ollama.ai) for installation instructions
   - Run: `ollama pull llama2`

## Using the Web Interface

### Starting the Server

1. Start Ollama service:
```bash
ollama serve
```

2. Start the web interface:
```bash
python entry_points/web_app.py
```

3. Access the dashboard at `http://localhost:8000`

### Dashboard Features

#### Chat Interface
- Type messages in the chat input box
- Use voice input by clicking the microphone button
- View chat history and clear when needed
- Supports markdown and code highlighting

#### Tag Visualization
- 3D Tag Cloud shows active tags in the system
- Tag Network displays relationships between tags
- Click tags to view detailed explanations
- Use filters to focus on specific tag types

#### LLM Thought Process
- View the AI's reasoning process in real-time
- Expand nodes to see detailed explanations
- Export thought processes for analysis
- Toggle auto-update for live updates

#### System Monitoring
- View resource usage (CPU, Memory, GPU)
- Monitor LLM status and performance
- Check system logs for troubleshooting
- Track active connections and requests

### Voice Interface

1. Enable microphone access when prompted
2. Click the microphone button to start voice input
3. Speak naturally - the system will process your speech
4. Click again to stop recording

### Configuration Options

#### Model Selection
- Choose from available LLM models
- Default: llama2
- Options depend on installed models

#### Explanation Levels
- Standard: Basic explanations
- Technical: Detailed technical information
- Developer: Implementation details

#### Visualization Settings
- Animation speed
- Node size
- Physics simulation
- Label visibility

## Working with Tags

Tags are the core mechanism for tracking context and state in GGFAI.

### Tag Types

1. **Intent Tags**
   - Generated from user input
   - Represent desired actions/queries
   - Include confidence scores

2. **Context Tags**
   - Track environmental state
   - Store temporal information
   - Maintain conversation context

3. **Visual Tags**
   - Generated from video input
   - Object detection results
   - Scene understanding

### Tag Operations

- View active tags in visualizations
- Query tag history
- Filter by type/source
- Export tag data

## Voice Commands

Common voice commands:

- "What can you do?"
- "Show system status"
- "Explain [tag name]"
- "Clear chat history"
- "Change explanation level to [level]"

## Troubleshooting

### Common Issues

1. **Connection Problems**
   - Verify Ollama is running
   - Check network connectivity
   - Clear browser cache

2. **Voice Recognition Issues**
   - Check microphone permissions
   - Verify microphone input levels
   - Try in a quieter environment

3. **Performance Issues**
   - Check resource usage
   - Reduce visualization complexity
   - Clear browser cache
   - Restart the server

### Logs

- Server logs: `logs/server.log`
- Event logs: `logs/events.log`
- Error logs: `logs/error.log`

### Getting Help

- Check documentation in `docs/`
- View system status page
- Contact support team

## Best Practices

1. **Optimal Usage**
   - Use natural language
   - Provide context when needed
   - Monitor resource usage

2. **Performance**
   - Clear chat history periodically
   - Disable unused visualizations
   - Use appropriate explanation levels

3. **Security**
   - Keep API keys secure
   - Use HTTPS in production
   - Monitor access logs

## Advanced Features

### Custom Extensions

1. Add custom entry points
2. Integrate new ML models
3. Create custom visualizations

### Integration

- REST API endpoints
- WebSocket connections
- Event system hooks

### Automation

- Scheduled tasks
- Event triggers
- Custom workflows

## Updates and Maintenance

### Updating

1. Pull latest changes
2. Update dependencies
3. Run database migrations
4. Restart services

### Backup

- Regular configuration backups
- Database backups
- Log rotation

### Monitoring

- Resource usage
- Error rates
- Response times
- Model performance