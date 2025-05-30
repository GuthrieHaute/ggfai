# GGFAI Framework

The **Grok & Guthrie Framework for AI (GGFAI)** is a modular, scalable framework for building custom home AI systems. It supports deployment on a wide range of hardware, from low-resource devices like Raspberry Pi to high-performance servers. Designed for accessibility, it enables beginners to deploy AI systems using a web-based interface, while offering advanced customization for experienced developers. The framework uses predefined slots, four tag-based trackers, and robust tag management to ensure safety, scalability, and maintainability. It integrates with GGUF models via Ollama for efficient local inference.

## Features
- **Modular Input Slots**: Predefined entry points for capturing user inputs, including voice (`voice.py`), text (`text.py`), sensors (`sensors.py`), gestures (`gesture.py`), VR (`vr.py`), biometrics (`biometric.py`), external data (`external.py`), and a web interface (`web_app.py`).
- **Intent Processing**: A core engine (`intent_engine.py`) leverages spaCy, Rasa, Transformers, and SpeechRecognition for multi-modal intent classification, excluding NLTK for simplicity.
- **Dynamic Agents**: Supports goal-driven agents with planning (`planning_service.py`), learning (`learning.py`), coordination (`coordinator.py`), explainability (`generate_explanation.py`), and tag evolution (`tag_analyzer.py`).
- **Tag-Based Trackers**: Four trackers manage system state: Intent (`intent_tracker.py`), Feature (`feature_tracker.py`), Context (`context_tracker.py`), and Analytics (`analytics_tracker.py`), with structured tag management.
- **Resource Management**: Includes predictive analytics (`resource_predictor.py`), resource profiling (`resource_profile_class.py`), and anomaly detection (`proactive_anomaly_detection.py`) for efficient operation.
- **Web Interface**: A browser-based UI (`web_app.py`) facilitates intent input, GGUF model selection, and tag visualization.
- **Ollama Integration**: Enables dynamic loading of GGUF models for local inference, enhancing flexibility across hardware.

## Project Structure
```
ggfai_framework/
├── entry_points/               # Input slots (voice.py, text.py, web_app.py, etc.)
├── ml_layer/                   # Intent engine (intent_engine.py) and agent logic
│   ├── agent/                  # Agent components (planning_service.py, learning.py, etc.)
│   └── models/                 # Pre-trained models (README.md)
├── trackers/                   # Tag trackers (intent_tracker.py, feature_tracker.py, etc.)
├── core/                       # Tag registry (tag_registry.py)
├── config/                     # Configuration (ai_prompt.txt)
├── static/                     # Web assets (style.css, app.js)
├── tests/                      # Unit tests (test_voice.py, test_web_app.py, etc.)
├── docs/                       # Documentation (architecture.md, setup_guide.md, etc.)
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

## Getting Started
### Prerequisites
- Python 3.8 or higher
- pip for installing dependencies
- Ollama (optional, for GGUF model inference)
- Pre-trained Rasa or Transformer models (optional, place in `ml_layer/models/`)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/ggfai_framework.git
   cd ggfai_framework
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Ollama Server** (if using GGUF models):
   ```bash
   ollama serve
   ```
4. **Verify Setup**:
   ```bash
   pytest tests/
   ```

### Running the Web Interface
1. **Start the Web Interface**:
   ```bash
   python entry_points/web_app.py
   ```
2. **Access the Interface**:
   Navigate to `http://localhost:8000` in a web browser to:
   - Input intents (e.g., "Play music")
   - Select GGUF models (if Ollama is enabled)
   - View tag dashboards and agent explanations

## Development
- **Entry Points**: Implement input processing in `entry_points/` (e.g., SpeechRecognition in `voice.py`, spaCy in `text.py`).
- **Intent Engine**: Enhance `intent_engine.py` with spaCy, Rasa, Transformers, and Ollama for intent classification.
- **Dynamic Agents**: Develop agent logic in `ml_layer/agent/` for planning, learning, coordination, and explainability.
- **Trackers**: Add storage and tag management logic to `trackers/` (e.g., `intent_tracker.py`).
- **Resource Management**: Utilize `resource_predictor.py`, `resource_profile_class.py`, and `proactive_anomaly_detection.py` for system optimization.
- **Web Interface**: Extend `web_app.py` for additional features like real-time intent processing or resource monitoring.
- **Tests**: Write unit tests in `tests/` and run with `pytest`.

## Documentation
- [Architecture](docs/architecture.md): Overview of components, data flow, and dynamic agents.
- [Setup Guide](docs/setup_guide.md): Instructions for installation, development, and deployment.
- [API Reference](docs/api_reference.md): Documentation for key classes and functions.
- [Contributing](docs/contributing.md): Guidelines for contributing to the project.

## Contributing
Contributions are welcome to enhance the GGFAI Framework. To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

Please include unit tests and update relevant documentation. Refer to the [Contributing Guide](docs/contributing.md) for details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- **GitHub Issues**: Report bugs or request features.
- **Community**: Engage with the GGFAI community for collaboration and support.

The GGFAI Framework aims to provide a robust and accessible platform for custom home AI development.