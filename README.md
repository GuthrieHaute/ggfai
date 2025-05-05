# GGF AI Framework

## Core Principles

* **Natural Conversational Interface:** All primary interaction with the AI **must** occur through fluid, contextual, natural language (voice/text). Rigid command syntaxes are explicitly disallowed in these primary channels.
* **Unified AI Persona:** The AI **must** present itself to the user as a single, cohesive intelligence. Internal components like agents and ML models are strictly implementation details, never exposed conceptually to the end-user.
* **Pervasive Intelligence:** Agents and ML capabilities are engineered to be deployed ubiquitously behind the scenes, fulfilling any role necessary to maximize the AI's autonomy, proactivity, and effectiveness.
* **Democratized Power:** The framework is designed to be accessible, enabling powerful, custom AI on common hardware for everyone, thereby leveling the playing field for AI-driven assistance and automation.
* **Extreme Modularity & Flexibility:** The architecture mandates deep modularity, empowering developers to easily combine, customize, and extend components to build diverse, powerful AI solutions that deliver a seamless user experience.

---

**Vision:** To establish the definitive standard for intuitive, powerfully intelligent home AI assistants that operate according to the Core Principles above. GGF AI provides the foundation for creating AI that seamlessly understands user desires expressed through natural dialogue and orchestrates its pervasive internal capabilities to fulfill requests proactively and intelligently.

**Engineered for Flexibility & Seamless Experience:** The **GGF AI Framework (GGFAI)** embodies the Core Principles through its highly modular and adaptable architecture. This structure empowers developers to deploy agents and ML capabilities into any role, constructing sophisticated AI solutions that present the mandated unified, intuitive conversational interface. This approach is designed to enable users to define, automate, and execute complex tasks and workflows purely through natural conversation, surpassing traditional low-code/no-code paradigms.

Built for scalability, GGFAI supports deployment on diverse hardware, from resource-constrained devices to high-performance servers, ensuring broad accessibility. While enabling rapid development via its web interface and pre-built components, it provides the necessary depth for advanced customization. Foundational elements like predefined input slots, tag-based state tracking, and robust tag management ensure safety, scalability, and maintainability. Integration with Ollama provides efficient local inference capabilities using GGUF models.

## Features

* **Modular Input Slots**: Standardized entry points (`entry_points/`) for diverse interactions: voice (`voice.py`), text (`text.py`), sensors (`sensors.py`), gestures (`gesture.py`), VR (`vr.py`), biometrics (`biometric.py`), external data (`external.py`), and web (`web_app.py`).
* **Natural Intent Understanding**: The core engine (`ml_layer/intent_engine.py`) performs deep semantic interpretation of **natural conversation exclusively**. Leveraging advanced libraries (`spaCy`, `Rasa`, `Transformers`, `SpeechRecognition`) and Ollama GGUF integration, it processes user intent fluidly. **Rigid command patterns are unsupported** in conversational inputs. The AI's responses and clarifications must also adhere to natural, context-aware dialogue standards.
* **Adaptive Intelligence Engine**: Dynamic, goal-driven capabilities (`ml_layer/agent/`) enable the AI to learn (`learning.py`), plan (`planning_service.py`), coordinate tasks (`coordinator.py`), and generate explanations (`generate_explanation.py`). These processes operate behind the unified persona, leveraging the tag system for context and capability awareness.
* **Dynamic Capability Recognition**: The system inherently recognizes its own functionalities through the interplay of the tag-based system (esp. `feature_tracker.py`) and adaptive agent processes. Integration of new features/models (correctly tagged) results in their automatic incorporation into the AI's operational awareness, informed by the ML layer, without requiring core logic rewrites.
* **Centralized ML Layer**: The `ml_layer/` is the AI's cognitive core, housing intent processing, agent logic, learning mechanisms, and model integrations, underpinning all advanced conversational, reasoning, and adaptive functions.
* **Tag-Based State Management**: Four essential trackers (`trackers/`) maintain system state: Intent (`intent_tracker.py`), Features (`feature_tracker.py`), Context (`context_tracker.py`), and Analytics (`analytics_tracker.py`). Governed by a robust tag registry (`core/tag_registry.py`) and analysis logic (`tag_analyzer.py`), this ensures data consistency and fuels contextual understanding.
* **Resource Management**: Core utilities (`core/`) provide predictive analytics (`resource_predictor.py`), resource profiling (`resource_profile_class.py`), and proactive anomaly detection (`proactive_anomaly_detection.py`) for optimized, stable performance.
* **Web Interface**: A browser-based UI (`entry_points/web_app.py`, `static/`) provides a channel for configuration, visualization, diagnostics, and any necessary structured control actions unsuitable for the primary conversational interface.
* **Ollama Integration**: Built-in support for dynamic loading and execution of GGUF models via Ollama ensures flexible, efficient local inference.

## Project Structure

ggfai_framework/
├── entry_points/       # Handles all input/output (voice, text, web, sensors, etc.) - Interface Layer
├── ml_layer/           # Core AI/ML components: intent processing, agent logic, models - Intelligence Layer
│   ├── agent/          # Logic for planning, learning, coordination, explanation
│   └── models/         # Storage/management for pre-trained ML models
├── trackers/           # Manages system state via tag-based tracking - State/Memory Layer
├── core/               # Foundational utilities: tag registry, resource management, base classes - Core Services
├── config/             # Configuration files (prompts, settings, credentials)
├── static/             # Assets for the web interface (CSS, JS, images)
├── tests/              # Unit, integration, and end-to-end tests - Quality Assurance
├── docs/               # Detailed documentation (architecture, setup, API, guides)
├── .gitignore          # Specifies intentionally untracked files for Git
├── LICENSE             # Project's software license (MIT)
├── README.md           # This file: Primary guide for vision, principles, and structure
├── requirements.txt    # Python package dependencies


## Getting Started

### Prerequisites

* Python 3.8+
* `pip` and `venv` (standard Python tools)
* `git`
* Ollama (Optional, for GGUF support) - Must be installed and running separately.
* Relevant ML Models (Ensure required models for spaCy, Rasa, etc., are downloaded)

### Installation & Setup

1.  **Clone**: `git clone https://github.com/username/ggfai_framework.git && cd ggfai_framework`
2.  **Environment**: `python -m venv venv && source venv/bin/activate` (Use `venv\Scripts\activate` on Windows)
3.  **Dependencies**: `pip install -r requirements.txt`
4.  **Models**: Download necessary base models (e.g., `python -m spacy download en_core_web_sm`). Consult component documentation for specifics.
5.  **Configuration**: Review and adjust settings in the `config/` directory as needed.
6.  **Ollama**: Ensure the `ollama serve` process is running if GGUF models will be used.
7.  **Verification**: `pytest tests/`

### Running the Default Web Interface

1.  **Execute**: `python entry_points/web_app.py`
2.  **Access**: Navigate browser to `http://localhost:8000` (or as configured).

## Development Guidance

Adhere to the Core Principles when extending the framework:

* **Inputs/Outputs (`entry_points/`)**: Add new interaction channels. Ensure conversational channels strictly adhere to natural language principles.
* **Intelligence (`ml_layer/`)**: Enhance intent recognition (`intent_engine.py`), agent capabilities (`agent/`), or integrate new models (`models/`). Focus on improving the unified AI's effectiveness.
* **State (`trackers/`, `core/tag_registry.py`)**: Refine state tracking for deeper context or capability awareness.
* **Core Services (`core/`)**: Optimize resource management or add foundational utilities.
* **UI (`web_app.py`, `static/`)**: Develop the GUI for configuration, diagnostics, or structured control, *not* as a replacement for conversational interaction.
* **Testing (`tests/`)**: Implement comprehensive tests for all contributions.

## Documentation

Consult the `docs/` directory for in-depth information:

* `architecture.md`: Detailed system design and data flow.
* `setup_guide.md`: Comprehensive installation, configuration, deployment instructions.
* `api_reference.md`: Code-level documentation.
* `contributing.md`: Contribution process, standards, and guidelines.

## Contributing

Contributions aligned with the Core Principles are welcome. Follow the standard GitHub fork-pull-request model and adhere to guidelines in `docs/contributing.md`.


## Contact

* **GitHub Issues**: Technical issues, bugs, feature proposals.
* **Community**: Discussions, collaboration, support channels (link if available).

The GGF AI Framework is engineered to establish and propagate a new standard for powerful, accessible, and truly natural conversational AI assistants.


Strengths & Alignment:

Modular Architecture: The clear separation into layers (entry_points, ml_layer, trackers, core) directly supports the goal of flexibility and allows for focused development and extension, which is crucial for such an ambitious project.
Dedicated NLP/NLU: Having a specific intent_engine.py that leverages established libraries (spaCy, Rasa, Transformers, SpeechRecognition) and integrates with LLMs (via Ollama) provides the core capability needed for understanding natural language conversation.
Agent-Based Execution: The explicit inclusion of dynamic agents (ml_layer/agent/) with planning, learning, and coordination capabilities is essential for enabling the AI to act autonomously and proactively behind the scenes, fulfilling the "pervasive intelligence" goal.
State & Context Management: The tag-based tracking system (trackers/) is a powerful concept for maintaining context, feature awareness, and user intent history. This is fundamental for enabling fluid, contextual dialogue and dynamic capability recognition.
Accessibility Focus: Designing for a range of hardware (including lower-end) and providing a web interface aligns with the democratization goal.
Clear Philosophy: The "Core Principles" now explicitly guide development towards the unified persona and natural interaction model.
Potential Challenges & Dependencies:

Integration Complexity: While the components are defined, achieving the seamless, unified persona requires extremely sophisticated integration and coordination between the intent engine, the various agents, the state trackers, and the response generation. The coordinator.py logic will be critical and complex.
"Natural Feel" Quality: The quality of the natural interaction (making it feel like a human assistant) is heavily dependent on the performance, tuning, and latency of the underlying ML models (both NLU and potentially NLG for responses) and the sophistication of the dialogue management built on top of the framework. The design enables it, but achieving it requires significant ongoing effort and refinement.
Workflow Automation Robustness: Replacing low-code suites implies handling complex, multi-step tasks reliably via conversation. This requires very robust planning, error handling, state management, and potentially user confirmation steps within the agent logic – a significant implementation challenge.
Resource Management: Balancing powerful capabilities with the goal of running on common/accessible hardware requires careful optimization, efficient models (like GGUF via Ollama helps here), and effective resource management (core/resource_...).
