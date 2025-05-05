// GGFAI Dashboard Frontend - Enhanced Edition
// Implements tag analysis visualization, chat interface, and LLM thought process visualization

// Global state
const state = {
    selectedModel: null,
    tagData: null,
    explanationData: null,
    socket: null,
    analysisMethod: 'hybrid',
    explanationLevel: 'standard',
    apiKey: document.querySelector('meta[name="api-key"]')?.content || '',
    debugMode: document.querySelector('meta[name="debug-mode"]')?.content === 'true',
    chatHistory: [],
    thoughtProcess: [],
    visualizationType: '3d-cloud',
    animationSpeed: 5,
    nodeSize: 5,
    showLabels: true,
    enablePhysics: true,
    threeJsScene: null,
    threeJsRenderer: null,
    threeJsCamera: null,
    threeJsControls: null,
    threeJsAnimationFrame: null
};

// WebSocket connection
let ws;
let currentModel = 'default';
let thinking = false;
let yoloEnabled = false;
let recognizedContext = new Set();

// Configuration management
let config = {};

async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        config = await response.json();
        updateConfigUI();
    } catch (error) {
        console.error('Error loading config:', error);
        showNotification('Error', 'Failed to load configuration', 'danger');
    }
}

function updateConfigUI() {
    // Update UI elements with current config values
    document.getElementById('explanation-level').value = config.EXPLANATION_LEVEL || 'standard';
    document.getElementById('default-llm').value = config.DEFAULT_MODEL || 'llama2';
    document.getElementById('default-yolo').value = config.DEFAULT_YOLO_MODEL || 'default';
    document.getElementById('enable-thought-process').checked = config.ENABLE_THOUGHT_PROCESS !== false;
    document.getElementById('enable-https').checked = config.ENABLE_HTTPS === true;
    document.getElementById('enable-debug').checked = config.DEBUG === true;
    document.getElementById('ws-max-size').value = (config.WS_MAX_SIZE || 1048576) / 1048576; // Convert bytes to MB
    document.getElementById('rate-limit').value = parseInt(config.RATE_LIMIT) || 100;
    
    // Voice settings
    if (document.getElementById('voice-engine')) {
        document.getElementById('voice-engine').value = config.DEFAULT_TTS_ENGINE || 'system';
    }
    
    // Update any active visualizations
    if (window.updateVisualization) {
        window.updateVisualization();
    }
}

async function saveConfig() {
    const updates = {
        EXPLANATION_LEVEL: document.getElementById('explanation-level').value,
        DEFAULT_MODEL: document.getElementById('default-llm').value,
        DEFAULT_YOLO_MODEL: document.getElementById('default-yolo').value,
        ENABLE_THOUGHT_PROCESS: document.getElementById('enable-thought-process').checked,
        ENABLE_HTTPS: document.getElementById('enable-https').checked,
        DEBUG: document.getElementById('enable-debug').checked,
        WS_MAX_SIZE: document.getElementById('ws-max-size').value * 1048576, // Convert MB to bytes
        RATE_LIMIT: document.getElementById('rate-limit').value + '/minute',
        DEFAULT_TTS_ENGINE: document.getElementById('voice-engine')?.value || config.DEFAULT_TTS_ENGINE
    };

    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-KEY': document.querySelector('meta[name="api-key"]').content
            },
            body: JSON.stringify(updates)
        });

        if (response.ok) {
            showNotification('Success', 'Configuration updated successfully', 'success');
            config = {...config, ...updates};
        } else {
            throw new Error('Failed to update configuration');
        }
    } catch (error) {
        console.error('Error saving config:', error);
        showNotification('Error', 'Failed to save configuration', 'danger');
    }
}

// Initialize WebSocket connection
function initWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        notify('Connected to AI system', 'success');
        updateSystemStatus('connected');
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateSystemStatus('disconnected');
        setTimeout(initWebSocket, 5000); // Attempt reconnection
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
}

// Handle incoming messages
function handleMessage(data) {
    switch(data.type) {
        case 'thinking_start':
            showThinking();
            break;
            
        case 'thinking_update':
            updateThoughtProcess(data.thoughts);
            break;
            
        case 'response':
            hideThinking();
            addMessage('ai', data.content);
            
            // Update recognized context if YOLO detected something
            if (data.visual_context) {
                data.visual_context.forEach(item => recognizedContext.add(item));
                updateContextDisplay();
            }
            break;
            
        case 'error':
            hideThinking();
            notify(data.message, 'danger');
            break;
            
        case 'system_update':
            updateSystemMetrics(data.metrics);
            break;
            
        case 'model_loaded':
            currentModel = data.model;
            updateModelDisplay();
            notify(`Model ${data.model} loaded`, 'success');
            break;
    }
}

// UI Helper Functions
function addMessage(type, content) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;
    
    // Format code blocks if present
    content = formatCodeBlocks(content);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showThinking() {
    thinking = true;
    const messagesContainer = document.getElementById('chat-messages');
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message message-thinking';
    thinkingDiv.id = 'thinking-indicator';
    
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = `
        <span></span>
        <span></span>
        <span></span>
    `;
    
    thinkingDiv.appendChild(typingIndicator);
    messagesContainer.appendChild(thinkingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function hideThinking() {
    thinking = false;
    const thinkingIndicator = document.getElementById('thinking-indicator');
    if (thinkingIndicator) {
        thinkingIndicator.remove();
    }
}

// YOLO Integration
function toggleYOLO() {
    const yoloToggle = document.getElementById('yolo-toggle');
    const yoloSelector = document.getElementById('yolo-model-selector');
    
    yoloEnabled = yoloToggle.checked;
    yoloSelector.style.display = yoloEnabled ? 'inline-block' : 'none';
    
    ws.send(JSON.stringify({
        type: 'config_update',
        yolo_enabled: yoloEnabled,
        yolo_model: yoloSelector.value
    }));
    
    notify(`YOLO vision ${yoloEnabled ? 'enabled' : 'disabled'}`, 'info');
}

function updateContextDisplay() {
    const contextDisplay = document.createElement('div');
    contextDisplay.className = 'vision-status' + (recognizedContext.size ? ' active' : '');
    contextDisplay.innerHTML = `
        <i class="fas fa-camera"></i>
        ${Array.from(recognizedContext).slice(-3).join(', ')}
        ${recognizedContext.size > 3 ? '...' : ''}
    `;
    
    // Replace or add the vision status display
    const existing = document.querySelector('.vision-status');
    if (existing) {
        existing.replaceWith(contextDisplay);
    } else {
        document.querySelector('.chat-container').appendChild(contextDisplay);
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadConfig();
    
    // Chat input handling
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    sendBtn.addEventListener('click', sendMessage);
    
    // Auto-resize input
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    });
    
    // YOLO toggle
    document.getElementById('yolo-toggle').addEventListener('change', toggleYOLO);
    
    // Clear chat
    document.getElementById('clear-chat').addEventListener('click', () => {
        document.getElementById('chat-messages').innerHTML = '';
        recognizedContext.clear();
        addMessage('ai', 'Hi there! How can I assist you today?');
    });
    
    // Voice input
    const micBtn = document.getElementById('mic-btn');
    micBtn.addEventListener('click', toggleVoiceInput);

    // Add voice settings button next to mic button
    const settingsIcon = document.createElement('div');
    settingsIcon.className = 'voice-settings-icon';
    settingsIcon.innerHTML = '<i class="fa fa-cog"></i>';
    settingsIcon.title = 'Voice Settings';
    settingsIcon.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering mic button
        $('#voiceSettingsModal').modal('show');
    });
    micBtn.appendChild(settingsIcon);

    // Initialize voice dashboard when modal opens
    $('#voiceSettingsModal').on('show.bs.modal', function () {
        if (!window.voiceDashboard) {
            window.voiceDashboard = new VoiceDashboard('voice-dashboard-container', {
                energy_threshold: 300,
                pause_threshold: 0.8,
                dynamic_energy_threshold: true,
                recognition_engines: ['GOOGLE', 'SPHINX'],
                vad_system: 'ENERGY',
                vad_sensitivity: 2,
                vad_threshold: 0.5,
                tts_engine: 'SYSTEM',
                tts_voice: 'default',
                tts_rate: 175,
                tts_volume: 1.0
            }, (config) => {
                // Send config to server
                fetch('/api/voice/config', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': document.querySelector('meta[name="api-key"]').content
                    },
                    body: JSON.stringify(config)
                })
                .then(response => response.json())
                .then(data => {
                    $.notify({
                        icon: 'fas fa-check',
                        title: '<strong>Success</strong>',
                        message: 'Voice settings updated successfully'
                    }, {
                        type: 'success'
                    });
                })
                .catch(error => {
                    console.error('Error saving voice config:', error);
                    $.notify({
                        icon: 'fas fa-exclamation-triangle',
                        title: '<strong>Error</strong>',
                        message: 'Failed to update voice settings'
                    }, {
                        type: 'danger'
                    });
                });
            });
        }
    });

    // Save settings button
    document.getElementById('save-settings')?.addEventListener('click', saveConfig);
});

// Send message function
function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (message && !thinking) {
        addMessage('user', message);
        
        ws.send(JSON.stringify({
            type: 'message',
            content: message,
            yolo_enabled: yoloEnabled,
            yolo_model: document.getElementById('yolo-model-selector').value
        }));
        
        input.value = '';
        input.style.height = 'auto';
    }
}

// Voice input handling
let isRecording = false;
let recognition;

function toggleVoiceInput() {
    const micBtn = document.getElementById('mic-btn');
    
    if (!isRecording) {
        startRecording(micBtn);
    } else {
        stopRecording(micBtn);
    }
}

function startRecording(micBtn) {
    if (!recognition) {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.continuous = true;
        recognition.interimResults = true;
        
        recognition.onresult = (event) => {
            const input = document.getElementById('chat-input');
            input.value = Array.from(event.results)
                .map(result => result[0].transcript)
                .join('');
        };
        
        recognition.onend = () => {
            if (isRecording) {
                recognition.start();
            }
        };
    }
    
    recognition.start();
    isRecording = true;
    micBtn.classList.add('btn-danger');
    micBtn.querySelector('i').className = 'fa fa-stop';
}

function stopRecording(micBtn) {
    recognition.stop();
    isRecording = false;
    micBtn.classList.remove('btn-danger');
    micBtn.querySelector('i').className = 'fa fa-microphone';
}

// Utility functions
function notify(message, type = 'info') {
    $.notify({
        message: message
    }, {
        type: type,
        placement: {
            from: 'top',
            align: 'right'
        },
        timer: 3000
    });
}

function formatCodeBlocks(content) {
    // Replace markdown code blocks with highlighted HTML
    return content.replace(/```(\w+)?\n([\s\S]+?)```/g, (match, language, code) => {
        const highlighted = hljs.highlight(code, {
            language: language || 'plaintext'
        }).value;
        return `<pre><code class="hljs ${language || ''}">${highlighted}</code></pre>`;
    });
}

function updateSystemStatus(status) {
    const statusIndicator = document.getElementById('system-status');
    const icon = statusIndicator.querySelector('i');
    
    switch(status) {
        case 'connected':
            icon.className = 'fa fa-circle text-success';
            break;
        case 'disconnected':
            icon.className = 'fa fa-circle text-danger';
            break;
        case 'busy':
            icon.className = 'fa fa-circle text-warning';
            break;
    }
}

function updateModelDisplay() {
    document.getElementById('current-model-display').textContent = currentModel;
}

// Visualization functions
function initializeVisualizations() {
    init3DTagCloud();
    initTagNetwork();
    initThoughtVisualization();
}

// Export functions
window.onModelSelect = function(model) {
    ws.send(JSON.stringify({
        type: 'load_model',
        model: model
    }));
};

window.toggleFullscreen = function() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
};