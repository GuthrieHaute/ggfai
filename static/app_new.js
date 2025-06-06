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
    threeJsAnimationFrame: null,
    yolo: {
        enabled: false,
        model: 'default',
        confidence: 0.5,
        availableModels: []
    }
};

// Initialize WebSocket connection
function initWebSocket() {
    try {
        // Include API key in the WebSocket URL
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        state.socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws?api_key=${encodeURIComponent(state.apiKey)}`);
        
        state.socket.onopen = function() {
            console.log('WebSocket connection established');
            // Subscribe to tag analysis updates
            state.socket.send(JSON.stringify({
                type: 'subscribe',
                channel: 'tag_analysis',
                api_key: state.apiKey
            }));
            
            // Show success notification
            showNotification('WebSocket Connected', 'Real-time connection established successfully.', 'success');
        };
        
        state.socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'tag_analysis_response') {
                updateTagVisualization(data.visualization_data);
                state.tagData = data.ranked_tags;
            } else if (data.type === 'explanation_response') {
                updateExplanationDisplay(data.explanation);
                state.explanationData = data.explanation;
            } else if (data.type === 'chat_response') {
                addAIMessage(data.text, data.timestamp);
                
                // If thought process is included, update it
                if (data.thought_process) {
                    updateThoughtProcess(data.thought_process);
                }
            } else if (data.type === 'thinking') {
                showThinkingIndicator(data.text);
            } else if (data.type === 'model_confirmation') {
                updateModelDisplay(data.model);
                showNotification('Model Changed', `Model switched to ${data.model}`, 'success');
            } else if (data.type === 'error') {
                showNotification('Error', data.message, 'danger');
            }
        };
        
        state.socket.onerror = function(error) {
            console.error('WebSocket error:', error);
            showNotification('Connection Error', 'Failed to establish WebSocket connection.', 'danger');
        };
        
        state.socket.onclose = function() {
            console.log('WebSocket connection closed');
            // Attempt to reconnect after 5 seconds
            setTimeout(initWebSocket, 5000);
        };
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        showNotification('Connection Error', 'Failed to initialize WebSocket connection.', 'danger');
    }
}

// Initialize resource chart
function initResourceChart() {
    console.log("Initializing resource chart");
    const chartElement = document.getElementById('resource-chart');
    if (chartElement) {
        // Create resource usage chart using Chart.js
        const ctx = chartElement.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 10}, (_, i) => i),
                datasets: [{
                    label: 'CPU Usage (%)',
                    data: Array.from({length: 10}, () => Math.floor(Math.random() * 100)),
                    borderColor: '#4a8fe7',
                    backgroundColor: 'rgba(74, 143, 231, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Memory Usage (%)',
                    data: Array.from({length: 10}, () => Math.floor(Math.random() * 100)),
                    borderColor: '#8ac926',
                    backgroundColor: 'rgba(138, 201, 38, 0.1)',
                    tension: 0.4
                }, {
                    label: 'GPU Usage (%)',
                    data: Array.from({length: 10}, () => Math.floor(Math.random() * 100)),
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                animation: {
                    duration: 1000
                },
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
        
        // Update chart with random data every 3 seconds
        setInterval(() => {
            chart.data.labels.push(chart.data.labels.length);
            chart.data.labels.shift();
            
            chart.data.datasets.forEach(dataset => {
                dataset.data.push(Math.floor(Math.random() * 100));
                dataset.data.shift();
            });
            
            chart.update();
        }, 3000);
    }
}

// Initialize tag visualization
function initTagVisualization() {
    console.log("Initializing tag visualization");
    
    // Create container for visualization
    const vizContainer = document.getElementById('tag-viz-container');
    if (!vizContainer) return;
    
    // Fetch initial tag data
    fetchTagAnalysis();
    
    // Set up analysis method selector
    const methodSelector = document.getElementById('analysis-method');
    if (methodSelector) {
        methodSelector.addEventListener('change', function() {
            state.analysisMethod = this.value;
        });
    }
    
    // Set up refresh button
    const refreshBtn = document.getElementById('refresh-viz');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            fetchTagAnalysis();
            showNotification('Refreshing', 'Updating tag visualization...', 'info');
        });
    }
    
    // Set up apply settings button
    const applyBtn = document.getElementById('apply-settings');
    if (applyBtn) {
        applyBtn.addEventListener('click', function() {
            fetchTagAnalysis();
            showNotification('Settings Applied', 'Analysis settings updated.', 'success');
        });
    }
    
    // Initialize 3D tag cloud
    init3DTagCloud();
    
    // Set up visualization settings
    const vizTypeSelector = document.getElementById('visualization-type');
    if (vizTypeSelector) {
        vizTypeSelector.addEventListener('change', function() {
            state.visualizationType = this.value;
        });
    }
    
    const animationSpeedSlider = document.getElementById('animation-speed');
    if (animationSpeedSlider) {
        animationSpeedSlider.addEventListener('input', function() {
            state.animationSpeed = parseInt(this.value);
        });
    }
    
    const nodeSizeSlider = document.getElementById('node-size');
    if (nodeSizeSlider) {
        nodeSizeSlider.addEventListener('input', function() {
            state.nodeSize = parseInt(this.value);
        });
    }
    
    const showLabelsCheckbox = document.getElementById('show-labels');
    if (showLabelsCheckbox) {
        showLabelsCheckbox.addEventListener('change', function() {
            state.showLabels = this.checked;
        });
    }
    
    const enablePhysicsCheckbox = document.getElementById('enable-physics');
    if (enablePhysicsCheckbox) {
        enablePhysicsCheckbox.addEventListener('change', function() {
            state.enablePhysics = this.checked;
        });
    }
    
    const applyVizSettingsBtn = document.getElementById('apply-viz-settings');
    if (applyVizSettingsBtn) {
        applyVizSettingsBtn.addEventListener('click', function() {
            if (state.visualizationType === '3d-cloud') {
                update3DTagCloud();
            } else {
                updateTagVisualization(state.tagData);
            }
            showNotification('Settings Applied', 'Visualization settings updated.', 'success');
        });
    }
    
    // Set up 3D tag cloud refresh button
    const refresh3DVizBtn = document.getElementById('refresh-3d-viz');
    if (refresh3DVizBtn) {
        refresh3DVizBtn.addEventListener('click', function() {
            update3DTagCloud();
            showNotification('Refreshing', 'Updating 3D tag cloud...', 'info');
        });
    }
}

// Initialize 3D tag cloud
function init3DTagCloud() {
    const container = document.getElementById('tag-cloud-container');
    if (!container) return;
    
    // Clear previous visualization
    if (state.threeJsAnimationFrame) {
        cancelAnimationFrame(state.threeJsAnimationFrame);
    }
    
    if (state.threeJsRenderer) {
        container.removeChild(state.threeJsRenderer.domElement);
    }
    
    // Create scene
    state.threeJsScene = new THREE.Scene();
    state.threeJsScene.background = new THREE.Color(0xf8f9fa);
    
    // Create camera
    state.threeJsCamera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    state.threeJsCamera.position.z = 30;
    
    // Create renderer
    state.threeJsRenderer = new THREE.WebGLRenderer({ antialias: true });
    state.threeJsRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(state.threeJsRenderer.domElement);
    
    // Add orbit controls
    state.threeJsControls = new THREE.OrbitControls(state.threeJsCamera, state.threeJsRenderer.domElement);
    state.threeJsControls.enableDamping = true;
    state.threeJsControls.dampingFactor = 0.05;
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    state.threeJsScene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    state.threeJsScene.add(directionalLight);
    
    // Add tags (placeholder)
    const dummyTags = [
        { tag_id: 'tag1', score: 0.9 },
        { tag_id: 'tag2', score: 0.8 },
        { tag_id: 'tag3', score: 0.7 },
        { tag_id: 'tag4', score: 0.6 },
        { tag_id: 'tag5', score: 0.5 }
    ];
    
    addTagsTo3DCloud(dummyTags);
    
    // Animation loop
    function animate() {
        state.threeJsAnimationFrame = requestAnimationFrame(animate);
        state.threeJsControls.update();
        state.threeJsRenderer.render(state.threeJsScene, state.threeJsCamera);
    }
    
    animate();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (state.threeJsCamera && state.threeJsRenderer && container) {
            state.threeJsCamera.aspect = container.clientWidth / container.clientHeight;
            state.threeJsCamera.updateProjectionMatrix();
            state.threeJsRenderer.setSize(container.clientWidth, container.clientHeight);
        }
    });
}

// Add tags to 3D cloud
function addTagsTo3DCloud(tags) {
    if (!state.threeJsScene) return;
    
    // Clear previous tags
    state.threeJsScene.children = state.threeJsScene.children.filter(child => 
        child.type === 'AmbientLight' || child.type === 'DirectionalLight');
    
    // Add tags as spheres
    tags.forEach((tag, index) => {
        // Create sphere geometry
        const size = 1 + (tag.score * 2);
        const geometry = new THREE.SphereGeometry(size, 32, 32);
        
        // Create material with color based on score
        const color = getTagColor(tag.score);
        const material = new THREE.MeshPhongMaterial({ 
            color: color,
            shininess: 100
        });
        
        // Create mesh
        const sphere = new THREE.Mesh(geometry, material);
        
        // Position in 3D space (distribute in a sphere)
        const phi = Math.acos(-1 + (2 * index) / tags.length);
        const theta = Math.sqrt(tags.length * Math.PI) * phi;
        const radius = 15;
        
        sphere.position.x = radius * Math.cos(theta) * Math.sin(phi);
        sphere.position.y = radius * Math.sin(theta) * Math.sin(phi);
        sphere.position.z = radius * Math.cos(phi);
        
        // Add to scene
        state.threeJsScene.add(sphere);
        
        // Add label if enabled
        if (state.showLabels) {
            // Create canvas for text
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 256;
            canvas.height = 128;
            
            // Draw text
            context.fillStyle = '#ffffff';
            context.fillRect(0, 0, canvas.width, canvas.height);
            context.font = '24px Arial';
            context.fillStyle = '#000000';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(tag.tag_id, canvas.width / 2, canvas.height / 2);
            
            // Create texture
            const texture = new THREE.CanvasTexture(canvas);
            
            // Create sprite material
            const spriteMaterial = new THREE.SpriteMaterial({ 
                map: texture,
                transparent: true
            });
            
            // Create sprite
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.scale.set(5, 2.5, 1);
            sprite.position.copy(sphere.position);
            sprite.position.y += size + 1;
            
            // Add to scene
            state.threeJsScene.add(sprite);
        }
    });
}

// Update 3D tag cloud
function update3DTagCloud() {
    if (state.tagData) {
        addTagsTo3DCloud(state.tagData);
    } else {
        fetchTagAnalysis().then(() => {
            if (state.tagData) {
                addTagsTo3DCloud(state.tagData);
            }
        });
    }
}

// Initialize explanation display
function initExplanationDisplay() {
    console.log("Initializing explanation display");
    
    // Set up explanation level selector
    const levelSelector = document.getElementById('explanation-level');
    if (levelSelector) {
        levelSelector.addEventListener('change', function() {
            state.explanationLevel = this.value;
            if (state.explanationData) {
                fetchExplanation(state.explanationData.trace_id);
            }
        });
    }
}

// Initialize chat interface
function initChatInterface() {
    console.log("Initializing chat interface");
    
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    const clearChat = document.getElementById('clear-chat');
    
    if (chatInput && chatSend) {
        // Send message on button click
        chatSend.addEventListener('click', function() {
            sendChatMessage();
        });
        
        // Send message on Enter key (but allow Shift+Enter for new line)
        chatInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendChatMessage();
            }
            
            // Auto-resize textarea
            setTimeout(() => {
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
            }, 0);
        });
    }
    
    if (clearChat) {
        clearChat.addEventListener('click', function() {
            clearChatHistory();
        });
    }
}

// Send chat message
function sendChatMessage() {
    const chatInput = document.getElementById('chat-input');
    const text = chatInput.value.trim();
    
    if (!text) return;
    
    // Add user message to chat
    addUserMessage(text);
    
    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Show thinking indicator
    showThinkingIndicator();
    
    // Send to server
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
        state.socket.send(JSON.stringify({
            type: 'user_input',
            text: text,
            model: state.selectedModel,
            explanation_level: state.explanationLevel
        }));
    } else {
        // Fallback to REST API if WebSocket is not available
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': state.apiKey
            },
            body: JSON.stringify({
                text: text,
                model: state.selectedModel,
                explanation_level: state.explanationLevel
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Remove thinking indicator
            removeThinkingIndicator();
            
            // Add AI response to chat
            addAIMessage(data.text, data.timestamp);
            
            // Update thought process if available
            if (data.thought_process) {
                updateThoughtProcess(data.thought_process);
            }
        })
        .catch(error => {
            console.error('Error sending message:', error);
            removeThinkingIndicator();
            addAIMessage('Sorry, I encountered an error processing your request.', Date.now());
            showNotification('Error', 'Failed to send message', 'danger');
        });
    }
}

// Add user message to chat
function addUserMessage(text) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    const timestamp = Date.now();
    const formattedTime = formatTimestamp(timestamp);
    
    // Format message with markdown
    const formattedText = formatMessageText(text);
    
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = 'message message-user';
    messageElement.innerHTML = `
        <div class="message-content">${formattedText}</div>
        <div class="message-time">${formattedTime}</div>
    `;
    
    // Add to chat
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Add to history
    state.chatHistory.push({
        role: 'user',
        text: text,
        timestamp: timestamp
    });
}

// Add AI message to chat
function addAIMessage(text, timestamp = Date.now()) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    // Remove thinking indicator if present
    removeThinkingIndicator();
    
    const formattedTime = formatTimestamp(timestamp);
    
    // Format message with markdown
    const formattedText = formatMessageText(text);
    
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = 'message message-ai';
    messageElement.innerHTML = `
        <div class="message-content">${formattedText}</div>
        <div class="message-time">${formattedTime}</div>
    `;
    
    // Add to chat
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Add to history
    state.chatHistory.push({
        role: 'ai',
        text: text,
        timestamp: timestamp
    });
}

// Show thinking indicator
function showThinkingIndicator(text = 'Thinking...') {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    // Remove existing thinking indicator
    removeThinkingIndicator();
    
    // Create thinking indicator
    const thinkingElement = document.createElement('div');
    thinkingElement.className = 'message message-thinking';
    thinkingElement.id = 'thinking-indicator';
    thinkingElement.innerHTML = `
        <div class="message-content">
            ${text} 
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    // Add to chat
    chatMessages.appendChild(thinkingElement);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Remove thinking indicator
function removeThinkingIndicator() {
    const thinkingIndicator = document.getElementById('thinking-indicator');
    if (thinkingIndicator) {
        thinkingIndicator.remove();
    }
}

// Format message text with markdown
function formatMessageText(text) {
    // Use marked library to convert markdown to HTML
    const renderer = new marked.Renderer();
    
    // Configure code highlighting
    renderer.code = function(code, language) {
        const validLanguage = hljs.getLanguage(language) ? language : 'plaintext';
        const highlightedCode = hljs.highlight(validLanguage, code).value;
        return `<pre><code class="hljs ${validLanguage}">${highlightedCode}</code></pre>`;
    };
    
    // Set options
    marked.setOptions({
        renderer: renderer,
        highlight: function(code, lang) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(language, code).value;
        },
        breaks: true,
        gfm: true
    });
    
    // Convert markdown to HTML
    return marked.parse(text);
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Clear chat history
function clearChatHistory() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    // Clear UI
    chatMessages.innerHTML = '';
    
    // Add welcome message
    addAIMessage('Hello! I\'m your GGFAI assistant. How can I help you today?');
    
    // Clear history
    state.chatHistory = [];
    
    showNotification('Chat Cleared', 'Chat history has been cleared.', 'info');
}

// Initialize thought process visualization
function initThoughtProcess() {
    console.log("Initializing thought process visualization");
    
    // Set up auto-update checkbox
    const autoUpdateThoughts = document.getElementById('auto-update-thoughts');
    if (autoUpdateThoughts) {
        autoUpdateThoughts.addEventListener('change', function() {
            if (this.checked) {
                showNotification('Auto-update Enabled', 'Thought process will update automatically.', 'info');
            } else {
                showNotification('Auto-update Disabled', 'Thought process updates paused.', 'info');
            }
        });
    }
    
    // Set up export button
    const exportThoughts = document.getElementById('export-thoughts');
    if (exportThoughts) {
        exportThoughts.addEventListener('click', function() {
            exportThoughtProcess();
        });
    }
}

// Update thought process visualization
function updateThoughtProcess(thoughtProcess) {
    const autoUpdate = document.getElementById('auto-update-thoughts')?.checked;
    if (!autoUpdate) return;
    
    const container = document.getElementById('thinking-container');
    if (!container) return;
    
    // Store thought process
    state.thoughtProcess = thoughtProcess;
    
    // Clear previous visualization
    container.innerHTML = '';
    
    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.className = 'thinking-tooltip';
    container.appendChild(tooltip);
    
    // Calculate positions
    const width = container.clientWidth;
    const height = container.clientHeight;
    const nodeRadius = 30;
    
    // Create nodes and connections
    thoughtProcess.forEach((thought, index) => {
        // Create node
        const node = document.createElement('div');
        node.className = 'thinking-node';
        node.style.width = `${nodeRadius * 2}px`;
        node.style.height = `${nodeRadius * 2}px`;
        node.style.backgroundColor = getThoughtColor(thought.type);
        node.innerHTML = getThoughtIcon(thought.type);
        
        // Position node in a tree-like structure
        const depth = thought.depth || 0;
        const siblings = thoughtProcess.filter(t => (t.depth || 0) === depth).length;
        const siblingIndex = thoughtProcess.filter(t => (t.depth || 0) === depth && thoughtProcess.indexOf(t) <= index).length - 1;
        
        const x = (width * (depth + 1)) / (Math.max(...thoughtProcess.map(t => t.depth || 0)) + 2);
        const y = height * (siblingIndex + 1) / (siblings + 1);
        
        node.style.left = `${x - nodeRadius}px`;
        node.style.top = `${y - nodeRadius}px`;
        
        // Add hover effect
        node.addEventListener('mouseenter', () => {
            tooltip.innerHTML = `
                <strong>${thought.type}</strong>
                <p>${thought.content}</p>
                <small>${formatTimestamp(thought.timestamp)}</small>
            `;
            tooltip.style.left = `${x + nodeRadius * 2}px`;
            tooltip.style.top = `${y}px`;
            tooltip.style.opacity = '1';
        });
        
        node.addEventListener('mouseleave', () => {
            tooltip.style.opacity = '0';
        });
        
        // Add click effect
        node.addEventListener('click', () => {
            showThoughtDetails(thought);
        });
        
        container.appendChild(node);
        
        // Create connection to parent if exists
        if (thought.parent_index !== undefined) {
            const parent = thoughtProcess[thought.parent_index];
            if (parent) {
                const parentDepth = parent.depth || 0;
                const parentSiblings = thoughtProcess.filter(t => (t.depth || 0) === parentDepth).length;
                const parentSiblingIndex = thoughtProcess.filter(t => (t.depth || 0) === parentDepth && thoughtProcess.indexOf(t) <= thought.parent_index).length - 1;
                
                const parentX = (width * (parentDepth + 1)) / (Math.max(...thoughtProcess.map(t => t.depth || 0)) + 2);
                const parentY = height * (parentSiblingIndex + 1) / (parentSiblings + 1);
                
                const connection = document.createElement('div');
                connection.className = 'thinking-connection';
                
                // Calculate connection position and dimensions
                const dx = x - parentX;
                const dy = y - parentY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const angle = Math.atan2(dy, dx);
                
                connection.style.width = `${distance}px`;
                connection.style.height = '2px';
                connection.style.left = `${parentX}px`;
                connection.style.top = `${parentY}px`;
                connection.style.transform = `rotate(${angle}rad)`;
                
                container.appendChild(connection);
            }
        }
    });
}

// Show thought details
function showThoughtDetails(thought) {
    const detailsContainer = document.getElementById('thought-details');
    if (!detailsContainer) return;
    
    // Format content with markdown
    const formattedContent = formatMessageText(thought.content);
    
    // Create details HTML
    detailsContainer.innerHTML = `
        <div class="card mb-3">
            <div class="card-header bg-${getThoughtTypeClass(thought.type)}">
                <h5 class="mb-0 text-white">${thought.type}</h5>
            </div>
            <div class="card-body">
                <div class="thought-content">${formattedContent}</div>
                <div class="text-muted mt-3">
                    <small>Timestamp: ${new Date(thought.timestamp).toLocaleString()}</small>
                    <br>
                    <small>Depth: ${thought.depth || 0}</small>
                    ${thought.confidence ? `<br><small>Confidence: ${thought.confidence.toFixed(2)}</small>` : ''}
                </div>
            </div>
        </div>
    `;
}

// Get thought color based on type
function getThoughtColor(type) {
    const colors = {
        'reasoning': '#4a8fe7',
        'planning': '#8ac926',
        'retrieval': '#ff6b6b',
        'generation': '#9b5de5',
        'evaluation': '#f15bb5',
        'decision': '#00bbf9',
        'reflection': '#fee440'
    };
    
    return colors[type.toLowerCase()] || '#aaaaaa';
}

// Get thought icon based on type
function getThoughtIcon(type) {
    const icons = {
        'reasoning': '<i class="fas fa-brain"></i>',
        'planning': '<i class="fas fa-tasks"></i>',
        'retrieval': '<i class="fas fa-database"></i>',
        'generation': '<i class="fas fa-pen"></i>',
        'evaluation': '<i class="fas fa-balance-scale"></i>',
        'decision': '<i class="fas fa-check-circle"></i>',
        'reflection': '<i class="fas fa-lightbulb"></i>'
    };
    
    return icons[type.toLowerCase()] || '<i class="fas fa-question"></i>';
}

// Get thought type Bootstrap class
function getThoughtTypeClass(type) {
    const classes = {
        'reasoning': 'primary',
        'planning': 'success',
        'retrieval': 'danger',
        'generation': 'purple',
        'evaluation': 'pink',
        'decision': 'info',
        'reflection': 'warning'
    };
    
    return classes[type.toLowerCase()] || 'secondary';
}

// Export thought process
function exportThoughtProcess() {
    if (!state.thoughtProcess || state.thoughtProcess.length === 0) {
        showNotification('Export Failed', 'No thought process data available.', 'warning');
        return;
    }
    
    // Create JSON string
    const jsonString = JSON.stringify(state.thoughtProcess, null, 2);
    
    // Create blob
    const blob = new Blob([jsonString], { type: 'application/json' });
    
    // Create download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `thought-process-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 0);
    
    showNotification('Export Complete', 'Thought process exported successfully.', 'success');
}

// Initialize system monitoring
function initSystemMonitoring() {
    console.log("Initializing system monitoring");
    
    // Set up auto-update logs checkbox
    const autoUpdateLogs = document.getElementById('auto-update-logs');
    if (autoUpdateLogs) {
        autoUpdateLogs.addEventListener('change', function() {
            if (this.checked) {
                showNotification('Auto-update Enabled', 'System logs will update automatically.', 'info');
            } else {
                showNotification('Auto-update Disabled', 'System log updates paused.', 'info');
            }
        });
    }
    
    // Fetch LLM status
    fetchLLMStatus();
    
    // Set up periodic updates
    setInterval(fetchLLMStatus, 10000); // Every 10 seconds
}

// Fetch LLM status
function fetchLLMStatus() {
    fetch('/api/llm-status', {
        headers: {
            'X-API-Key': state.apiKey
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        updateLLMStatusTable(data);
    })
    .catch(error => {
        console.error('Error fetching LLM status:', error);
    });
}

// Update LLM status table
function updateLLMStatusTable(data) {
    const table = document.getElementById('llm-status-table');
    if (!table) return;
    
    // Clear table
    table.innerHTML = '';
    
    // Add rows
    if (data.length === 0) {
        table.innerHTML = '<tr><td colspan="4" class="text-center">No models currently loaded</td></tr>';
        return;
    }
    
    data.forEach(model => {
        const row = document.createElement('tr');
        
        // Status class
        const statusClass = model.status === 'active' ? 'success' : 
                           model.status === 'loading' ? 'warning' : 
                           model.status === 'error' ? 'danger' : 'secondary';
        
        row.innerHTML = `
            <td>${model.name}</td>
            <td><span class="badge bg-${statusClass}">${model.status}</span></td>
            <td>${formatMemorySize(model.memory_usage)}</td>
            <td>${model.request_count}</td>
        `;
        
        table.appendChild(row);
    });
}

// Format memory size
function formatMemorySize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Fetch tag analysis data from API
function fetchTagAnalysis() {
    return fetch(`/api/tag-analysis?method=${state.analysisMethod}`, {
        headers: {
            'X-API-Key': state.apiKey
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        state.tagData = data.ranked_tags;
        updateTagVisualization(data.visualization_data);
        return data;
    })
    .catch(error => {
        console.error('Error fetching tag analysis:', error);
        showNotification('Data Error', 'Failed to fetch tag analysis data.', 'danger');
    });
}

// Fetch explanation data from API
function fetchExplanation(traceId) {
    return fetch(`/api/explanation/${traceId}?level=${state.explanationLevel}`, {
        headers: {
            'X-API-Key': state.apiKey
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        state.explanationData = data;
        updateExplanationDisplay(data);
        return data;
    })
    .catch(error => {
        console.error('Error fetching explanation:', error);
        showNotification('Data Error', 'Failed to fetch explanation data.', 'danger');
    });
}

// Update tag visualization with new data
function updateTagVisualization(data) {
    const container = document.getElementById('tag-viz-container');
    if (!container) return;
    
    // Clear previous visualization
    container.innerHTML = '';
    
    // Create network visualization using vis.js
    const nodes = new vis.DataSet(data.nodes);
    const edges = new vis.DataSet(data.edges);
    
    // Create network
    const network = new vis.Network(container, {
        nodes: nodes,
        edges: edges
    }, {
        physics: {
            enabled: state.enablePhysics,
            stabilization: {
                iterations: 100
            },
            barnesHut: {
                gravitationalConstant: -2000,
                centralGravity: 0.1,
                springLength: 150,
                springConstant: 0.05
            }
        },
        nodes: {
            shape: 'dot',
            size: 16 * state.nodeSize / 5,
            font: {
                size: 12 * state.nodeSize / 5,
                face: 'Public Sans'
            },
            borderWidth: 2,
            shadow: true
        },
        edges: {
            width: 2,
            shadow: true,
            smooth: {
                type: 'continuous'
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true,
            dragView: true
        }
    });
    
    // Handle node click
    network.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            if (nodeId !== 'context') {
                // Extract tag ID from node label
                const node = nodes.get(nodeId);
                const tagId = node.label.split('\n')[0];
                
                // Fetch explanation for this tag
                fetchExplanation(tagId);
                showNotification('Loading Explanation', `Fetching explanation for ${tagId}...`, 'info');
            }
        }
    });
}

// Update explanation display with new data
function updateExplanationDisplay(data) {
    const container = document.getElementById('explanation-container');
    if (!container) return;
    
    // Create explanation card content
    const cardBody = container.querySelector('.card-body');
    if (cardBody) {
        cardBody.innerHTML = `
            <div class="explanation-narrative">${formatNarrative(data.narrative)}</div>
            ${data.visualization_data ? '<div id="explanation-viz" class="explanation-viz mt-4" style="height: 300px;"></div>' : ''}
        `;
    }
    
    // Create visualization if data is available
    if (data.visualization_data) {
        const vizContainer = document.getElementById('explanation-viz');
        if (vizContainer) {
            const network = new vis.Network(
                vizContainer,
                {
                    nodes: new vis.DataSet(data.visualization_data.nodes),
                    edges: new vis.DataSet(data.visualization_data.edges)
                },
                data.visualization_data.options || {
                    physics: {
                        enabled: state.enablePhysics,
                        stabilization: true
                    },
                    nodes: {
                        shape: 'dot',
                        size: 16,
                        font: {
                            size: 12,
                            face: 'Public Sans'
                        },
                        borderWidth: 2,
                        shadow: true
                    },
                    edges: {
                        width: 2,
                        shadow: true
                    }
                }
            );
        }
    }
}

// Format explanation narrative with expandable sections
function formatNarrative(narrative) {
    if (!narrative) return '<p>No explanation available</p>';
    
    // Format with markdown
    const formattedNarrative = formatMessageText(narrative);
    
    // Split into sections
    const sections = formattedNarrative.split('<hr>');
    
    // Format main section
    let html = `<div class="main-narrative">${sections[0]}</div>`;
    
    // Format additional sections as expandable
    if (sections.length > 1) {
        html += '<div class="accordion mt-3" id="explanationAccordion">';
        
        for (let i = 1; i < sections.length; i++) {
            const sectionId = `section-${i}`;
            html += `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading${i}">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${i}" aria-expanded="false" aria-controls="collapse${i}">
                            Additional Details ${i}
                        </button>
                    </h2>
                    <div id="collapse${i}" class="accordion-collapse collapse" aria-labelledby="heading${i}" data-bs-parent="#explanationAccordion">
                        <div class="accordion-body">
                            ${sections[i]}
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
    }
    
    return html;
}

// Get color based on score
function getTagColor(score, alpha = 1) {
    // Color gradient from red (low) to green (high)
    const r = Math.floor(255 * (1 - score));
    const g = Math.floor(255 * score);
    const b = 100;
    
    return alpha < 1 ? 
        `rgba(${r}, ${g}, ${b}, ${alpha})` : 
        `rgb(${r}, ${g}, ${b})`;
}

// Handle model selection
function onModelSelect(modelId) {
    console.log(`Model selected: ${modelId}`);
    state.selectedModel = modelId;
    
    // Update UI
    updateModelDisplay(modelId);
    
    // Send model change via WebSocket if available
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
        state.socket.send(JSON.stringify({
            type: 'model_change',
            model: modelId
        }));
    } else {
        // Fallback to REST API
        fetch('/api/select-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': state.apiKey
            },
            body: JSON.stringify({ model_id: modelId })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Model selection response:', data);
            // Refresh tag analysis with new model
            fetchTagAnalysis();
            showNotification('Model Changed', `Model switched to ${modelId}`, 'success');
        })
        .catch(error => {
            console.error('Error selecting model:', error);
            showNotification('Error', `Failed to switch model to ${modelId}`, 'danger');
        });
    }
}

// Update model display
function updateModelDisplay(modelId) {
    const modelDisplay = document.getElementById('current-model-display');
    if (modelDisplay) {
        modelDisplay.textContent = modelId;
    }
}

// Set analysis method
function setAnalysisMethod(method) {
    state.analysisMethod = method;
    
    // Update dropdown in modal
    const methodSelector = document.getElementById('analysis-method');
    if (methodSelector) {
        methodSelector.value = method;
    }
    
    showNotification('Method Changed', `Analysis method set to ${method}`, 'info');
}

// Set explanation level
function setExplanationLevel(level) {
    state.explanationLevel = level;
    
    // Update dropdown in modal
    const levelSelector = document.getElementById('explanation-level');
    if (levelSelector) {
        levelSelector.value = level;
    }
    
    // Update explanation if one is displayed
    if (state.explanationData) {
        fetchExplanation(state.explanationData.trace_id);
    }
    
    showNotification('Level Changed', `Explanation level set to ${level}`, 'info');
}

// Toggle thought process tab
function toggleThoughtProcess() {
    const tab = document.getElementById('thought-process-tab');
    if (tab) {
        tab.click();
    }
}

// Toggle fullscreen
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            showNotification('Error', `Could not enter fullscreen mode: ${err.message}`, 'warning');
        });
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

// Initialize voice input
function initVoiceInput() {
    const micButton = document.getElementById('mic-btn');
    if (!micButton) return;
    
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    
    micButton.addEventListener('click', function() {
        if (isRecording) {
            // Stop recording
            mediaRecorder.stop();
            micButton.classList.remove('btn-danger');
            micButton.classList.add('btn-primary');
            isRecording = false;
        } else {
            // Start recording
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.addEventListener('dataavailable', event => {
                        audioChunks.push(event.data);
                    });
                    
                    mediaRecorder.addEventListener('stop', () => {
                        const audioBlob = new Blob(audioChunks);
                        sendAudioToServer(audioBlob);
                    });
                    
                    mediaRecorder.start();
                    micButton.classList.remove('btn-primary');
                    micButton.classList.add('btn-danger');
                    isRecording = true;
                    
                    showNotification('Recording', 'Voice recording started...', 'info');
                })
                .catch(error => {
                    console.error('Error accessing microphone:', error);
                    showNotification('Microphone Error', 'Could not access microphone', 'danger');
                });
        }
    });
}

// Send audio to server for processing
function sendAudioToServer(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    
    showNotification('Processing', 'Processing voice input...', 'info');
    
    fetch('/api/voice-input', {
        method: 'POST',
        headers: {
            'X-API-Key': state.apiKey
        },
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Voice input response:', data);
        
        // Add transcription to chat as user message
        if (data.text) {
            addUserMessage(data.text);
            
            // Show thinking indicator
            showThinkingIndicator();
            
            // If response is included, add it
            if (data.response) {
                setTimeout(() => {
                    removeThinkingIndicator();
                    addAIMessage(data.response);
                    
                    // Update thought process if available
                    if (data.thought_process) {
                        updateThoughtProcess(data.thought_process);
                    }
                }, 1000);
            }
        }
        
        showNotification('Voice Processed', 'Voice input processed successfully', 'success');
    })
    .catch(error => {
        console.error('Error sending voice input:', error);
        showNotification('Error', 'Failed to process voice input', 'danger');
    });
}

// Show notification
function showNotification(title, message, type) {
    $.notify({
        icon: type === 'success' ? 'fas fa-check' : 
              type === 'danger' ? 'fas fa-exclamation-triangle' :
              type === 'warning' ? 'fas fa-exclamation-circle' : 'fas fa-info-circle',
        title: `<strong>${title}</strong>`,
        message: message
    }, {
        type: type,
        placement: {
            from: "top",
            align: "right"
        },
        time: 1000,
        delay: 3000
    });
}

// Check system health
function checkSystemHealth() {
    fetch('/health', {
        headers: {
            'X-API-Key': state.apiKey
        }
    })
    .then(response => {
        const statusIcon = document.querySelector('#system-status i');
        if (statusIcon) {
            if (response.ok) {
                statusIcon.className = 'fa fa-circle text-success';
                document.querySelector('#system-status').title = 'System Healthy';
            } else {
                statusIcon.className = 'fa fa-circle text-danger';
                document.querySelector('#system-status').title = 'System Error';
            }
        }
        return response.json();
    })
    .then(data => {
        console.log('Health check:', data);
        
        // Add log entry if auto-update is enabled
        const autoUpdateLogs = document.getElementById('auto-update-logs')?.checked;
        if (autoUpdateLogs) {
            addLogEntry(`[INFO] Health check: ${data.status}`);
        }
    })
    .catch(error => {
        console.error('Health check error:', error);
        const statusIcon = document.querySelector('#system-status i');
        if (statusIcon) {
            statusIcon.className = 'fa fa-circle text-warning';
            document.querySelector('#system-status').title = 'Health Check Failed';
        }
        
        // Add log entry if auto-update is enabled
        const autoUpdateLogs = document.getElementById('auto-update-logs')?.checked;
        if (autoUpdateLogs) {
            addLogEntry(`[ERROR] Health check failed: ${error.message}`);
        }
    });
}

// Add log entry
function addLogEntry(text) {
    const logs = document.getElementById('system-logs');
    if (!logs) return;
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = `${new Date().toLocaleTimeString()} ${text}`;
    
    logs.appendChild(entry);
    
    // Scroll to bottom
    logs.parentElement.scrollTop = logs.parentElement.scrollHeight;
    
    // Limit number of entries
    const maxEntries = 100;
    const entries = logs.querySelectorAll('.log-entry');
    if (entries.length > maxEntries) {
        for (let i = 0; i < entries.length - maxEntries; i++) {
            entries[i].remove();
        }
    }
}

// Initialize YOLO toggle
function initYoloToggle() {
    console.log("Initializing YOLO toggle");
    
    const yoloToggle = document.getElementById('yolo-toggle');
    const yoloModelSelector = document.getElementById('yolo-model-selector');
    
    if (!yoloToggle || !yoloModelSelector) return;
    
    // Fetch available YOLO models
    fetchYoloModels();
    
    // Fetch current YOLO status
    fetchYoloStatus();
    
    // Add event listener for toggle
    yoloToggle.addEventListener('change', function() {
        const enabled = this.checked;
        toggleYolo(enabled);
        
        // Show/hide model selector
        yoloModelSelector.style.display = enabled ? 'inline-block' : 'none';
    });
    
    // Add event listener for model selector
    yoloModelSelector.addEventListener('change', function() {
        const model = this.value;
        if (state.yolo.enabled) {
            updateYoloModel(model);
        }
    });
}

// Fetch available YOLO models
function fetchYoloModels() {
    fetch('/api/yolo/models', {
        headers: {
            'X-API-KEY': state.apiKey
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.models && Array.isArray(data.models)) {
            state.yolo.availableModels = data.models;
            
            // Update model selector
            const yoloModelSelector = document.getElementById('yolo-model-selector');
            if (yoloModelSelector) {
                // Clear existing options
                yoloModelSelector.innerHTML = '';
                
                // Add options for each model
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    yoloModelSelector.appendChild(option);
                });
                
                // Set current model
                yoloModelSelector.value = state.yolo.model;
            }
        }
    })
    .catch(error => {
        console.error('Error fetching YOLO models:', error);
        showNotification('Error', 'Failed to fetch YOLO models', 'danger');
    });
}

// Fetch current YOLO status
function fetchYoloStatus() {
    fetch('/api/yolo/config', {
        headers: {
            'X-API-KEY': state.apiKey
        }
    })
    .then(response => response.json())
    .then(data => {
        state.yolo.enabled = data.enabled;
        state.yolo.model = data.model;
        state.yolo.confidence = data.confidence;
        
        // Update UI
        const yoloToggle = document.getElementById('yolo-toggle');
        const yoloModelSelector = document.getElementById('yolo-model-selector');
        
        if (yoloToggle) {
            yoloToggle.checked = data.enabled;
        }
        
        if (yoloModelSelector) {
            yoloModelSelector.value = data.model;
            yoloModelSelector.style.display = data.enabled ? 'inline-block' : 'none';
        }
    })
    .catch(error => {
        console.error('Error fetching YOLO status:', error);
    });
}

// Toggle YOLO processing
function toggleYolo(enabled) {
    if (!state.socket) {
        showNotification('Error', 'WebSocket not connected', 'danger');
        return;
    }
    
    state.socket.send(JSON.stringify({
        type: 'yolo_toggle',
        command: 'toggle',
        enabled: enabled,
        model: state.yolo.model,
        confidence: state.yolo.confidence
    }));
    
    // Update state
    state.yolo.enabled = enabled;
    
    // Show notification
    showNotification(
        'YOLO Vision', 
        enabled ? 'YOLO vision processing enabled' : 'YOLO vision processing disabled',
        enabled ? 'success' : 'info'
    );
    
    // Add to system logs
    addSystemLog(enabled ? 'YOLO vision processing enabled' : 'YOLO vision processing disabled');
}

// Update YOLO model
function updateYoloModel(model) {
    if (!state.socket) {
        showNotification('Error', 'WebSocket not connected', 'danger');
        return;
    }
    
    state.socket.send(JSON.stringify({
        type: 'yolo_toggle',
        command: 'toggle',
        enabled: true,
        model: model,
        confidence: state.yolo.confidence
    }));
    
    // Update state
    state.yolo.model = model;
    
    // Show notification
    showNotification('YOLO Model', `YOLO model changed to ${model}`, 'success');
    
    // Add to system logs
    addSystemLog(`YOLO model changed to ${model}`);
}

// Basic DOM ready handler
document.addEventListener('DOMContentLoaded', function() {
    console.log("GGFAI dashboard loaded");
    
    // Initialize components
    initResourceChart();
    initTagVisualization();
    initExplanationDisplay();
    initChatInterface();
    initThoughtProcess();
    initSystemMonitoring();
    initVoiceInput();
    initYoloToggle();
    initWebSocket();
    
    // Set default model in UI
    const modelDisplay = document.getElementById('current-model-display');
    if (modelDisplay) {
        // This will be updated once the actual model is loaded
        modelDisplay.textContent = "Loading...";
    }
    
    // Check system health
    checkSystemHealth();
    // Set up periodic health checks
    setInterval(checkSystemHealth, 30000); // Check every 30 seconds
    
    // Show welcome notification
    setTimeout(() => {
        showNotification('Welcome', 'GGFAI Dashboard loaded successfully', 'success');
    }, 1000);
});