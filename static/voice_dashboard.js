/**
 * Voice Processing Dashboard Component
 * Provides UI controls for voice input/output settings
 */

class VoiceDashboard {
    /**
     * Initialize voice dashboard
     * @param {string} containerId - ID of container element
     * @param {Object} initialConfig - Initial voice configuration
     * @param {Function} onConfigChange - Callback when config changes
     */
    constructor(containerId, initialConfig = {}, onConfigChange = null) {
        this.container = document.getElementById(containerId);
        this.config = initialConfig;
        this.onConfigChange = onConfigChange;
        this.availableVoices = [];
        this.availableTTSEngines = [];
        this.availableRecognitionEngines = [];
        
        if (!this.container) {
            console.error(`Container element with ID ${containerId} not found`);
            return;
        }
        
        this.init();
    }
    
    /**
     * Initialize dashboard
     */
    init() {
        this.render();
        this.fetchAvailableOptions();
        this.attachEventListeners();
        this.updateVADOptions();
    }
    
    /**
     * Render dashboard UI
     */
    render() {
        this.container.innerHTML = `
            <div class="voice-dashboard">
                <h2>Voice Processing Settings</h2>
                
                <div class="dashboard-section">
                    <h3>Speech Recognition</h3>
                    
                    <div class="form-group">
                        <label for="recognition-engine">Recognition Engine:</label>
                        <select id="recognition-engine" multiple>
                            <option value="loading">Loading...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="energy-threshold">Energy Threshold:</label>
                        <input type="range" id="energy-threshold" min="100" max="1000" step="10" 
                               value="${this.config.energy_threshold || 300}">
                        <span id="energy-threshold-value">${this.config.energy_threshold || 300}</span>
                    </div>
                    
                    <div class="form-group">
                        <label for="pause-threshold">Pause Threshold (s):</label>
                        <input type="range" id="pause-threshold" min="0.3" max="2.0" step="0.1" 
                               value="${this.config.pause_threshold || 0.8}">
                        <span id="pause-threshold-value">${this.config.pause_threshold || 0.8}</span>
                    </div>
                    
                    <div class="form-group">
                        <label for="dynamic-threshold">
                            <input type="checkbox" id="dynamic-threshold" 
                                   ${this.config.dynamic_energy_threshold ? 'checked' : ''}>
                            Dynamic Energy Threshold
                        </label>
                    </div>
                    
                    <div class="form-group">
                        <label for="vad-system">Voice Activity Detection System:</label>
                        <select id="vad-system">
                            <option value="ENERGY" ${this.config.vad_system === 'ENERGY' ? 'selected' : ''}>Energy-based (Default)</option>
                            <option value="WEBRTC" ${this.config.vad_system === 'WEBRTC' ? 'selected' : ''}>WebRTCVAD</option>
                            <option value="SILERO" ${this.config.vad_system === 'SILERO' ? 'selected' : ''}>Silero VAD</option>
                        </select>
                    </div>
                    
                    <div class="form-group vad-settings" id="vad-sensitivity-group">
                        <label for="vad-sensitivity">VAD Sensitivity:</label>
                        <input type="range" id="vad-sensitivity" min="0" max="3" step="1" 
                               value="${this.config.vad_sensitivity || 2}">
                        <span id="vad-sensitivity-value">${this.config.vad_sensitivity || 2}</span>
                        <small>Higher values are more sensitive (0-3)</small>
                    </div>
                    
                    <div class="form-group vad-settings" id="vad-threshold-group" style="display: none;">
                        <label for="vad-threshold">Silero VAD Threshold:</label>
                        <input type="range" id="vad-threshold" min="0" max="1" step="0.05" 
                               value="${this.config.vad_threshold || 0.5}">
                        <span id="vad-threshold-value">${this.config.vad_threshold || 0.5}</span>
                        <small>Higher values require more confidence (0.0-1.0)</small>
                    </div>
                    
                    <div class="form-group">
                        <button id="test-mic-btn" class="btn">Test Microphone</button>
                        <span id="mic-status"></span>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h3>Text-to-Speech</h3>
                    
                    <div class="form-group">
                        <label for="tts-engine">TTS Engine:</label>
                        <select id="tts-engine">
                            <option value="loading">Loading...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="tts-voice">Voice:</label>
                        <select id="tts-voice">
                            <option value="loading">Loading...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="tts-rate">Speech Rate:</label>
                        <input type="range" id="tts-rate" min="100" max="300" step="5" 
                               value="${this.config.tts_rate || 175}">
                        <span id="tts-rate-value">${this.config.tts_rate || 175}</span>
                    </div>
                    
                    <div class="form-group">
                        <label for="tts-volume">Volume:</label>
                        <input type="range" id="tts-volume" min="0" max="1" step="0.1" 
                               value="${this.config.tts_volume || 1.0}">
                        <span id="tts-volume-value">${this.config.tts_volume || 1.0}</span>
                    </div>
                    
                    <div class="form-group">
                        <input type="text" id="tts-test-text" placeholder="Enter text to test TTS">
                        <button id="test-tts-btn" class="btn">Test TTS</button>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h3>Advanced Settings</h3>
                    
                    <div class="form-group">
                        <label for="api-keys">API Keys:</label>
                        <div id="api-keys-container">
                            <div class="api-key-row">
                                <input type="text" placeholder="Service (e.g., azure_key)" class="api-key-name">
                                <input type="password" placeholder="API Key" class="api-key-value">
                                <button class="remove-api-key-btn">✕</button>
                            </div>
                        </div>
                        <button id="add-api-key-btn" class="btn">Add API Key</button>
                    </div>
                    
                    <div class="form-group">
                        <button id="save-config-btn" class="btn primary">Save Configuration</button>
                        <button id="reset-config-btn" class="btn">Reset to Defaults</button>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Fetch available options from server
     */
    fetchAvailableOptions() {
        // In a real implementation, these would be fetched from the server
        // For now, we'll use mock data
        
        // Mock available recognition engines
        this.availableRecognitionEngines = [
            { id: 'GOOGLE', name: 'Google Web Speech API' },
            { id: 'SPHINX', name: 'CMU Sphinx (Offline)' },
            { id: 'WHISPER', name: 'OpenAI Whisper' },
            { id: 'VOSK', name: 'Vosk (Offline)' },
            { id: 'AZURE', name: 'Microsoft Azure' },
            { id: 'GOOGLE_CLOUD', name: 'Google Cloud Speech' },
            { id: 'AMAZON', name: 'Amazon Transcribe' },
            { id: 'OLLAMA', name: 'Ollama' }
        ];
        
        // Mock available TTS engines
        this.availableTTSEngines = [
            { id: 'SYSTEM', name: 'System Default' },
            { id: 'PYTTSX3', name: 'pyttsx3' },
            { id: 'ESPEAK', name: 'eSpeak' },
            { id: 'ELEVENLABS', name: 'ElevenLabs' },
            { id: 'COQUI', name: 'Coqui TTS' },
            { id: 'KOKORO', name: 'Kokoro' },
            { id: 'OLLAMA', name: 'Ollama' }
        ];
        
        // Mock available voices
        this.availableVoices = [
            { id: 'default', name: 'Default', engine: 'system' },
            { id: 'en-us-male', name: 'English (US) - Male', engine: 'system' },
            { id: 'en-us-female', name: 'English (US) - Female', engine: 'system' },
            { id: 'en-gb-male', name: 'English (UK) - Male', engine: 'system' },
            { id: 'en-gb-female', name: 'English (UK) - Female', engine: 'system' }
        ];
        
        // Update UI with fetched options
        this.updateAvailableOptions();
    }
    
    /**
     * Update UI with available options
     */
    updateAvailableOptions() {
        // Update recognition engines
        const recognitionEngineSelect = document.getElementById('recognition-engine');
        recognitionEngineSelect.innerHTML = '';
        
        this.availableRecognitionEngines.forEach(engine => {
            const option = document.createElement('option');
            option.value = engine.id;
            option.textContent = engine.name;
            
            // Check if this engine is selected in the config
            if (this.config.recognition_engines && 
                this.config.recognition_engines.includes(engine.id)) {
                option.selected = true;
            }
            
            recognitionEngineSelect.appendChild(option);
        });
        
        // Update TTS engines
        const ttsEngineSelect = document.getElementById('tts-engine');
        ttsEngineSelect.innerHTML = '';
        
        this.availableTTSEngines.forEach(engine => {
            const option = document.createElement('option');
            option.value = engine.id;
            option.textContent = engine.name;
            
            // Check if this engine is selected in the config
            if (this.config.tts_engine === engine.id) {
                option.selected = true;
            }
            
            ttsEngineSelect.appendChild(option);
        });
        
        // Update voices
        this.updateVoiceOptions();
    }
    
    /**
     * Update VAD options based on selected VAD system
     */
    updateVADOptions() {
        const vadSystemSelect = document.getElementById('vad-system');
        const vadSensitivityGroup = document.getElementById('vad-sensitivity-group');
        const vadThresholdGroup = document.getElementById('vad-threshold-group');
        const selectedSystem = vadSystemSelect.value;
        
        // Show/hide appropriate controls based on selected VAD system
        if (selectedSystem === 'WEBRTC') {
            vadSensitivityGroup.style.display = 'block';
            vadThresholdGroup.style.display = 'none';
        } else if (selectedSystem === 'SILERO') {
            vadSensitivityGroup.style.display = 'none';
            vadThresholdGroup.style.display = 'block';
        } else {
            // Energy-based VAD
            vadSensitivityGroup.style.display = 'block';
            vadThresholdGroup.style.display = 'none';
        }
        
        // Update config
        this.config.vad_system = selectedSystem;
    }
    
    /**
     * Update voice options based on selected TTS engine
     */
    updateVoiceOptions() {
        const ttsEngineSelect = document.getElementById('tts-engine');
        const ttsVoiceSelect = document.getElementById('tts-voice');
        const selectedEngine = ttsEngineSelect.value;
        
        // Filter voices by engine
        const filteredVoices = this.availableVoices.filter(
            voice => voice.engine === 'system' || voice.engine.toUpperCase() === selectedEngine
        );
        
        // Update voice select
        ttsVoiceSelect.innerHTML = '';
        
        filteredVoices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice.id;
            option.textContent = voice.name;
            
            // Check if this voice is selected in the config
            if (this.config.tts_voice === voice.id) {
                option.selected = true;
            }
            
            ttsVoiceSelect.appendChild(option);
        });
        
        // If no voices available, show message
        if (filteredVoices.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No voices available for this engine';
            ttsVoiceSelect.appendChild(option);
        }
    }
    
    /**
     * Attach event listeners to UI elements
     */
    attachEventListeners() {
        // Energy threshold slider
        const energyThresholdSlider = document.getElementById('energy-threshold');
        const energyThresholdValue = document.getElementById('energy-threshold-value');
        
        energyThresholdSlider.addEventListener('input', () => {
            energyThresholdValue.textContent = energyThresholdSlider.value;
        });
        
        // Pause threshold slider
        const pauseThresholdSlider = document.getElementById('pause-threshold');
        const pauseThresholdValue = document.getElementById('pause-threshold-value');
        
        pauseThresholdSlider.addEventListener('input', () => {
            pauseThresholdValue.textContent = pauseThresholdSlider.value;
        });
        
        // TTS rate slider
        const ttsRateSlider = document.getElementById('tts-rate');
        const ttsRateValue = document.getElementById('tts-rate-value');
        
        ttsRateSlider.addEventListener('input', () => {
            ttsRateValue.textContent = ttsRateSlider.value;
        });
        
        // TTS volume slider
        const ttsVolumeSlider = document.getElementById('tts-volume');
        const ttsVolumeValue = document.getElementById('tts-volume-value');
        
        ttsVolumeSlider.addEventListener('input', () => {
            ttsVolumeValue.textContent = ttsVolumeSlider.value;
        });
        
        // VAD sensitivity slider
        const vadSensitivitySlider = document.getElementById('vad-sensitivity');
        const vadSensitivityValue = document.getElementById('vad-sensitivity-value');
        
        vadSensitivitySlider.addEventListener('input', () => {
            vadSensitivityValue.textContent = vadSensitivitySlider.value;
        });
        
        // VAD threshold slider
        const vadThresholdSlider = document.getElementById('vad-threshold');
        const vadThresholdValue = document.getElementById('vad-threshold-value');
        
        vadThresholdSlider.addEventListener('input', () => {
            vadThresholdValue.textContent = vadThresholdSlider.value;
        });
        
        // VAD system change
        const vadSystemSelect = document.getElementById('vad-system');
        
        vadSystemSelect.addEventListener('change', () => {
            this.updateVADOptions();
        });
        
        // TTS engine change
        const ttsEngineSelect = document.getElementById('tts-engine');
        
        ttsEngineSelect.addEventListener('change', () => {
            this.updateVoiceOptions();
        });
        
        // Test microphone button
        const testMicBtn = document.getElementById('test-mic-btn');
        const micStatus = document.getElementById('mic-status');
        
        testMicBtn.addEventListener('click', () => {
            micStatus.textContent = 'Listening...';
            
            // In a real implementation, this would call the server to test the microphone
            // For now, we'll simulate a successful test
            setTimeout(() => {
                micStatus.textContent = 'Microphone working!';
                setTimeout(() => {
                    micStatus.textContent = '';
                }, 3000);
            }, 1500);
        });
        
        // Test TTS button
        const testTTSBtn = document.getElementById('test-tts-btn');
        const ttsTestText = document.getElementById('tts-test-text');
        
        testTTSBtn.addEventListener('click', () => {
            const text = ttsTestText.value.trim();
            
            if (!text) {
                alert('Please enter text to test TTS');
                return;
            }
            
            // In a real implementation, this would call the server to test TTS
            // For now, we'll simulate a successful test
            testTTSBtn.textContent = 'Speaking...';
            setTimeout(() => {
                testTTSBtn.textContent = 'Test TTS';
            }, 2000);
        });
        
        // Add API key button
        const addApiKeyBtn = document.getElementById('add-api-key-btn');
        const apiKeysContainer = document.getElementById('api-keys-container');
        
        addApiKeyBtn.addEventListener('click', () => {
            const row = document.createElement('div');
            row.className = 'api-key-row';
            row.innerHTML = `
                <input type="text" placeholder="Service (e.g., azure_key)" class="api-key-name">
                <input type="password" placeholder="API Key" class="api-key-value">
                <button class="remove-api-key-btn">✕</button>
            `;
            
            apiKeysContainer.appendChild(row);
            
            // Attach event listener to remove button
            row.querySelector('.remove-api-key-btn').addEventListener('click', () => {
                row.remove();
            });
        });
        
        // Remove API key buttons
        document.querySelectorAll('.remove-api-key-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                btn.parentElement.remove();
            });
        });
        
        // Save configuration button
        const saveConfigBtn = document.getElementById('save-config-btn');
        
        saveConfigBtn.addEventListener('click', () => {
            const config = this.getConfigFromUI();
            
            // Call onConfigChange callback if provided
            if (this.onConfigChange) {
                this.onConfigChange(config);
            }
            
            // In a real implementation, this would save the config to the server
            saveConfigBtn.textContent = 'Saved!';
            setTimeout(() => {
                saveConfigBtn.textContent = 'Save Configuration';
            }, 2000);
        });
        
        // Reset configuration button
        const resetConfigBtn = document.getElementById('reset-config-btn');
        
        resetConfigBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to reset to default settings?')) {
                // Reset to default config
                this.config = {
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
                };
                
                // Update UI
                this.updateUIFromConfig();
            }
        });
    }
    
    /**
     * Get configuration from UI elements
     * @returns {Object} Configuration object
     */
    getConfigFromUI() {
        const config = {};
        
        // Speech recognition settings
        config.energy_threshold = parseInt(document.getElementById('energy-threshold').value);
        config.pause_threshold = parseFloat(document.getElementById('pause-threshold').value);
        config.dynamic_energy_threshold = document.getElementById('dynamic-threshold').checked;
        
        // VAD settings
        config.vad_system = document.getElementById('vad-system').value;
        config.vad_sensitivity = parseInt(document.getElementById('vad-sensitivity').value);
        config.vad_threshold = parseFloat(document.getElementById('vad-threshold').value);
        
        // Get selected recognition engines
        const recognitionEngineSelect = document.getElementById('recognition-engine');
        config.recognition_engines = Array.from(recognitionEngineSelect.selectedOptions).map(option => option.value);
        
        // TTS settings
        config.tts_engine = document.getElementById('tts-engine').value;
        config.tts_voice = document.getElementById('tts-voice').value;
        config.tts_rate = parseInt(document.getElementById('tts-rate').value);
        config.tts_volume = parseFloat(document.getElementById('tts-volume').value);
        
        // API keys
        config.api_keys = {};
        document.querySelectorAll('.api-key-row').forEach(row => {
            const keyName = row.querySelector('.api-key-name').value.trim();
            const keyValue = row.querySelector('.api-key-value').value.trim();
            
            if (keyName && keyValue) {
                config.api_keys[keyName] = keyValue;
            }
        });
        
        return config;
    }
    
    /**
     * Update UI elements from configuration
     */
    updateUIFromConfig() {
        // Speech recognition settings
        document.getElementById('energy-threshold').value = this.config.energy_threshold || 300;
        document.getElementById('energy-threshold-value').textContent = this.config.energy_threshold || 300;
        
        document.getElementById('pause-threshold').value = this.config.pause_threshold || 0.8;
        document.getElementById('pause-threshold-value').textContent = this.config.pause_threshold || 0.8;
        
        document.getElementById('dynamic-threshold').checked = this.config.dynamic_energy_threshold !== false;
        
        // VAD settings
        const vadSystemSelect = document.getElementById('vad-system');
        if (this.config.vad_system) {
            vadSystemSelect.value = this.config.vad_system;
        }
        
        document.getElementById('vad-sensitivity').value = this.config.vad_sensitivity || 2;
        document.getElementById('vad-sensitivity-value').textContent = this.config.vad_sensitivity || 2;
        
        document.getElementById('vad-threshold').value = this.config.vad_threshold || 0.5;
        document.getElementById('vad-threshold-value').textContent = this.config.vad_threshold || 0.5;
        
        // Update VAD options display
        this.updateVADOptions();
        
        // TTS settings
        document.getElementById('tts-rate').value = this.config.tts_rate || 175;
        document.getElementById('tts-rate-value').textContent = this.config.tts_rate || 175;
        
        document.getElementById('tts-volume').value = this.config.tts_volume || 1.0;
        document.getElementById('tts-volume-value').textContent = this.config.tts_volume || 1.0;
        
        // Update available options (this will also update the selects)
        this.updateAvailableOptions();
        
        // API keys
        const apiKeysContainer = document.getElementById('api-keys-container');
        apiKeysContainer.innerHTML = '';
        
        if (this.config.api_keys) {
            Object.entries(this.config.api_keys).forEach(([key, value]) => {
                const row = document.createElement('div');
                row.className = 'api-key-row';
                row.innerHTML = `
                    <input type="text" value="${key}" class="api-key-name">
                    <input type="password" value="${value}" class="api-key-value">
                    <button class="remove-api-key-btn">✕</button>
                `;
                
                apiKeysContainer.appendChild(row);
                
                // Attach event listener to remove button
                row.querySelector('.remove-api-key-btn').addEventListener('click', () => {
                    row.remove();
                });
            });
        }
        
        // If no API keys, add an empty row
        if (apiKeysContainer.children.length === 0) {
            const row = document.createElement('div');
            row.className = 'api-key-row';
            row.innerHTML = `
                <input type="text" placeholder="Service (e.g., azure_key)" class="api-key-name">
                <input type="password" placeholder="API Key" class="api-key-value">
                <button class="remove-api-key-btn">✕</button>
            `;
            
            apiKeysContainer.appendChild(row);
            
            // Attach event listener to remove button
            row.querySelector('.remove-api-key-btn').addEventListener('click', () => {
                row.remove();
            });
        }
    }
    
    /**
     * Update configuration
     * @param {Object} config - New configuration
     */
    updateConfig(config) {
        this.config = config;
        this.updateUIFromConfig();
    }
}

// Example usage:
// const voiceDashboard = new VoiceDashboard('voice-dashboard-container', initialConfig, (config) => {
//     console.log('Config updated:', config);
//     // Send config to server
// });