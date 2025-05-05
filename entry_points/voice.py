# voice.py - Production-level Voice Processing Module for GGFAI Framework
# Handles both speech recognition (STT) and text-to-speech (TTS) with multiple engines

import os
import time
import json
import logging
import threading
import tempfile
import queue
import uuid
import subprocess
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from pathlib import Path
import concurrent.futures

# Speech recognition
import speech_recognition as sr

# Audio processing
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Basic TTS fallback
import pyttsx3

# Core imports
from ..core.tag_registry import Tag
from ..core.run_with_grace import run_with_grace

# Constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_PHRASE_TIMEOUT = 5
DEFAULT_PHRASE_LIMIT = 15
DEFAULT_ENERGY_THRESHOLD = 300
DEFAULT_PAUSE_THRESHOLD = 0.8
DEFAULT_DYNAMIC_THRESHOLD = True
DEFAULT_NOISE_DURATION = 1.0
DEFAULT_VAD_SENSITIVITY = 2
DEFAULT_AUDIO_DIR = "audio_cache"
DEFAULT_TTS_VOICE = "default"
DEFAULT_TTS_RATE = 175
DEFAULT_TTS_VOLUME = 1.0
DEFAULT_TTS_ENGINE = "system"
MAX_PROCESSING_THREADS = 4
AUDIO_CACHE_MAX_SIZE_MB = 100
AUDIO_CACHE_MAX_AGE_DAYS = 7

# Supported speech recognition engines
class RecognitionEngine(Enum):
    """Available speech recognition engines."""
    GOOGLE = auto()
    WHISPER = auto()
    SPHINX = auto()
    VOSK = auto()
    AZURE = auto()
    GOOGLE_CLOUD = auto()
    AMAZON = auto()
    OLLAMA = auto()
    SYSTEM = auto()

# Supported TTS engines
class TTSEngine(Enum):
    """Available text-to-speech engines."""
    SYSTEM = auto()  # pyttsx3
    ESPEAK = auto()
    PYTTSX3 = auto()
    GOOGLE = auto()
    AZURE = auto()
    AMAZON = auto()
    ELEVENLABS = auto()
    COQUI = auto()
    KOKORO = auto()
    OLLAMA = auto()

# Supported Voice Activity Detection systems
class VADSystem(Enum):
    """Supported Voice Activity Detection systems."""
    ENERGY = auto()  # Default energy-based VAD from SpeechRecognition
    WEBRTC = auto()  # WebRTCVAD
    SILERO = auto()  # Silero VAD

@dataclass
class VoiceConfig:
    """Configuration for voice processing."""
    # Speech recognition settings
    energy_threshold: int = DEFAULT_ENERGY_THRESHOLD
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD
    dynamic_energy_threshold: bool = DEFAULT_DYNAMIC_THRESHOLD
    mic_index: Optional[int] = None
    recognition_engines: List[RecognitionEngine] = field(
        default_factory=lambda: [
            RecognitionEngine.GOOGLE,
            RecognitionEngine.SPHINX
        ]
    )
    max_phrase_length: int = DEFAULT_PHRASE_LIMIT  # seconds
    min_phrase_length: float = 0.3  # seconds
    noise_adaptation_time: float = DEFAULT_NOISE_DURATION  # seconds
    
    # Voice Activity Detection settings
    vad_system: VADSystem = VADSystem.ENERGY  # Default to energy-based VAD
    vad_sensitivity: int = DEFAULT_VAD_SENSITIVITY  # 1-3 where 3 is most sensitive
    vad_frame_duration: int = 30  # Frame duration in ms for WebRTCVAD (valid values: 10, 20, 30)
    vad_threshold: float = 0.5  # Threshold for Silero VAD (0.0-1.0)
    vad_min_speech_duration_ms: int = 250  # Minimum speech duration in ms for Silero VAD
    vad_min_silence_duration_ms: int = 100  # Minimum silence duration in ms for Silero VAD
    vad_window_size_samples: int = 512  # Window size in samples for Silero VAD
    
    save_audio_chunks: bool = False  # For debugging
    
    # TTS settings
    tts_engine: TTSEngine = TTSEngine.SYSTEM
    tts_voice: str = DEFAULT_TTS_VOICE
    tts_rate: int = DEFAULT_TTS_RATE
    tts_volume: float = DEFAULT_TTS_VOLUME
    
    # API keys and credentials
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Audio processing
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    audio_cache_dir: str = DEFAULT_AUDIO_DIR
    
    # Performance settings
    max_threads: int = MAX_PROCESSING_THREADS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = asdict(self)
        # Convert enums to strings for serialization
        result["recognition_engines"] = [e.name for e in self.recognition_engines]
        result["tts_engine"] = self.tts_engine.name
        result["vad_system"] = self.vad_system.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceConfig':
        """Create config from dictionary."""
        # Convert string engine names to enum values
        if "recognition_engines" in data:
            data["recognition_engines"] = [
                RecognitionEngine[e] for e in data["recognition_engines"]
            ]
        if "tts_engine" in data:
            data["tts_engine"] = TTSEngine[data["tts_engine"]]
        if "vad_system" in data:
            data["vad_system"] = VADSystem[data["vad_system"]]
        return cls(**data)
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'VoiceConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

class AudioCache:
    """
    Manages cached audio files with automatic cleanup.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, cache_dir: str = DEFAULT_AUDIO_DIR, 
                 max_size_mb: int = AUDIO_CACHE_MAX_SIZE_MB,
                 max_age_days: int = AUDIO_CACHE_MAX_AGE_DAYS):
        """
        Initialize audio cache.
        
        Args:
            cache_dir: Directory to store cached audio files
            max_size_mb: Maximum cache size in MB
            max_age_days: Maximum age of cached files in days
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.max_age_days = max_age_days
        self.logger = logging.getLogger("GGFAI.voice.cache")
        self._lock = threading.RLock()
        
        # Create cache directory if it doesn't exist
        self._ensure_cache_dir()
        
        # Start background cleanup thread
        self._start_cleanup_thread()
    
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Audio cache directory: {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create audio cache directory: {e}")
            # Fall back to temp directory
            self.cache_dir = Path(tempfile.gettempdir()) / "ggfai_audio_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using fallback audio cache directory: {self.cache_dir}")
    
    def _start_cleanup_thread(self) -> None:
        """Start background thread for periodic cache cleanup."""
        def cleanup_worker():
            while True:
                try:
                    self.cleanup()
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
                # Run cleanup every hour
                time.sleep(3600)
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
    
    def get_path(self, filename: str) -> Path:
        """Get full path for a cached file."""
        return self.cache_dir / filename
    
    def save_audio(self, audio_data: bytes, format: str = "wav") -> str:
        """
        Save audio data to cache.
        
        Args:
            audio_data: Raw audio bytes
            format: Audio format (wav, mp3, etc.)
            
        Returns:
            Filename of cached audio
        """
        with self._lock:
            # Generate unique filename
            filename = f"{uuid.uuid4()}.{format}"
            path = self.get_path(filename)
            
            try:
                with open(path, "wb") as f:
                    f.write(audio_data)
                self.logger.debug(f"Saved audio to cache: {filename}")
                return filename
            except Exception as e:
                self.logger.error(f"Failed to save audio to cache: {e}")
                return ""
    
    def load_audio(self, filename: str) -> Optional[bytes]:
        """
        Load audio data from cache.
        
        Args:
            filename: Cached audio filename
            
        Returns:
            Raw audio bytes or None if not found
        """
        with self._lock:
            path = self.get_path(filename)
            if not path.exists():
                return None
            
            try:
                with open(path, "rb") as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"Failed to load audio from cache: {e}")
                return None
    
    def cleanup(self) -> int:
        """
        Clean up old and excess cache files.
        
        Returns:
            Number of files removed
        """
        with self._lock:
            self.logger.info("Running audio cache cleanup")
            removed = 0
            
            try:
                # Get all cache files with their stats
                files = []
                total_size = 0
                
                for path in self.cache_dir.glob("*.*"):
                    if path.is_file():
                        stats = path.stat()
                        age_days = (time.time() - stats.st_mtime) / (24 * 3600)
                        size_mb = stats.st_size / (1024 * 1024)
                        total_size += size_mb
                        files.append((path, age_days, size_mb))
                
                # Sort by age (oldest first)
                files.sort(key=lambda x: x[1], reverse=True)
                
                # Remove old files
                for path, age_days, _ in files:
                    if age_days > self.max_age_days:
                        path.unlink()
                        removed += 1
                
                # If still over size limit, remove more files
                if total_size > self.max_size_mb:
                    # Re-check remaining files
                    remaining = [f for f in files if f[0].exists()]
                    remaining.sort(key=lambda x: x[1], reverse=True)  # Oldest first
                    
                    for path, _, size_mb in remaining:
                        if total_size <= self.max_size_mb:
                            break
                        path.unlink()
                        total_size -= size_mb
                        removed += 1
                
                self.logger.info(f"Removed {removed} cached audio files")
                return removed
            except Exception as e:
                self.logger.error(f"Cache cleanup failed: {e}")
                return 0

class VoiceInputProcessor:
    """
    Production-level voice input processor with multiple recognition engines,
    error recovery, and performance optimizations.
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        """
        Initialize voice processor with configuration.
        
        Args:
            config: Voice processing configuration
        """
        self.config = config or VoiceConfig()
        self.logger = logging.getLogger("GGFAI.voice.input")
        
        # Initialize recognizer with config
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self.config.energy_threshold
        self.recognizer.pause_threshold = self.config.pause_threshold
        self.recognizer.dynamic_energy_threshold = self.config.dynamic_energy_threshold
        
        # Initialize thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_threads
        )
        
        # Initialize audio cache
        self.audio_cache = AudioCache(self.config.audio_cache_dir)
        
        # Initialize microphone
        self._verify_microphone()
        
        # Initialize optional engines
        self._init_optional_engines()
        
        self.logger.info("Voice input processor initialized")
    
    def _verify_microphone(self) -> None:
        """Check if specified microphone is available and list alternatives."""
        try:
            mics = sr.Microphone.list_microphone_names()
            self.logger.info(f"Available microphones: {len(mics)}")
            
            for i, mic in enumerate(mics):
                self.logger.debug(f"Mic {i}: {mic}")
            
            if self.config.mic_index is not None:
                if self.config.mic_index >= len(mics):
                    self.logger.warning(
                        f"Mic index {self.config.mic_index} not found. Using default."
                    )
                    self.config.mic_index = None
                else:
                    self.logger.info(
                        f"Using microphone {self.config.mic_index}: {mics[self.config.mic_index]}"
                    )
        except Exception as e:
            self.logger.error(f"Microphone check failed: {str(e)}")
    
    def _init_optional_engines(self) -> None:
        """Initialize optional recognition engines and VAD systems."""
        # Initialize recognition engines
        self._init_recognition_engines()
        
        # Initialize VAD systems
        self._init_vad_systems()
    
    def _init_recognition_engines(self) -> None:
        """Initialize optional speech recognition engines."""
        # Check for Whisper
        if RecognitionEngine.WHISPER in self.config.recognition_engines:
            try:
                import whisper
                self.whisper_model = whisper.load_model("base")
                self.logger.info("Whisper model loaded")
            except ImportError:
                self.logger.warning("Whisper not installed, removing from engines")
                self.config.recognition_engines.remove(RecognitionEngine.WHISPER)
        
        # Check for Vosk
        if RecognitionEngine.VOSK in self.config.recognition_engines:
            try:
                from vosk import Model, KaldiRecognizer
                model_path = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-en-us")
                if os.path.exists(model_path):
                    self.vosk_model = Model(model_path)
                    self.logger.info(f"Vosk model loaded from {model_path}")
                else:
                    self.logger.warning(f"Vosk model not found at {model_path}, removing from engines")
                    self.config.recognition_engines.remove(RecognitionEngine.VOSK)
            except ImportError:
                self.logger.warning("Vosk not installed, removing from engines")
                self.config.recognition_engines.remove(RecognitionEngine.VOSK)
        
        # Check for cloud services
        for engine in [RecognitionEngine.AZURE, RecognitionEngine.GOOGLE_CLOUD, 
                      RecognitionEngine.AMAZON, RecognitionEngine.OLLAMA]:
            if engine in self.config.recognition_engines:
                key_name = f"{engine.name.lower()}_key"
                if key_name not in self.config.api_keys or not self.config.api_keys[key_name]:
                    self.logger.warning(f"No API key for {engine.name}, removing from engines")
                    self.config.recognition_engines.remove(engine)
    
    def _init_vad_systems(self) -> None:
        """Initialize Voice Activity Detection systems."""
        # Initialize WebRTCVAD
        if self.config.vad_system == VADSystem.WEBRTC:
            try:
                import webrtcvad
                # Validate frame duration (valid values: 10, 20, 30)
                valid_durations = [10, 20, 30]
                if self.config.vad_frame_duration not in valid_durations:
                    self.logger.warning(
                        f"Invalid WebRTCVAD frame duration: {self.config.vad_frame_duration}ms. "
                        f"Using default: 30ms"
                    )
                    self.config.vad_frame_duration = 30
                
                # Validate sensitivity (valid values: 0, 1, 2, 3)
                if not 0 <= self.config.vad_sensitivity <= 3:
                    self.logger.warning(
                        f"Invalid WebRTCVAD sensitivity: {self.config.vad_sensitivity}. "
                        f"Using default: 2"
                    )
                    self.config.vad_sensitivity = 2
                
                # Initialize WebRTCVAD
                self.webrtc_vad = webrtcvad.Vad(self.config.vad_sensitivity)
                self.logger.info(
                    f"WebRTCVAD initialized (sensitivity={self.config.vad_sensitivity}, "
                    f"frame_duration={self.config.vad_frame_duration}ms)"
                )
            except ImportError:
                self.logger.warning("WebRTCVAD not installed, falling back to energy-based VAD")
                self.config.vad_system = VADSystem.ENERGY
        
        # Initialize Silero VAD
        elif self.config.vad_system == VADSystem.SILERO:
            try:
                import torch
                
                # Validate threshold (valid range: 0.0-1.0)
                if not 0.0 <= self.config.vad_threshold <= 1.0:
                    self.logger.warning(
                        f"Invalid Silero VAD threshold: {self.config.vad_threshold}. "
                        f"Using default: 0.5"
                    )
                    self.config.vad_threshold = 0.5
                
                # Initialize Silero VAD
                self.silero_model, self.silero_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                
                # Get the VAD function from utils
                self.silero_get_speech_timestamps = self.silero_utils[0]
                self.silero_save_audio = self.silero_utils[1]
                self.silero_read_audio = self.silero_utils[2]
                self.silero_vad_collect_chunks = self.silero_utils[3]
                
                # Move model to CPU (or GPU if available)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.silero_model = self.silero_model.to(self.device)
                
                self.logger.info(
                    f"Silero VAD initialized (threshold={self.config.vad_threshold}, "
                    f"device={self.device})"
                )
            except ImportError as e:
                self.logger.warning(f"Silero VAD not installed ({str(e)}), falling back to energy-based VAD")
                self.config.vad_system = VADSystem.ENERGY
            except Exception as e:
                self.logger.error(f"Failed to initialize Silero VAD: {str(e)}")
                self.config.vad_system = VADSystem.ENERGY
        
        # Default to energy-based VAD
        if self.config.vad_system == VADSystem.ENERGY:
            self.logger.info("Using energy-based VAD from SpeechRecognition library")
    
    def _adjust_for_ambient_noise(self, source) -> None:
        """
        Calibrate recognizer for current noise conditions.
        Implements adaptive noise floor detection.
        """
        try:
            self.logger.info("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(
                source, duration=self.config.noise_adaptation_time
            )
            self.logger.info(f"Energy threshold set to {self.recognizer.energy_threshold}")
        except Exception as e:
            self.logger.warning(f"Ambient noise adjustment failed: {str(e)}")
    
    def _capture_with_webrtc_vad(self, source, timeout: int, phrase_time_limit: int) -> Optional[sr.AudioData]:
        """
        Capture voice input using WebRTCVAD for voice activity detection.
        
        Args:
            source: Audio source (microphone)
            timeout: Seconds to wait for speech before giving up
            phrase_time_limit: Maximum length of voice command in seconds
            
        Returns:
            AudioData object or None if no speech detected
        """
        import webrtcvad
        import array
        import struct
        
        # WebRTCVAD requires specific sample rates (8000, 16000, 32000, 48000)
        sample_rate = 16000
        frame_duration_ms = self.config.vad_frame_duration
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Initialize variables for recording
        frames = []
        is_speech = False
        silent_frames = 0
        max_silent_frames = int(1000 / frame_duration_ms)  # 1 second of silence
        start_time = time.time()
        end_time = start_time + timeout
        
        # Adjust the source to use the required sample rate
        with source.recorder(sample_rate=sample_rate) as recorder:
            self.logger.info(f"Listening with WebRTCVAD (sensitivity={self.config.vad_sensitivity})...")
            
            # Record until timeout or max phrase length
            while time.time() < end_time and (time.time() - start_time) < phrase_time_limit:
                # Read a frame of audio
                try:
                    frame = recorder.read(frame_size)
                    
                    # Convert to format expected by WebRTCVAD
                    frame_bytes = struct.pack("h" * len(frame), *frame)
                    
                    # Check if frame contains speech
                    frame_is_speech = self.webrtc_vad.is_speech(frame_bytes, sample_rate)
                    
                    if frame_is_speech:
                        if not is_speech:
                            self.logger.debug("Speech detected")
                            is_speech = True
                        frames.append(frame)
                        silent_frames = 0
                    else:
                        frames.append(frame)
                        if is_speech:
                            silent_frames += 1
                            if silent_frames >= max_silent_frames:
                                self.logger.debug("End of speech detected")
                                break
                except Exception as e:
                    self.logger.error(f"Error capturing audio with WebRTCVAD: {e}")
                    break
            
            # If we detected speech, convert frames to AudioData
            if is_speech and frames:
                # Combine all frames
                all_frames = array.array('h', [])
                for frame in frames:
                    all_frames.extend(frame)
                
                # Convert to AudioData
                audio_data = sr.AudioData(
                    all_frames.tobytes(),
                    sample_rate=sample_rate,
                    sample_width=2  # 16-bit audio
                )
                return audio_data
            
            return None
    
    def _capture_with_silero_vad(self, source, timeout: int, phrase_time_limit: int) -> Optional[sr.AudioData]:
        """
        Capture voice input using Silero VAD for voice activity detection.
        
        Args:
            source: Audio source (microphone)
            timeout: Seconds to wait for speech before giving up
            phrase_time_limit: Maximum length of voice command in seconds
            
        Returns:
            AudioData object or None if no speech detected
        """
        import torch
        import numpy as np
        
        # Silero VAD works best with 16kHz audio
        sample_rate = 16000
        
        # Initialize variables for recording
        start_time = time.time()
        end_time = start_time + timeout
        
        # Record audio for processing
        with source.recorder(sample_rate=sample_rate) as recorder:
            self.logger.info(f"Listening with Silero VAD (threshold={self.config.vad_threshold})...")
            
            # Record for the maximum possible duration
            max_duration = min(timeout, phrase_time_limit)
            try:
                # Record audio
                audio_array = recorder.read(int(sample_rate * max_duration))
                audio_tensor = torch.tensor(np.array(audio_array), dtype=torch.float32)
                
                # Normalize audio
                if audio_tensor.abs().max() > 0:
                    audio_tensor = audio_tensor / audio_tensor.abs().max()
                
                # Get speech timestamps
                speech_timestamps = self.silero_get_speech_timestamps(
                    audio_tensor,
                    self.silero_model,
                    threshold=self.config.vad_threshold,
                    sampling_rate=sample_rate,
                    min_speech_duration_ms=self.config.vad_min_speech_duration_ms,
                    min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
                    window_size_samples=self.config.vad_window_size_samples,
                    return_seconds=False
                )
                
                # If speech detected, extract the audio segments
                if speech_timestamps:
                    self.logger.debug(f"Speech detected: {len(speech_timestamps)} segments")
                    
                    # Collect speech chunks
                    speech_audio = self.silero_vad_collect_chunks(
                        speech_timestamps, 
                        audio_tensor,
                        sampling_rate=sample_rate
                    )
                    
                    # Convert to numpy array then to bytes
                    speech_np = speech_audio.numpy()
                    speech_bytes = (speech_np * 32767).astype(np.int16).tobytes()
                    
                    # Create AudioData object
                    audio_data = sr.AudioData(
                        speech_bytes,
                        sample_rate=sample_rate,
                        sample_width=2  # 16-bit audio
                    )
                    return audio_data
                else:
                    self.logger.debug("No speech detected with Silero VAD")
            except Exception as e:
                self.logger.error(f"Error capturing audio with Silero VAD: {e}")
            
            return None
    
    @run_with_grace(operation_name="capture_voice", max_attempts=3)
    def capture_voice_input(self, timeout: Optional[int] = None, 
                           phrase_time_limit: Optional[int] = None) -> Optional[str]:
        """
        Capture voice input and convert to text with error recovery.
        
        Args:
            timeout: Seconds to wait for speech before giving up
            phrase_time_limit: Maximum length of voice command in seconds
            
        Returns:
            Transcribed text or None if failed
        """
        timeout = timeout or DEFAULT_PHRASE_TIMEOUT
        phrase_time_limit = phrase_time_limit or self.config.max_phrase_length
        
        with sr.Microphone(device_index=self.config.mic_index) as source:
            # Adjust for ambient noise (only for energy-based VAD)
            if self.config.vad_system == VADSystem.ENERGY:
                self._adjust_for_ambient_noise(source)
            
            try:
                audio = None
                
                # Use the appropriate VAD system
                if self.config.vad_system == VADSystem.WEBRTC:
                    self.logger.info(f"Using WebRTCVAD for voice detection")
                    audio = self._capture_with_webrtc_vad(source, timeout, phrase_time_limit)
                
                elif self.config.vad_system == VADSystem.SILERO:
                    self.logger.info(f"Using Silero VAD for voice detection")
                    audio = self._capture_with_silero_vad(source, timeout, phrase_time_limit)
                
                else:  # Default to energy-based VAD
                    self.logger.info(f"Using energy-based VAD for voice detection")
                    self.logger.info(f"Listening for voice input (timeout={timeout}s, limit={phrase_time_limit}s)...")
                    audio = self.recognizer.listen(
                        source, 
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit
                    )
                
                # If no audio was captured, return None
                if audio is None:
                    return None
                
                # Save audio for debugging if enabled
                if self.config.save_audio_chunks:
                    self._save_audio_chunk(audio)
                
                # Process audio with all configured engines in parallel
                return self._process_audio_with_engines(audio)
            except sr.WaitTimeoutError:
                self.logger.debug("No speech detected within timeout period")
            except Exception as e:
                self.logger.error(f"Voice capture failed: {str(e)}")
            
            return None
    
    def _save_audio_chunk(self, audio: sr.AudioData) -> None:
        """Save audio chunk to cache for debugging."""
        try:
            audio_data = audio.get_wav_data()
            self.audio_cache.save_audio(audio_data, "wav")
        except Exception as e:
            self.logger.error(f"Failed to save audio chunk: {e}")
    
    def _process_audio_with_engines(self, audio: sr.AudioData) -> Optional[str]:
        """
        Process audio with all configured engines in parallel.
        Returns the first successful result.
        
        Args:
            audio: Audio data to process
            
        Returns:
            Transcribed text or None if all engines failed
        """
        # Create tasks for each engine
        futures = []
        results_queue = queue.Queue()
        
        for engine in self.config.recognition_engines:
            futures.append(self.executor.submit(
                self._recognize_with_engine, audio, engine, results_queue
            ))
        
        # Wait for first successful result or all to fail
        try:
            # Wait for first result with timeout
            result = results_queue.get(timeout=10)
            
            # Cancel remaining tasks
            for future in futures:
                future.cancel()
            
            return result
        except queue.Empty:
            self.logger.warning("All recognition engines failed or timed out")
            return None
    
    def _recognize_with_engine(self, audio: sr.AudioData, 
                              engine: RecognitionEngine,
                              results_queue: queue.Queue) -> None:
        """
        Recognize audio with specific engine and put result in queue.
        
        Args:
            audio: Audio data to process
            engine: Recognition engine to use
            results_queue: Queue to put result in
        """
        try:
            self.logger.debug(f"Attempting recognition with {engine.name}")
            result = None
            
            if engine == RecognitionEngine.GOOGLE:
                result = self.recognizer.recognize_google(audio)
            
            elif engine == RecognitionEngine.SPHINX:
                result = self.recognizer.recognize_sphinx(audio)
            
            elif engine == RecognitionEngine.WHISPER:
                # Convert audio to format Whisper expects
                audio_data = audio.get_wav_data()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    f.write(audio_data)
                    f.flush()
                    # Process with Whisper
                    transcription = self.whisper_model.transcribe(f.name)
                    result = transcription["text"]
            
            elif engine == RecognitionEngine.VOSK:
                # Convert audio to format Vosk expects
                audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                rec = KaldiRecognizer(self.vosk_model, 16000)
                rec.AcceptWaveform(audio_data)
                result_json = json.loads(rec.FinalResult())
                result = result_json.get("text", "")
            
            elif engine == RecognitionEngine.AZURE:
                # Azure Speech SDK implementation
                if "azure_key" in self.config.api_keys:
                    import azure.cognitiveservices.speech as speechsdk
                    speech_config = speechsdk.SpeechConfig(
                        subscription=self.config.api_keys["azure_key"],
                        region=self.config.api_keys.get("azure_region", "westus")
                    )
                    audio_data = audio.get_wav_data()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                        f.write(audio_data)
                        f.flush()
                        audio_config = speechsdk.audio.AudioConfig(filename=f.name)
                        speech_recognizer = speechsdk.SpeechRecognizer(
                            speech_config=speech_config, 
                            audio_config=audio_config
                        )
                        result_future = speech_recognizer.recognize_once_async()
                        speech_result = result_future.get()
                        result = speech_result.text
            
            elif engine == RecognitionEngine.GOOGLE_CLOUD:
                # Google Cloud Speech-to-Text implementation
                if "google_cloud_key" in self.config.api_keys:
                    from google.cloud import speech
                    client = speech.SpeechClient()
                    audio_data = audio.get_raw_data(
                        convert_rate=16000, convert_width=2
                    )
                    audio_content = speech.RecognitionAudio(content=audio_data)
                    config = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=16000,
                        language_code="en-US",
                    )
                    response = client.recognize(config=config, audio=audio_content)
                    if response.results:
                        result = response.results[0].alternatives[0].transcript
            
            elif engine == RecognitionEngine.AMAZON:
                # Amazon Transcribe implementation
                if "aws_access_key" in self.config.api_keys:
                    import boto3
                    transcribe = boto3.client(
                        'transcribe',
                        aws_access_key_id=self.config.api_keys["aws_access_key"],
                        aws_secret_access_key=self.config.api_keys["aws_secret_key"],
                        region_name=self.config.api_keys.get("aws_region", "us-east-1")
                    )
                    # Implementation would require uploading to S3 first
                    # Simplified for example
                    pass
            
            elif engine == RecognitionEngine.OLLAMA:
                # Ollama implementation
                if "ollama_model" in self.config.api_keys:
                    import ollama
                    audio_data = audio.get_wav_data()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                        f.write(audio_data)
                        f.flush()
                        response = ollama.generate(
                            model=self.config.api_keys["ollama_model"],
                            prompt=f"Transcribe the following audio file: {f.name}",
                            system="You are a speech recognition system. Transcribe the audio accurately."
                        )
                        result = response['response']
            
            if result:
                self.logger.info(f"Recognition successful with {engine.name}: {result}")
                results_queue.put(result)
        except Exception as e:
            self.logger.warning(f"Recognition with {engine.name} failed: {str(e)}")
    
    def process_to_intent(self, text: str, audio_meta: Optional[Dict[str, Any]] = None, 
                         intent_engine=None) -> Optional[Dict[str, Any]]:
        """
        Convert recognized text to GGFAI intent format with ML-based analysis when available.
        
        Args:
            text: Recognized speech text
            audio_meta: Optional audio metadata (quality, sample rate, etc.)
            intent_engine: Optional IntentEngine instance for advanced processing
            
        Returns:
            Intent dictionary or None if parsing fails
        """
        if not text or not text.strip():
            return None
        
        # Base intent structure
        intent = {
            "text": text.strip(),
            "intent": "unknown",
            "category": "voice",
            "priority": 0.7,  # Default medium priority
            "confidence": 1.0,
            "source": "voice",
            "timestamp": time.time()
        }
        
        # Add audio metrics if available
        if audio_meta:
            intent.update({
                "audio_quality": audio_meta.get("quality", 1.0),
                "sample_rate": audio_meta.get("sample_rate", 16000),
                "sample_width": audio_meta.get("sample_width", 2)
            })
        
        # Use intent engine if available for advanced NLP processing
        if intent_engine:
            try:
                self.logger.info("Using intent engine for advanced processing")
                # Create context set for intent engine
                context = set(["voice_input", "audio_interaction"])
                
                # Process with intent engine
                processed_intent = intent_engine.process(text, context)
                
                # Merge the results, preserving audio metadata
                if processed_intent:
                    # Keep audio metadata and source information
                    audio_info = {k: v for k, v in intent.items() 
                                 if k in ["audio_quality", "sample_rate", "sample_width", "source"]}
                    
                    # Update with processed intent but preserve audio metadata
                    intent = processed_intent
                    intent.update(audio_info)
                    
                    self.logger.debug(f"Intent engine result: {intent.get('intent')}")
                    return intent
                    
            except Exception as e:
                self.logger.error(f"Intent engine processing failed: {e}")
                self.logger.info("Falling back to basic intent detection")
        
        # Fallback to simple heuristic parsing if intent engine is unavailable or fails
        text_lower = text.lower()
        
        # Media intents
        if any(word in text_lower for word in ["play", "music", "song", "listen"]):
            intent.update({
                "intent": "play_music",
                "category": "media",
                "priority": 0.9
            })
        
        # Information intents
        elif any(word in text_lower for word in ["weather", "forecast", "temperature"]):
            intent.update({
                "intent": "get_weather",
                "category": "information"
            })
        elif any(word in text_lower for word in ["news", "headlines", "events"]):
            intent.update({
                "intent": "get_news",
                "category": "information"
            })
        elif any(word in text_lower for word in ["time", "date", "day"]):
            intent.update({
                "intent": "get_time",
                "category": "information"
            })
        
        # Control intents
        elif any(word in text_lower for word in ["light", "lights", "dim", "bright"]):
            intent.update({
                "intent": "control_lights",
                "category": "home_control"
            })
        elif any(word in text_lower for word in ["temperature", "thermostat", "heat", "cool"]):
            intent.update({
                "intent": "adjust_temperature",
                "category": "home_control"
            })
        
        # Task intents
        elif any(word in text_lower for word in ["remind", "reminder", "remember"]):
            intent.update({
                "intent": "set_reminder",
                "category": "productivity"
            })
        elif any(word in text_lower for word in ["timer", "alarm", "wake"]):
            intent.update({
                "intent": "set_timer",
                "category": "productivity"
            })
        
        # System intents
        elif any(word in text_lower for word in ["stop", "cancel", "end"]):
            intent.update({
                "intent": "stop_action",
                "category": "system",
                "priority": 0.95  # High priority for stop commands
            })
        
        return intent
    
    def capture_and_process(self, intent_engine=None) -> Optional[Dict[str, Any]]:
        """
        Complete voice processing pipeline from capture to intent.
        
        Args:
            intent_engine: Optional IntentEngine instance for advanced processing
            
        Returns:
            Intent dictionary or None if failed
        """
        text = self.capture_voice_input()
        if text:
            # Create audio metadata dictionary
            audio_meta = {
                "quality": 1.0,  # Default quality
                "sample_rate": self.config.sample_rate,
                "sample_width": self.config.channels * 2  # Assuming 16-bit audio
            }
            return self.process_to_intent(text, audio_meta, intent_engine)
        return None
    
    def create_tag_from_intent(self, intent: Dict[str, Any]) -> Tag:
        """
        Create a Tag object from intent dictionary.
        
        Args:
            intent: Intent dictionary
            
        Returns:
            Tag object for tag_registry
        """
        return Tag(
            name=f"voice_intent_{int(time.time())}",
            intent=intent.get("intent", "unknown"),
            category=intent.get("category", "voice"),
            subcategory="voice_input",
            priority=intent.get("priority", 0.7),
            metadata={
                "text": intent.get("text", ""),
                "confidence": intent.get("confidence", 1.0),
                "source": "voice",
                "timestamp": intent.get("timestamp", time.time())
            }
        )

class VoiceOutputProcessor:
    """
    Production-level voice output processor with multiple TTS engines,
    caching, and performance optimizations.
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        """
        Initialize voice output processor with configuration.
        
        Args:
            config: Voice processing configuration
        """
        self.config = config or VoiceConfig()
        self.logger = logging.getLogger("GGFAI.voice.output")
        
        # Initialize audio cache
        self.audio_cache = AudioCache(self.config.audio_cache_dir)
        
        # Initialize thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_threads
        )
        
        # Initialize TTS engines
        self._init_tts_engines()
        
        self.logger.info("Voice output processor initialized")
    
    def _init_tts_engines(self) -> None:
        """Initialize TTS engines based on configuration."""
        # Initialize pyttsx3 (system TTS)
        try:
            self.pyttsx3_engine = pyttsx3.init()
            voices = self.pyttsx3_engine.getProperty('voices')
            
            # Set voice if specified
            if self.config.tts_voice != "default" and voices:
                for voice in voices:
                    if self.config.tts_voice in voice.id:
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        break
            
            # Set rate and volume
            self.pyttsx3_engine.setProperty('rate', self.config.tts_rate)
            self.pyttsx3_engine.setProperty('volume', self.config.tts_volume)
            
            self.logger.info("System TTS initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize system TTS: {e}")
            self.pyttsx3_engine = None
        
        # Initialize other TTS engines based on configuration
        self.tts_engines = {}
        
        # Check for ElevenLabs
        if self.config.tts_engine == TTSEngine.ELEVENLABS:
            if "elevenlabs_key" in self.config.api_keys:
                try:
                    import elevenlabs
                    elevenlabs.set_api_key(self.config.api_keys["elevenlabs_key"])
                    self.tts_engines[TTSEngine.ELEVENLABS] = True
                    self.logger.info("ElevenLabs TTS initialized")
                except ImportError:
                    self.logger.warning("ElevenLabs not installed")
            else:
                self.logger.warning("No API key for ElevenLabs")
        
        # Check for Coqui
        if self.config.tts_engine == TTSEngine.COQUI:
            try:
                from TTS.api import TTS
                self.coqui_tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
                self.tts_engines[TTSEngine.COQUI] = True
                self.logger.info("Coqui TTS initialized")
            except ImportError:
                self.logger.warning("Coqui TTS not installed")
        
        # Check for Kokoro
        if self.config.tts_engine == TTSEngine.KOKORO:
            try:
                from kokoro import KPipeline
                # Initialize with American English by default
                self.kokoro_pipeline = KPipeline(lang_code='a')
                self.tts_engines[TTSEngine.KOKORO] = True
                self.logger.info("Kokoro TTS initialized")
            except ImportError:
                self.logger.warning("Kokoro TTS not installed")
            except Exception as e:
                self.logger.error(f"Failed to initialize Kokoro TTS: {e}")
        
        # Check for Ollama
        if self.config.tts_engine == TTSEngine.OLLAMA:
            if "ollama_model" in self.config.api_keys:
                try:
                    import ollama
                    self.tts_engines[TTSEngine.OLLAMA] = True
                    self.logger.info("Ollama TTS initialized")
                except ImportError:
                    self.logger.warning("Ollama not installed")
            else:
                self.logger.warning("No model specified for Ollama TTS")
    
    @run_with_grace(operation_name="speak", max_attempts=2)
    def speak(self, text: str, block: bool = True) -> bool:
        """
        Convert text to speech and play audio.
        
        Args:
            text: Text to speak
            block: Whether to block until speech is complete
            
        Returns:
            True if successful, False otherwise
        """
        if not text:
            return False
        
        self.logger.info(f"Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Check cache first
        cache_key = f"{self.config.tts_engine.name}_{hash(text)}_{self.config.tts_voice}"
        cached_file = self.audio_cache.get_path(f"{cache_key}.wav")
        
        if cached_file.exists():
            self.logger.debug("Using cached audio")
            return self._play_audio_file(str(cached_file), block)
        
        # Generate speech based on configured engine
        try:
            if self.config.tts_engine == TTSEngine.SYSTEM or self.config.tts_engine == TTSEngine.PYTTSX3:
                return self._speak_with_pyttsx3(text, block)
            
            elif self.config.tts_engine == TTSEngine.ESPEAK:
                return self._speak_with_espeak(text, block)
            
            elif self.config.tts_engine == TTSEngine.ELEVENLABS:
                return self._speak_with_elevenlabs(text, block)
            
            elif self.config.tts_engine == TTSEngine.COQUI:
                return self._speak_with_coqui(text, block)
            
            elif self.config.tts_engine == TTSEngine.KOKORO:
                return self._speak_with_kokoro(text, block)
            
            elif self.config.tts_engine == TTSEngine.OLLAMA:
                return self._speak_with_ollama(text, block)
            
            else:
                # Fallback to system TTS
                self.logger.warning(f"Unsupported TTS engine: {self.config.tts_engine.name}, falling back to system TTS")
                return self._speak_with_pyttsx3(text, block)
        
        except Exception as e:
            self.logger.error(f"Speech generation failed: {e}")
            # Try fallback if primary engine fails
            try:
                return self._speak_with_pyttsx3(text, block)
            except Exception as e2:
                self.logger.error(f"Fallback speech generation failed: {e2}")
                return False
    
    def _speak_with_pyttsx3(self, text: str, block: bool) -> bool:
        """Speak using pyttsx3 (system TTS)."""
        if not self.pyttsx3_engine:
            self.logger.error("System TTS not initialized")
            return False
        
        try:
            if block:
                self.pyttsx3_engine.say(text)
                self.pyttsx3_engine.runAndWait()
            else:
                # Run in separate thread
                def speak_thread():
                    engine = pyttsx3.init()
                    engine.setProperty('rate', self.config.tts_rate)
                    engine.setProperty('volume', self.config.tts_volume)
                    engine.say(text)
                    engine.runAndWait()
                
                threading.Thread(target=speak_thread, daemon=True).start()
            
            return True
        except Exception as e:
            self.logger.error(f"pyttsx3 speech failed: {e}")
            return False
    
    def _speak_with_espeak(self, text: str, block: bool) -> bool:
        """Speak using espeak command-line tool."""
        try:
            cmd = ["espeak", f"-a {int(self.config.tts_volume * 100)}", 
                  f"-s {self.config.tts_rate}", f"-v {self.config.tts_voice}", text]
            
            if block:
                subprocess.run(cmd, check=True)
            else:
                subprocess.Popen(cmd)
            
            return True
        except Exception as e:
            self.logger.error(f"espeak speech failed: {e}")
            return False
    
    def _speak_with_elevenlabs(self, text: str, block: bool) -> bool:
        """Speak using ElevenLabs API."""
        if TTSEngine.ELEVENLABS not in self.tts_engines:
            self.logger.error("ElevenLabs not initialized")
            return False
        
        try:
            import elevenlabs
            
            # Generate audio
            audio = elevenlabs.generate(
                text=text,
                voice=self.config.tts_voice,
                model="eleven_monolingual_v1"
            )
            
            # Save to cache
            cache_key = f"elevenlabs_{hash(text)}_{self.config.tts_voice}"
            cache_path = self.audio_cache.get_path(f"{cache_key}.mp3")
            
            with open(cache_path, "wb") as f:
                f.write(audio)
            
            # Play audio
            return self._play_audio_file(str(cache_path), block)
        
        except Exception as e:
            self.logger.error(f"ElevenLabs speech failed: {e}")
            return False
    
    def _speak_with_coqui(self, text: str, block: bool) -> bool:
        """Speak using Coqui TTS."""
        if not hasattr(self, 'coqui_tts'):
            self.logger.error("Coqui TTS not initialized")
            return False
        
        try:
            # Generate audio
            cache_key = f"coqui_{hash(text)}_{self.config.tts_voice}"
            cache_path = self.audio_cache.get_path(f"{cache_key}.wav")
            
            self.coqui_tts.tts_to_file(
                text=text,
                file_path=str(cache_path)
            )
            
            # Play audio
            return self._play_audio_file(str(cache_path), block)
        
        except Exception as e:
            self.logger.error(f"Coqui speech failed: {e}")
            return False
    
    def _speak_with_kokoro(self, text: str, block: bool) -> bool:
        """Speak using Kokoro TTS."""
        if TTSEngine.KOKORO not in self.tts_engines:
            self.logger.error("Kokoro TTS not initialized")
            return False
        
        try:
            # Generate unique cache key for this text and voice combo
            cache_key = f"kokoro_{hash(text)}_{self.config.tts_voice}"
            cache_path = self.audio_cache.get_path(f"{cache_key}.wav")

            if not cache_path.exists():
                # Initialize Kokoro if not already done
                if not hasattr(self, 'kokoro_pipeline'):
                    import torch
                    from kokoro import KPipeline
                    # Detect device capability
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    compute_type = 'float16' if device == 'cuda' else 'float32'
                    self.logger.info(f"Initializing Kokoro TTS (Device: {device}, Compute: {compute_type})")
                    # Use American English by default
                    self.kokoro_pipeline = KPipeline(lang_code='a', device=device)

                # Use the configured voice or default to af_heart
                voice = self.config.tts_voice if self.config.tts_voice != "default" else "af_heart"
                if not voice.startswith(('a', 'b')):  # Validate voice prefix (a=American, b=British)
                    self.logger.warning(f"Invalid voice prefix '{voice}', defaulting to af_heart")
                    voice = "af_heart"

                # Convert TTS rate to speed multiplier (normalize around 175 wpm)
                speed = self.config.tts_rate / 175.0  
                
                try:
                    # Generate audio with proper error handling
                    audio_generated = False
                    for result in self.kokoro_pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+'):
                        if result.audio is not None:
                            import scipy.io.wavfile as wavfile
                            # Normalize and convert float audio to 16-bit int
                            audio_np = result.audio.numpy()
                            max_val = np.max(np.abs(audio_np))
                            if max_val > 0:  # Avoid division by zero
                                audio_np = audio_np / max_val
                            audio_int = (audio_np * 32767).astype('int16')
                            wavfile.write(str(cache_path), 24000, audio_int)
                            audio_generated = True
                            break
                    
                    if not audio_generated:
                        raise RuntimeError("No audio data generated")

                except Exception as gen_e:
                    self.logger.error(f"Audio generation failed: {gen_e}")
                    if hasattr(self, 'kokoro_pipeline'):
                        del self.kokoro_pipeline  # Force reinitialization on next attempt
                    return False

            # Play the generated audio
            return self._play_audio_file(str(cache_path), block)

        except Exception as e:
            self.logger.error(f"Kokoro speech failed: {e}")
            if hasattr(self, 'kokoro_pipeline'):
                del self.kokoro_pipeline  # Force reinitialization on next attempt
            return False
    
    def _speak_with_ollama(self, text: str, block: bool) -> bool:
        """Speak using Ollama TTS."""
        if TTSEngine.OLLAMA not in self.tts_engines:
            self.logger.error("Ollama not initialized")
            return False
        
        try:
            import ollama
            
            # Generate audio description using Ollama
            response = ollama.generate(
                model=self.config.api_keys["ollama_model"],
                prompt=f"Convert this text to speech: {text}",
                system="You are a text-to-speech system. Generate audio for the given text."
            )
            
            # This is a placeholder - actual implementation would depend on
            # how Ollama handles audio generation
            self.logger.warning("Ollama TTS not fully implemented, falling back to system TTS")
            return self._speak_with_pyttsx3(text, block)
        
        except Exception as e:
            self.logger.error(f"Ollama speech failed: {e}")
            return False
    
    def _play_audio_file(self, file_path: str, block: bool) -> bool:
        """
        Play audio file using appropriate player.
        
        Args:
            file_path: Path to audio file
            block: Whether to block until playback is complete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if PYDUB_AVAILABLE:
                # Use pydub for playback
                audio = AudioSegment.from_file(file_path)
                
                if block:
                    pydub_play(audio)
                else:
                    threading.Thread(
                        target=pydub_play, 
                        args=(audio,), 
                        daemon=True
                    ).start()
            else:
                # Fallback to system player
                if os.name == 'nt':  # Windows
                    player = 'start'
                    shell = True
                elif os.name == 'posix':  # Linux/Mac
                    player = 'aplay' if file_path.endswith('.wav') else 'mpg123'
                    shell = False
                
                if block:
                    subprocess.run([player, file_path], shell=shell)
                else:
                    subprocess.Popen([player, file_path], shell=shell)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
            return False

class VoiceProcessor:
    """
    Unified voice processing interface combining input and output capabilities.
    Provides a simple API for voice interaction with the GGFAI Framework.
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        """
        Initialize voice processor with configuration.
        
        Args:
            config: Voice processing configuration
        """
        self.config = config or VoiceConfig()
        self.logger = logging.getLogger("GGFAI.voice")
        
        # Initialize input and output processors
        self.input_processor = VoiceInputProcessor(self.config)
        self.output_processor = VoiceOutputProcessor(self.config)
        
        # Initialize state
        self.is_listening = False
        self.is_speaking = False
        self._lock = threading.RLock()
        
        self.logger.info("Voice processor initialized")
    
    def listen(self, timeout: Optional[int] = None, 
              phrase_time_limit: Optional[int] = None) -> Optional[str]:
        """
        Listen for voice input and return transcribed text.
        
        Args:
            timeout: Seconds to wait for speech before giving up
            phrase_time_limit: Maximum length of voice command in seconds
            
        Returns:
            Transcribed text or None if failed
        """
        with self._lock:
            self.is_listening = True
        
        try:
            result = self.input_processor.capture_voice_input(
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
        finally:
            with self._lock:
                self.is_listening = False
        
        return result
    
    def speak(self, text: str, block: bool = True) -> bool:
        """
        Convert text to speech and play audio.
        
        Args:
            text: Text to speak
            block: Whether to block until speech is complete
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            self.is_speaking = True
        
        try:
            result = self.output_processor.speak(text, block)
            if not block:
                # If non-blocking, we need to manually set is_speaking to False
                # after a reasonable delay
                def reset_speaking():
                    time.sleep(len(text) * 0.1)  # Rough estimate of speech duration
                    with self._lock:
                        self.is_speaking = False
                
                threading.Thread(target=reset_speaking, daemon=True).start()
            
            return result
        finally:
            if block:
                with self._lock:
                    self.is_speaking = False
    
    def process_voice_input(self, intent_engine=None) -> Optional[Dict[str, Any]]:
        """
        Process voice input and return intent dictionary.
        
        Args:
            intent_engine: Optional IntentEngine instance for advanced processing
            
        Returns:
            Intent dictionary or None if failed
        """
        return self.input_processor.capture_and_process(intent_engine)
    
    def process_to_tag(self, intent_engine=None) -> Optional[Tag]:
        """
        Process voice input and return Tag object.
        
        Args:
            intent_engine: Optional IntentEngine instance for advanced processing
            
        Returns:
            Tag object or None if failed
        """
        intent = self.process_voice_input(intent_engine)
        if intent:
            return self.input_processor.create_tag_from_intent(intent)
        return None
    
    def conversation(self, greeting: str = "How can I help you?", 
                     intent_engine=None) -> Optional[Dict[str, Any]]:
        """
        Conduct a voice conversation with greeting and response.
        
        Args:
            greeting: Initial greeting to speak
            intent_engine: Optional IntentEngine instance for advanced processing
            
        Returns:
            Intent dictionary from user response or None if failed
        """
        # Speak greeting
        self.speak(greeting)
        
        # Listen for response and process with intent engine if available
        return self.process_voice_input(intent_engine)
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available TTS voices.
        
        Returns:
            List of voice dictionaries with id, name, and gender
        """
        voices = []
        
        # Get system voices
        try:
            engine = pyttsx3.init()
            for voice in engine.getProperty('voices'):
                voices.append({
                    "id": voice.id,
                    "name": voice.name,
                    "gender": getattr(voice, 'gender', 'unknown'),
                    "engine": "system"
                })
        except Exception as e:
            self.logger.error(f"Failed to get system voices: {e}")
        
        # Add other engine voices if available
        if TTSEngine.ELEVENLABS in self.output_processor.tts_engines:
            try:
                import elevenlabs
                for voice in elevenlabs.voices():
                    voices.append({
                        "id": voice.voice_id,
                        "name": voice.name,
                        "gender": getattr(voice, 'gender', 'unknown'),
                        "engine": "elevenlabs"
                    })
            except Exception as e:
                self.logger.error(f"Failed to get ElevenLabs voices: {e}")
        
        return voices
    
    def get_available_recognition_engines(self) -> List[str]:
        """
        Get list of available speech recognition engines.
        
        Returns:
            List of engine names
        """
        return [engine.name for engine in self.config.recognition_engines]
    
    def get_available_tts_engines(self) -> List[str]:
        """
        Get list of available TTS engines.
        
        Returns:
            List of engine names
        """
        available = ["SYSTEM", "PYTTSX3"]
        
        # Check for other engines
        for engine in TTSEngine:
            if engine in self.output_processor.tts_engines:
                available.append(engine.name)
        
        return available
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update voice processor configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Convert to VoiceConfig if needed
        if not isinstance(config, VoiceConfig):
            config = VoiceConfig.from_dict(config)
        
        # Update config
        self.config = config
        
        # Reinitialize processors with new config
        self.input_processor = VoiceInputProcessor(self.config)
        self.output_processor = VoiceOutputProcessor(self.config)
        
        self.logger.info("Voice processor configuration updated")
    
    def save_config(self, path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        self.config.save(path)
        self.logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load_config(cls, path: str) -> 'VoiceProcessor':
        """
        Load configuration from file and create processor.
        
        Args:
            path: Path to load configuration from
            
        Returns:
            VoiceProcessor with loaded configuration
        """
        config = VoiceConfig.load(path)
        return cls(config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = VoiceProcessor()
    
    print("Voice Processor Demo")
    print("-------------------")
    print("1. Speaking test")
    processor.speak("Hello, I am the GGFAI voice processor. I can speak and listen.")
    
    print("\n2. Listening test")
    print("Speak now (waiting for command)...")
    intent = processor.process_voice_input()
    
    if intent:
        print("Captured intent:")
        print(intent)
        
        print("\n3. Conversation test")
        processor.speak(f"I understood that you want to {intent.get('intent', 'do something')}. Is there anything else?")
    else:
        print("No valid intent captured")