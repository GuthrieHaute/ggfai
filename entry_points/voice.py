# voice.py - Voice input processing module for GGFAI
# written by DeepSeek Chat (honor call: The Voice Architect)

import time
import logging
from typing import Optional, Dict, Any
import speech_recognition as sr

class VoiceInputProcessor:
    """
    Processes voice input using SpeechRecognition library and converts to intent format.
    Handles microphone selection, ambient noise adjustment, and error recovery.
    """
    
    def __init__(self, energy_threshold: int = 300, pause_threshold: float = 0.8,
                 dynamic_energy_threshold: bool = True, mic_index: Optional[int] = None):
        """
        Initialize voice processor with audio capture parameters.
        
        Args:
            energy_threshold: Minimum audio energy to consider as speech
            pause_threshold: Seconds of silence to mark end of phrase
            dynamic_energy_threshold: Adjust threshold based on ambient noise
            mic_index: Optional specific microphone device index
        """
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold
        self.mic_index = mic_index
        self.logger = logging.getLogger("GGFAI.voice")
        
        # Test microphone availability
        self._verify_microphone()

    def _verify_microphone(self):
        """Check if specified microphone is available."""
        try:
            mics = sr.Microphone.list_microphone_names()
            if self.mic_index is not None and self.mic_index >= len(mics):
                self.logger.warning(f"Mic index {self.mic_index} not found. Using default.")
                self.mic_index = None
        except Exception as e:
            self.logger.error(f"Microphone check failed: {str(e)}")

    def _adjust_for_ambient_noise(self, source):
        """Calibrate recognizer for current noise conditions."""
        try:
            self.logger.info("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.logger.info(f"Energy threshold set to {self.recognizer.energy_threshold}")
        except Exception as e:
            self.logger.warning(f"Ambient noise adjustment failed: {str(e)}")

    def capture_voice_input(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """
        Capture voice input and convert to text.
        
        Args:
            timeout: Seconds to wait for speech before giving up
            phrase_time_limit: Maximum length of voice command in seconds
            
        Returns:
            Transcribed text or None if failed
        """
        with sr.Microphone(device_index=self.mic_index) as source:
            self._adjust_for_ambient_noise(source)
            
            try:
                self.logger.info("Listening for voice input...")
                audio = self.recognizer.listen(source, timeout=timeout, 
                                             phrase_time_limit=phrase_time_limit)
                return self._recognize_audio(audio)
            except sr.WaitTimeoutError:
                self.logger.debug("No speech detected within timeout period")
            except Exception as e:
                self.logger.error(f"Voice capture failed: {str(e)}")
            return None

    def _recognize_audio(self, audio) -> Optional[str]:
        """Attempt multiple recognition backends with fallback."""
        try:
            # Primary recognition with Google Web Speech API
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            self.logger.warning("Speech recognition could not understand audio")
        except sr.RequestError as e:
            self.logger.error(f"Could not request results from service: {str(e)}")
            # Fallback to Sphinx if offline
            try:
                text = self.recognizer.recognize_sphinx(audio)
                self.logger.info(f"Offline recognition: {text}")
                return text
            except Exception as e:
                self.logger.error(f"Offline recognition failed: {str(e)}")
        return None

    def process_to_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Convert recognized text to GGFAI intent format.
        
        Args:
            text: Recognized speech text
            
        Returns:
            Intent dictionary or None if parsing fails
        """
        if not text or not text.strip():
            return None
            
        # Basic intent parsing (will be enhanced by intent_engine.py)
        intent = {
            "text": text.strip(),
            "intent": "unknown",
            "category": "voice",
            "priority": 0.7,  # Default medium priority
            "confidence": 1.0,
            "source": "voice",
            "timestamp": time.time()
        }
        
        # Simple heuristic parsing (to be replaced with proper NLP)
        text_lower = text.lower()
        if "play" in text_lower and ("music" in text_lower or "song" in text_lower):
            intent.update({
                "intent": "play_music",
                "category": "media",
                "priority": 0.9
            })
        elif "weather" in text_lower:
            intent.update({
                "intent": "get_weather",
                "category": "information"
            })
            
        return intent

    def capture_and_process(self) -> Optional[Dict[str, Any]]:
        """Complete voice processing pipeline from capture to intent."""
        text = self.capture_voice_input()
        if text:
            return self.process_to_intent(text)
        return None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = VoiceInputProcessor()
    
    print("Speak now (waiting for command)...")
    intent = processor.capture_and_process()
    
    if intent:
        print("Captured intent:")
        print(intent)
    else:
        print("No valid intent captured")