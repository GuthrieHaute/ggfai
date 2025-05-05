"""
LLM Coordinator - Manages LLM instances across the GGFAI framework

This module coordinates LLM instances, preventing resource conflicts and ensuring consistent behavior.
It provides a centralized registry for LLM instances and handles coordination between different components.
"""

import time
import logging
import threading
import random
from typing import Dict, Optional, Any, Set, List, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger("GGFAI.llm_coordinator")

class LLMStatus(Enum):
    """Status of an LLM instance."""
    IDLE = auto()       # Available for use
    BUSY = auto()       # Currently processing a request
    LOADING = auto()    # Being loaded into memory
    UNLOADING = auto()  # Being unloaded from memory
    ERROR = auto()      # In error state

@dataclass
class LLMInstance:
    """Information about an LLM instance."""
    model_id: str
    status: LLMStatus = LLMStatus.IDLE
    owner: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    error_count: int = 0
    instance: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = time.time()
        self.usage_count += 1

class LLMCoordinator:
    """
    Coordinates LLM instances across the GGFAI framework.
    
    Ensures proper resource management and consistent behavior of LLM instances.
    Uses dependency injection for better testability and configuration.
    """
    
    def __init__(self, cleanup_interval: float = 300.0, personality_config: Optional[Dict] = None):
        """Initialize the LLM coordinator.
        
        Args:
            cleanup_interval: Time between cleanup runs in seconds
            personality_config: Optional configuration for personality traits
        """
        self._instances: Dict[str, LLMInstance] = {}
        self._lock = threading.RLock()
        self._waiting: Dict[str, List[Tuple[str, float]]] = {}  # model_id -> [(requester_id, timestamp)]
        self._shutdown_flag = False
        self._cleanup_interval = cleanup_interval
        
        self.conversation_state = {}
        self.personality_base = personality_config or {
            "warmth": 0.7,
            "empathy": 0.8,
            "formality": 0.5,
            "adaptability": 0.9
        }
        
        # Start the cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("LLM Coordinator initialized")

    def _cleanup_loop(self):
        """Periodically clean up unused LLM instances."""
        while not self._shutdown_flag:
            time.sleep(self._cleanup_interval)
            try:
                self._cleanup_instances()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    def _cleanup_instances(self):
        """Clean up unused LLM instances."""
        with self._lock:
            current_time = time.time()
            to_remove = []
            
            for model_id, instance in self._instances.items():
                if (current_time - instance.last_used > self._cleanup_interval and 
                    instance.status == LLMStatus.IDLE):
                    to_remove.append(model_id)
                    
            for model_id in to_remove:
                self._unload_instance(model_id)

    def _unload_instance(self, model_id: str):
        """Unload an LLM instance."""
        instance = self._instances.get(model_id)
        if instance:
            try:
                instance.status = LLMStatus.UNLOADING
                # Add any cleanup/unloading logic here
                del self._instances[model_id]
                logger.info(f"Unloaded LLM instance {model_id}")
            except Exception as e:
                logger.error(f"Error unloading instance {model_id}: {e}")
                instance.status = LLMStatus.ERROR

    async def process_interaction(self, user_input: str, context: dict = None) -> dict:
        """Process user interaction with enhanced natural dialogue capabilities"""
        # Initialize or retrieve conversation state
        session_id = context.get("session_id", "default")
        if session_id not in self.conversation_state:
            self.conversation_state[session_id] = {
                "interaction_count": 0,
                "emotional_trajectory": [],
                "topic_chain": [],
                "user_preferences": {},
                "style_history": [],
                "rapport_level": 0.5
            }
            
        state = self.conversation_state[session_id]
        state["interaction_count"] += 1
        
        # Deep context analysis
        emotional_context = await self._analyze_emotional_context(user_input, state)
        conversation_style = self._detect_conversation_style(user_input, state)
        relationship_context = self._analyze_relationship_context(state)
        
        # Adapt personality dynamically
        adapted_traits = self._adapt_personality(
            emotional_context,
            conversation_style,
            relationship_context
        )
        
        # Generate natural response with context
        response = await self._generate_natural_response(
            user_input,
            emotional_context,
            adapted_traits,
            state
        )
        
        # Update conversation state
        self._update_state(state, {
            "user_input": user_input,
            "response": response,
            "emotional_context": emotional_context,
            "conversation_style": conversation_style
        })
        
        return {
            "response": response,
            "emotional_context": emotional_context,
            "conversation_style": conversation_style,
            "personality_traits": adapted_traits,
            "rapport_level": state["rapport_level"]
        }
        
    async def _analyze_emotional_context(self, text: str, state: dict) -> dict:
        """Analyze emotional context with history awareness"""
        current_emotion = {
            "valence": 0.0,  # -1.0 to 1.0
            "arousal": 0.0,  # 0.0 to 1.0
            "dominance": 0.0,  # 0.0 to 1.0
            "primary_emotion": "neutral",
            "secondary_emotions": [],
            "intensity": 0.0,
            "needs_acknowledgment": False
        }
        
        # Consider emotional trajectory
        if state["emotional_trajectory"]:
            prev_emotions = state["emotional_trajectory"][-3:]
            emotional_shift = self._detect_emotional_shift(current_emotion, prev_emotions)
            current_emotion["needs_acknowledgment"] = emotional_shift > 0.3
            
        return current_emotion
        
    def _detect_conversation_style(self, text: str, state: dict) -> dict:
        """Detect and analyze conversation style"""
        style = {
            "formality": 0.5,
            "directness": 0.5,
            "engagement": 0.5,
            "emotional_expressiveness": 0.5,
            "turn_length": "medium",
            "vocabulary_level": "neutral"
        }
        
        # Track style consistency
        if state["style_history"]:
            prev_styles = state["style_history"][-3:]
            style["consistency"] = self._measure_style_consistency(style, prev_styles)
            
        return style
        
    def _analyze_relationship_context(self, state: dict) -> dict:
        """Analyze the development of the conversation relationship"""
        return {
            "rapport_level": state["rapport_level"],
            "interaction_depth": min(1.0, state["interaction_count"] / 20),
            "trust_indicators": self._analyze_trust_indicators(state),
            "engagement_level": self._measure_engagement(state)
        }
        
    def _adapt_personality(
        self,
        emotional_context: dict,
        conversation_style: dict,
        relationship_context: dict
    ) -> dict:
        """Adapt personality traits based on conversation context"""
        adapted = self.personality_base.copy()
        
        # Adjust warmth based on emotional needs
        if emotional_context["needs_acknowledgment"]:
            adapted["warmth"] = min(1.0, adapted["warmth"] * 1.2)
            
        # Match formality to user's style
        adapted["formality"] = (
            adapted["formality"] * 0.3 +
            conversation_style["formality"] * 0.7
        )
        
        # Increase empathy for negative emotions
        if emotional_context["valence"] < 0:
            adapted["empathy"] = min(1.0, adapted["empathy"] * 1.3)
            
        return adapted
        
    def _detect_emotional_shift(self, current: dict, previous: list) -> float:
        """Detect significant shifts in emotional state"""
        if not previous:
            return 0.0
            
        # Calculate emotional distance
        last_emotion = previous[-1]
        dimensions = ["valence", "arousal", "dominance"]
        
        distances = [
            abs(current[dim] - last_emotion[dim])
            for dim in dimensions
        ]
        
        return sum(distances) / len(dimensions)
        
    def _measure_style_consistency(self, current: dict, previous: list) -> float:
        """Measure consistency in conversation style"""
        if not previous:
            return 1.0
            
        features = ["formality", "directness", "engagement"]
        consistency_scores = []
        
        for prev_style in previous:
            distances = [
                abs(current[feature] - prev_style[feature])
                for feature in features
            ]
            consistency_scores.append(1.0 - (sum(distances) / len(features)))
            
        return sum(consistency_scores) / len(consistency_scores)
        
    def _analyze_trust_indicators(self, state: dict) -> dict:
        """Analyze indicators of trust in the conversation"""
        return {
            "self_disclosure": self._measure_self_disclosure(state),
            "engagement_consistency": self._measure_engagement_consistency(state),
            "emotional_openness": self._measure_emotional_openness(state)
        }
        
    def _measure_engagement(self, state: dict) -> float:
        """Measure user engagement level"""
        if not state["topic_chain"]:
            return 0.5
            
        recent_interactions = state["topic_chain"][-5:]
        
        # Consider factors like response length, question-asking, topic development
        engagement_signals = [
            len(interaction["user_input"].split())
            for interaction in recent_interactions
        ]
        
        return min(1.0, sum(engagement_signals) / (len(engagement_signals) * 20))
        
    async def _generate_natural_response(
        self,
        user_input: str,
        emotional_context: dict,
        personality: dict,
        state: dict
    ) -> str:
        """Generate natural, contextually appropriate responses"""
        # Select response style based on context
        style_params = self._select_response_style(emotional_context, personality, state)
        
        # Generate response candidates
        candidates = await self._generate_response_candidates(
            user_input,
            style_params,
            max_candidates=3
        )
        
        # Score and select best response
        scored_candidates = [
            (
                candidate,
                self._score_response_naturalness(candidate, style_params),
                self._score_response_appropriateness(candidate, emotional_context)
            )
            for candidate in candidates
        ]
        
        best_response = max(scored_candidates, key=lambda x: (x[1] + x[2]) / 2)[0]
        
        # Add natural language markers
        return self._add_natural_markers(best_response, personality)
        
    def _update_state(self, state: dict, interaction: dict):
        """Update conversation state with new interaction"""
        state["emotional_trajectory"].append(interaction["emotional_context"])
        if len(state["emotional_trajectory"]) > 10:
            state["emotional_trajectory"].pop(0)
            
        state["topic_chain"].append({
            "user_input": interaction["user_input"],
            "response": interaction["response"],
            "timestamp": time.time()
        })
        
        state["style_history"].append(interaction["conversation_style"])
        if len(state["style_history"]) > 5:
            state["style_history"].pop(0)
            
        # Update rapport based on interaction quality
        state["rapport_level"] = min(1.0, state["rapport_level"] + self._calculate_rapport_change(interaction))
        
    def _calculate_rapport_change(self, interaction: dict) -> float:
        """Calculate change in rapport based on interaction quality"""
        base_change = 0.02  # Small positive change for continued interaction
        
        # Adjust based on emotional alignment
        if interaction["emotional_context"]["needs_acknowledgment"]:
            base_change *= 2  # Larger improvement for emotional support
            
        # Adjust based on style matching
        style_match = interaction["conversation_style"].get("consistency", 0.5)
        base_change *= (1 + (style_match - 0.5))
        
        return base_change
        
    def _add_natural_markers(self, response: str, personality: dict) -> str:
        """Add natural language markers based on personality"""
        markers = {
            "high_warmth": [
                "I hear you",
                "I understand",
                "That makes sense",
                "You know what"
            ],
            "high_empathy": [
                "I can see why",
                "That must be",
                "I appreciate",
                "It sounds like"
            ],
            "thoughtful": [
                "Let me think",
                "Well",
                "Hmm",
                "You know"
            ]
        }
        
        # Select markers based on personality
        selected_markers = []
        if personality["warmth"] > 0.6:
            selected_markers.extend(markers["high_warmth"])
        if personality["empathy"] > 0.6:
            selected_markers.extend(markers["high_empathy"])
            
        # Add marker if appropriate
        if selected_markers and random.random() < 0.3:
            marker = random.choice(selected_markers)
            response = f"{marker}, {response}"
            
        return response
    
    def shutdown(self) -> None:
        """Shut down the LLM coordinator."""
        self._shutdown_flag = True
        
        # Wait for the cleanup thread to finish
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)
        
        logger.info("LLM Coordinator shut down")