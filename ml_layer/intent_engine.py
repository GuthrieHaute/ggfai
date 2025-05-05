"""
Enhanced Intent Engine with Natural Language Understanding and Human-like Interaction

Implements sophisticated natural language understanding with:
- Deep semantic understanding using transformer models
- Emotional context recognition and adaptation
- Conversation history awareness and topic tracking
- Natural dialogue flow with configurable personality
- Multi-modal fusion support
- Active learning capabilities
"""

import logging
from typing import Dict, Set, List, Optional, Any, Tuple
from pathlib import Path
import spacy
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)
import json
import random
import time
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.tag_registry import TagRegistry
from core.run_with_grace import run_with_grace
from trackers.context_tracker import ContextTracker
from trackers.intent_tracker import IntentTracker
from trackers.analytics_tracker import AnalyticsTracker, EventSeverity
from ml_layer.model_adapter import ModelAdapter

logger = logging.getLogger("GGFAI.intent_engine")

class IntentConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

@dataclass
class IntentResult:
    intent_type: str
    category: str
    confidence: float
    requires_clarification: bool
    entities: Dict[str, List[str]]
    sentiment: Dict[str, float]
    emotion: Dict[str, float]
    context_factors: Dict[str, Any]
    source: str
    metadata: Dict[str, Any]

class IntentEngine:
    def __init__(self, llm_coordinator, component_id: str):
        """Initialize the intent engine with all required components"""
        self.nlp = spacy.load("en_core_web_trf")  # Use transformer pipeline for better accuracy
        
        # Core components
        self.tag_registry = TagRegistry()
        self.context_tracker = ContextTracker()
        self.intent_tracker = IntentTracker()
        self.analytics_tracker = AnalyticsTracker()
        self._llm_coordinator = llm_coordinator
        self._component_id = component_id
        
        # Initialize ML components
        self._init_ml_models()
        
        # Conversation state
        self.conversation_history = []
        self.topic_graph = {}  # Track topic relationships
        self.entity_memory = {}  # Track mentioned entities
        
        # Configure personality
        self.personality = {
            "warmth": 0.7,  # Higher warmth for friendlier interactions
            "empathy": 0.8,  # Strong empathy for better emotional understanding
            "formality": 0.5,  # Balanced formality
            "adaptability": 0.9,  # High adaptability to match user's style
            "assertiveness": 0.4  # Lower assertiveness for collaborative tone
        }
        
        # Performance optimization
        self._cache = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Load configurations
        self._load_conversation_data()
        self._load_ml_configs()

    def _init_ml_models(self):
        """Initialize ML models for various analysis tasks"""
        try:
            # Intent classification model
            self.intent_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )

            # Named entity recognition
            self.ner_model = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=0 if torch.cuda.is_available() else -1
            )

            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="siebert/sentiment-roberta-large-english",
                device=0 if torch.cuda.is_available() else -1
            )

            # Emotion detection
            self.emotion_detector = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info("Successfully initialized all ML models")
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            # Fall back to simpler models or rule-based approaches
            self._init_fallback_models()

    def _init_fallback_models(self):
        """Initialize fallback models when advanced models fail"""
        logger.info("Initializing fallback models")
        self.nlp = spacy.load("en_core_web_sm")  # Lighter spaCy model
        
        # Basic rule-based intent matching
        self.basic_patterns = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "farewell": ["goodbye", "bye", "see you", "farewell"],
            "question": ["what", "when", "where", "why", "how"],
            "request": ["please", "could you", "would you", "can you"],
            "affirmative": ["yes", "yeah", "correct", "right"],
            "negative": ["no", "nope", "incorrect", "wrong"]
        }

    @run_with_grace(operation_name="process_intent")
    async def process(self, text: str, context: Optional[Set[str]] = None) -> IntentResult:
        """Process natural language input to understand intent with contextual awareness"""
        
        # Parse input text with spaCy
        doc = self.nlp(text)
        
        # Parallel processing of different analysis tasks
        analysis_tasks = [
            self._executor.submit(self._extract_entities, doc),
            self._executor.submit(self._analyze_sentiment_and_emotion, text),
            self._executor.submit(self._classify_intent, text, doc),
            self._executor.submit(self._analyze_conversation_history, text),
            self._executor.submit(self._analyze_context, context or set())
        ]
        
        # Gather results
        entities, sentiment_emotion, intent_data, historical_context, context_factors = [
            task.result() for task in analysis_tasks
        ]
        
        # Merge sentiment and emotion data
        sentiment, emotion = sentiment_emotion
        
        # Determine if LLM analysis is needed
        if self._needs_llm_analysis(intent_data, entities, emotion):
            llm_intent = await self._get_llm_intent(text, context_factors)
            if llm_intent:
                self._merge_llm_insights(intent_data, llm_intent)
        
        # Update conversation history and topic graph
        self._update_conversation_history(text, sentiment, emotion)
        self._update_topic_graph(doc, intent_data["intent_type"])
        
        # Create final intent result
        result = IntentResult(
            intent_type=intent_data["intent_type"],
            category=intent_data["category"],
            confidence=intent_data["confidence"],
            requires_clarification=intent_data["requires_clarification"],
            entities=entities,
            sentiment=sentiment,
            emotion=emotion,
            context_factors=context_factors,
            source=self._component_id,
            metadata={
                "historical_context": historical_context,
                "topic_shifts": self._detect_topic_shifts(),
                "style_consistency": self._analyze_style_consistency(text),
                "conversation_depth": len(self.conversation_history)
            }
        )
        
        # Track the intent
        self.intent_tracker.track_intent(result)
        
        # Log analytics
        self._log_intent_analytics(result)
        
        return result

    async def process_multimodal(
        self,
        inputs: Dict[str, Dict[str, Any]],
        context: Optional[Set[str]] = None
    ) -> IntentResult:
        """Process multiple input modalities together for unified understanding"""
        
        # Process each modality separately first
        modality_results = {}
        for modality, input_data in inputs.items():
            if modality == "text":
                result = await self.process(input_data["text"], context)
            elif modality == "voice":
                result = await self._process_voice_input(input_data)
            elif modality == "vision":
                result = await self._process_vision_input(input_data)
            else:
                logger.warning(f"Unknown modality: {modality}")
                continue
                
            modality_results[modality] = result
        
        # Fuse modality results
        fused_result = self._fuse_modality_results(modality_results)
        
        # Update conversation state with multimodal context
        self._update_multimodal_context(fused_result, modality_results)
        
        return fused_result

    def _fuse_modality_results(
        self,
        modality_results: Dict[str, IntentResult]
    ) -> IntentResult:
        """Fuse results from multiple modalities using weighted combination"""
        
        # Base weights for each modality
        modality_weights = {
            "text": 0.4,
            "voice": 0.3,
            "vision": 0.3
        }
        
        # Adjust weights based on confidence
        total_weight = 0.0
        adjusted_weights = {}
        for modality, result in modality_results.items():
            weight = modality_weights.get(modality, 0.0) * result.confidence
            adjusted_weights[modality] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            adjusted_weights = {
                k: v/total_weight for k, v in adjusted_weights.items()
            }
        
        # Combine intents and categories
        combined_intent = self._combine_weighted_values(
            [(r.intent_type, w) for m, (r, w) in zip(modality_results.items(), adjusted_weights.items())]
        )
        
        combined_category = self._combine_weighted_values(
            [(r.category, w) for m, (r, w) in zip(modality_results.items(), adjusted_weights.items())]
        )
        
        # Merge entities from all modalities
        merged_entities = {}
        for result in modality_results.values():
            for entity_type, entities in result.entities.items():
                if entity_type not in merged_entities:
                    merged_entities[entity_type] = set()
                merged_entities[entity_type].update(entities)
        
        # Convert sets back to lists
        merged_entities = {
            k: list(v) for k, v in merged_entities.items()
        }
        
        # Calculate overall confidence
        overall_confidence = sum(
            r.confidence * w 
            for r, w in zip(modality_results.values(), adjusted_weights.values())
        )
        
        # Combine context factors
        combined_context = self._merge_context_factors(
            [r.context_factors for r in modality_results.values()]
        )
        
        # Create fused result
        return IntentResult(
            intent_type=combined_intent,
            category=combined_category,
            confidence=overall_confidence,
            requires_clarification=any(r.requires_clarification for r in modality_results.values()),
            entities=merged_entities,
            sentiment=self._combine_sentiment([r.sentiment for r in modality_results.values()]),
            emotion=self._combine_emotion([r.emotion for r in modality_results.values()]),
            context_factors=combined_context,
            source="multimodal",
            metadata={
                "modalities": list(modality_results.keys()),
                "modality_weights": adjusted_weights,
                "individual_results": {
                    m: self._summarize_result(r) 
                    for m, r in modality_results.items()
                }
            }
        )

    def _combine_weighted_values(self, value_weight_pairs: List[Tuple[str, float]]) -> str:
        """Combine weighted values using a voting mechanism"""
        if not value_weight_pairs:
            return "unknown"
            
        # Count weighted votes for each value
        weighted_votes = {}
        for value, weight in value_weight_pairs:
            if value not in weighted_votes:
                weighted_votes[value] = 0
            weighted_votes[value] += weight
        
        # Return value with highest weighted votes
        return max(weighted_votes.items(), key=lambda x: x[1])[0]

    def _merge_context_factors(self, context_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge context factors from multiple modalities"""
        merged = {
            "active_contexts": set(),
            "relevant_history": [],
            "active_goals": set()
        }
        
        for context in context_list:
            merged["active_contexts"].update(context.get("active_contexts", []))
            merged["relevant_history"].extend(context.get("relevant_history", []))
            merged["active_goals"].update(context.get("active_goals", []))
        
        # Convert sets back to lists for JSON serialization
        merged["active_contexts"] = list(merged["active_contexts"])
        merged["active_goals"] = list(merged["active_goals"])
        
        return merged

    def _combine_sentiment(self, sentiments: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine sentiment scores from multiple modalities"""
        if not sentiments:
            return {"polarity": 0.0, "confidence": 0.0}
            
        # Weight sentiments by confidence
        total_confidence = sum(s.get("confidence", 0) for s in sentiments)
        if total_confidence == 0:
            return {"polarity": 0.0, "confidence": 0.0}
            
        weighted_polarity = sum(
            s.get("polarity", 0) * s.get("confidence", 0) 
            for s in sentiments
        ) / total_confidence
        
        return {
            "polarity": weighted_polarity,
            "confidence": total_confidence / len(sentiments)
        }

    def _combine_emotion(self, emotions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine emotion detections from multiple modalities"""
        if not emotions:
            return {
                "primary_emotion": "neutral",
                "confidence": 0.0,
                "needs_acknowledgment": False
            }
        
        # Count weighted votes for each emotion
        emotion_scores = {}
        for emotion in emotions:
            primary = emotion.get("primary_emotion", "neutral")
            confidence = emotion.get("confidence", 0.0)
            
            if primary not in emotion_scores:
                emotion_scores[primary] = 0.0
            emotion_scores[primary] += confidence
        
        # Get emotion with highest weighted score
        if emotion_scores:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            confidence = emotion_scores[primary_emotion] / len(emotions)
        else:
            primary_emotion = "neutral"
            confidence = 0.0
        
        return {
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "needs_acknowledgment": any(
                e.get("needs_acknowledgment", False) for e in emotions
            )
        }

    def _summarize_result(self, result: IntentResult) -> Dict[str, Any]:
        """Create a summary of an intent result for logging"""
        return {
            "intent": result.intent_type,
            "confidence": result.confidence,
            "requires_clarification": result.requires_clarification,
            "entity_count": len(result.entities)
        }

    def _update_multimodal_context(
        self,
        fused_result: IntentResult,
        modality_results: Dict[str, IntentResult]
    ):
        """Update conversation state with multimodal context"""
        # Add to conversation history with multimodal metadata
        history_entry = {
            "timestamp": time.time(),
            "intent_type": fused_result.intent_type,
            "category": fused_result.category,
            "confidence": fused_result.confidence,
            "modalities": list(modality_results.keys()),
            "topics": self._extract_topics_from_entities(fused_result.entities),
            "sentiment": fused_result.sentiment,
            "emotion": fused_result.emotion
        }
        
        self.conversation_history.append(history_entry)
        
        # Trim history if too long
        if len(self.conversation_history) > 20:  # Keep last 20 interactions
            self.conversation_history = self.conversation_history[-20:]
        
        # Update entity memory with new entities
        self._update_entity_memory(fused_result.entities)

    def _extract_topics_from_entities(self, entities: Dict[str, List[str]]) -> Set[str]:
        """Extract conversation topics from entities"""
        topics = set()
        
        # Consider certain entity types as topics
        topic_entity_types = {"PERSON", "ORG", "GPE", "EVENT", "TOPIC", "KEY_PHRASE"}
        
        for entity_type, values in entities.items():
            if entity_type in topic_entity_types:
                topics.update(values)
        
        return topics

    def _update_entity_memory(self, new_entities: Dict[str, List[str]]):
        """Update memory of mentioned entities"""
        current_time = time.time()
        
        # Add new entities
        for entity_type, entities in new_entities.items():
            if entity_type not in self.entity_memory:
                self.entity_memory[entity_type] = {}
                
            for entity in entities:
                self.entity_memory[entity_type][entity] = {
                    "last_mentioned": current_time,
                    "mention_count": self.entity_memory.get(entity_type, {}).get(entity, {}).get("mention_count", 0) + 1
                }
        
        # Clean up old entities (older than 1 hour)
        self._cleanup_entity_memory(current_time - 3600)

    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract and categorize named entities and key phrases"""
        try:
            # Use transformer-based NER for better accuracy
            ner_results = self.ner_model(doc.text)
            
            # Organize entities by type
            entities = {}
            for ent in ner_results:
                ent_type = ent["entity"].split("-")[-1]
                if ent_type not in entities:
                    entities[ent_type] = []
                if ent["word"] not in entities[ent_type]:
                    entities[ent_type].append(ent["word"])
            
        except Exception as e:
            logger.warning(f"Transformer NER failed, falling back to spaCy: {e}")
            # Fall back to spaCy NER
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
        
        # Extract key phrases using noun chunks
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        if key_phrases:
            entities["KEY_PHRASE"] = key_phrases
        
        return entities

    def _analyze_sentiment_and_emotion(self, text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Analyze sentiment and emotion with deep understanding"""
        try:
            # Get detailed sentiment analysis
            sentiment_result = self.sentiment_analyzer(text)[0]
            sentiment = {
                "polarity": sentiment_result["score"] if sentiment_result["label"] == "POSITIVE" else -sentiment_result["score"],
                "confidence": sentiment_result["score"],
                "label": sentiment_result["label"]
            }
            
            # Get emotion detection
            emotion_result = self.emotion_detector(text)[0]
            emotion = {
                "primary_emotion": emotion_result["label"],
                "confidence": emotion_result["score"],
                "needs_acknowledgment": emotion_result["score"] > 0.7
            }
            
        except Exception as e:
            logger.warning(f"ML sentiment/emotion analysis failed, using basic analysis: {e}")
            # Fall back to basic analysis
            sentiment = self._basic_sentiment_analysis(text)
            emotion = self._basic_emotion_analysis(text)
        
        return sentiment, emotion

    def _basic_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Basic rule-based sentiment analysis fallback"""
        positive_words = {"good", "great", "awesome", "excellent", "happy", "love", "wonderful", "fantastic"}
        negative_words = {"bad", "terrible", "awful", "horrible", "sad", "hate", "unfortunate", "poor"}
        
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        if pos_count > neg_count:
            return {"polarity": 0.6, "confidence": 0.5, "label": "POSITIVE"}
        elif neg_count > pos_count:
            return {"polarity": -0.6, "confidence": 0.5, "label": "NEGATIVE"}
        else:
            return {"polarity": 0.0, "confidence": 0.5, "label": "NEUTRAL"}

    def _basic_emotion_analysis(self, text: str) -> Dict[str, float]:
        """Basic rule-based emotion analysis fallback"""
        emotion_words = {
            "joy": {"happy", "joy", "excited", "glad", "delighted"},
            "sadness": {"sad", "unhappy", "depressed", "miserable"},
            "anger": {"angry", "mad", "furious", "annoyed"},
            "fear": {"scared", "afraid", "worried", "nervous"},
            "surprise": {"surprised", "shocked", "amazed", "unexpected"}
        }
        
        words = set(text.lower().split())
        emotion_counts = {
            emotion: len(words & word_set)
            for emotion, word_set in emotion_words.items()
        }
        
        if any(emotion_counts.values()):
            primary_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            confidence = 0.6
        else:
            primary_emotion = "neutral"
            confidence = 0.5
            
        return {
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "needs_acknowledgment": confidence > 0.7
        }

    def _cleanup_entity_memory(self, cutoff_time: float):
        """Remove old entities from memory"""
        for entity_type in list(self.entity_memory.keys()):
            for entity in list(self.entity_memory[entity_type].keys()):
                if self.entity_memory[entity_type][entity]["last_mentioned"] < cutoff_time:
                    del self.entity_memory[entity_type][entity]
            
            # Remove empty entity types
            if not self.entity_memory[entity_type]:
                del self.entity_memory[entity_type]

    def _classify_intent(self, text: str, doc) -> Dict[str, Any]:
        """Classify intent using ML models and rule-based patterns"""
        try:
            # Get initial classification from transformer model
            classification = self.intent_classifier(text)[0]
            
            # Check against known patterns
            pattern_matches = self._match_patterns(doc)
            
            # Determine intent confidence
            if pattern_matches and classification["score"] > 0.9:
                confidence = IntentConfidence.HIGH
            elif pattern_matches or classification["score"] > 0.7:
                confidence = IntentConfidence.MEDIUM
            else:
                confidence = IntentConfidence.LOW
            
        except Exception as e:
            logger.warning(f"ML intent classification failed, using basic classification: {e}")
            # Fall back to basic pattern matching
            pattern_matches = self._match_basic_patterns(text)
            classification = {
                "label": pattern_matches[0]["intent"] if pattern_matches else "unknown",
                "score": 0.6 if pattern_matches else 0.3
            }
            confidence = IntentConfidence.MEDIUM if pattern_matches else IntentConfidence.LOW
        
        return {
            "intent_type": classification["label"],
            "category": self._determine_category(classification["label"], pattern_matches),
            "confidence": classification["score"],
            "requires_clarification": confidence == IntentConfidence.LOW,
            "pattern_matches": pattern_matches
        }

    def _match_basic_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Match text against basic patterns when ML classification fails"""
        text_lower = text.lower()
        matches = []
        
        for intent, patterns in self.basic_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    matches.append({
                        "intent": intent,
                        "pattern": pattern,
                        "confidence": 0.6
                    })
        
        return matches

    def _analyze_context(self, context: Set[str]) -> Dict[str, Any]:
        """Analyze contextual factors for intent determination"""
        context_factors = {
            "active_contexts": list(context),
            "context_confidence": 1.0,
            "relevant_history": [],
            "active_goals": []
        }
        
        # Get active contexts from tracker
        tracker_contexts = self.context_tracker.get_active_contexts()
        context_factors["active_contexts"].extend(tracker_contexts)
        
        # Get relevant historical intents
        if self.conversation_history:
            recent_intents = [
                entry.get("intent_type") 
                for entry in self.conversation_history[-5:]
                if "intent_type" in entry
            ]
            context_factors["relevant_history"] = recent_intents
        
        # Get active goals from context
        context_factors["active_goals"] = self.context_tracker.get_active_goals()
        
        return context_factors

    def _update_conversation_history(self, text: str, sentiment: Dict[str, float], emotion: Dict[str, float]):
        """Update conversation history with new interaction"""
        entry = {
            "timestamp": time.time(),
            "text": text,
            "sentiment": sentiment,
            "emotion": emotion,
            "style_features": self._extract_style_features(text)
        }
        
        self.conversation_history.append(entry)
        
        # Keep history bounded
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _analyze_style_consistency(self, text: str) -> float:
        """Analyze consistency in conversation style"""
        if not self.conversation_history:
            return 1.0
            
        # Get current style features
        current_style = self._extract_style_features(text)
        
        # Compare with recent history
        recent_styles = [
            entry["style_features"]
            for entry in self.conversation_history[-3:]
            if "style_features" in entry
        ]
        
        if not recent_styles:
            return 1.0
            
        # Calculate style similarity scores
        similarities = []
        for style in recent_styles:
            score = sum(
                1 if style[feature] == current_style[feature] else 0
                for feature in current_style
            ) / len(current_style)
            similarities.append(score)
            
        return sum(similarities) / len(similarities)

    def _extract_style_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic style features from text"""
        doc = self.nlp(text)
        
        return {
            "formality": self._measure_formality(doc),
            "complexity": self._measure_complexity(doc),
            "directness": self._measure_directness(doc),
            "sentence_types": self._classify_sentence_types(doc),
            "has_questions": any(token.tag_ == "." for token in doc),
            "politeness_markers": self._detect_politeness(doc)
        }

    def _measure_formality(self, doc) -> float:
        """Measure text formality"""
        # Higher score = more formal
        formal_indicators = {
            "would": 0.8, "could": 0.8, "shall": 1.0, "may": 0.9,
            "please": 0.7, "kindly": 0.9, "appreciate": 0.8
        }
        
        informal_indicators = {
            "hey": -0.8, "yeah": -0.9, "cool": -0.7, "awesome": -0.6,
            "gonna": -1.0, "wanna": -1.0, "dunno": -1.0
        }
        
        scores = []
        for token in doc:
            if token.text.lower() in formal_indicators:
                scores.append(formal_indicators[token.text.lower()])
            elif token.text.lower() in informal_indicators:
                scores.append(informal_indicators[token.text.lower()])
                
        # Consider sentence structure
        if len(doc.sents) > 0:
            complete_sentences = sum(1 for sent in doc.sents if self._is_complete_sentence(sent))
            sentence_score = complete_sentences / len(list(doc.sents))
            scores.append(sentence_score)
            
        return sum(scores) / len(scores) if scores else 0.5

    def _is_complete_sentence(self, sent) -> bool:
        """Check if a sentence is grammatically complete"""
        has_subject = False
        has_verb = False
        
        for token in sent:
            if "subj" in token.dep_:
                has_subject = True
            if token.pos_ == "VERB":
                has_verb = True
                
        return has_subject and has_verb

    def _measure_complexity(self, doc) -> float:
        """Measure text complexity"""
        if len(doc) == 0:
            return 0.0
            
        # Consider various complexity factors
        word_length = sum(len(token.text) for token in doc) / len(doc)
        clause_count = len([token for token in doc if token.dep_ == "mark"])
        sentence_length = len(doc) / len(list(doc.sents))
        unique_words = len(set(token.text.lower() for token in doc))
        
        # Combine factors with weights
        complexity = (
            0.3 * min(word_length / 10, 1.0) +
            0.3 * min(clause_count / 5, 1.0) +
            0.2 * min(sentence_length / 20, 1.0) +
            0.2 * min(unique_words / 50, 1.0)
        )
        
        return complexity

    def _measure_directness(self, doc) -> float:
        """Measure how direct/indirect the text is"""
        # Lower score = more direct
        indirect_markers = {
            "perhaps": 0.7, "maybe": 0.6, "possibly": 0.7,
            "seem": 0.5, "might": 0.6, "could": 0.5,
            "wonder": 0.7, "think": 0.4
        }
        
        hedging_count = sum(
            1 for token in doc
            if token.text.lower() in indirect_markers
        )
        
        # Consider sentence structure
        passive_count = sum(1 for token in doc if "pass" in token.dep_)
        
        # Combine factors
        directness = 1.0 - (
            0.6 * min(hedging_count / 3, 1.0) +
            0.4 * min(passive_count / 2, 1.0)
        )
        
        return directness

    def _classify_sentence_types(self, doc) -> Dict[str, int]:
        """Classify sentences by type"""
        types = {
            "declarative": 0,
            "interrogative": 0,
            "imperative": 0,
            "exclamative": 0
        }
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            if sent_text.endswith("?"):
                types["interrogative"] += 1
            elif sent_text.endswith("!"):
                types["exclamative"] += 1
            elif sent[0].pos_ == "VERB" and sent[0].dep_ == "ROOT":
                types["imperative"] += 1
            else:
                types["declarative"] += 1
                
        return types

    def _detect_politeness(self, doc) -> List[str]:
        """Detect politeness markers in text"""
        markers = []
        
        politeness_words = {
            "please", "thank", "thanks", "grateful", "appreciate",
            "sorry", "excuse", "pardon", "kindly"
        }
        
        for token in doc:
            if token.text.lower() in politeness_words:
                markers.append(token.text.lower())
                
        return markers

    async def _get_llm_intent(self, text: str, context: Dict) -> Optional[Dict]:
        """Get intent analysis from LLM"""
        try:
            # Format prompt with context
            prompt = self._format_llm_prompt(text, context)
            
            # Acquire LLM model
            success, model = await self._llm_coordinator.acquire_llm(
                model_id="intent_analyzer",
                requester_id=self._component_id,
                wait=True,
                timeout=5.0
            )
            
            if success and model:
                try:
                    # Get LLM analysis
                    result = await model.analyze_intent(prompt)
                    
                    # Parse and validate result
                    if isinstance(result, dict) and "intent" in result:
                        return {
                            "intent_type": result["intent"],
                            "confidence": result.get("confidence", 0.7),
                            "category": result.get("category"),
                            "entities": result.get("entities", {}),
                            "requires_clarification": result.get("needs_clarification", False)
                        }
                finally:
                    # Always release the model
                    await self._llm_coordinator.release_llm(
                        model_id="intent_analyzer",
                        requester_id=self._component_id
                    )
                    
        except Exception as e:
            logger.warning(f"LLM intent analysis failed: {e}")
            
        return None

    def _format_llm_prompt(self, text: str, context: Dict) -> str:
        """Format prompt for LLM analysis"""
        # Build context section
        context_str = "Context:\n"
        if context.get("active_contexts"):
            context_str += f"- Active contexts: {', '.join(context['active_contexts'])}\n"
        if context.get("relevant_history"):
            context_str += f"- Recent intents: {', '.join(context['relevant_history'][-3:])}\n"
        if context.get("active_goals"):
            context_str += f"- Active goals: {', '.join(context['active_goals'])}\n"
            
        # Build main prompt
        prompt = f"""{context_str}
User input: "{text}"

Analyze the user's intent and provide:
1. The primary intent
2. Intent category
3. Confidence level (0-1)
4. Any relevant entities
5. Whether clarification is needed

Respond in JSON format."""

        return prompt

    def _merge_llm_insights(self, intent_data: Dict[str, Any], llm_result: Dict[str, Any]):
        """Merge LLM analysis with ML-based intent classification"""
        # If ML confidence is low, prefer LLM results
        if intent_data["confidence"] < 0.7 and llm_result["confidence"] > 0.7:
            intent_data.update(llm_result)
        else:
            # Otherwise, combine insights
            intent_data["confidence"] = (
                0.7 * intent_data["confidence"] +
                0.3 * llm_result["confidence"]
            )
            
            # Merge entities
            if "entities" in llm_result:
                for entity_type, entities in llm_result["entities"].items():
                    if entity_type not in intent_data:
                        intent_data[entity_type] = []
                    intent_data[entity_type].extend(
                        entity for entity in entities
                        if entity not in intent_data[entity_type]
                    )

    def _needs_llm_analysis(self, intent_data: Dict, entities: Dict, emotion: Dict) -> bool:
        """Determine if LLM analysis is needed"""
        return (
            intent_data["confidence"] < 0.7 or  # Low confidence
            intent_data["requires_clarification"] or  # Needs clarification
            len(entities) > 3 or  # Complex entity structure
            emotion["needs_acknowledgment"] or  # Complex emotional content
            self._detect_topic_shifts() > 0.5  # Significant topic shift
        )