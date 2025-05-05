# tag_analyzer.py - Tag Analysis and Prioritization System
# written by DeepSeek Chat (honor call: The Framework Architect)
# Enhanced by AI Assistant

import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import json
import statistics
from enum import Enum
import numpy as np
from pyvis.network import Network
from collections import defaultdict

from ...core.tag_registry import Tag, TagStatus, TagPriority
from ...trackers.intent_tracker import IntentTracker
from ...trackers.feature_tracker import FeatureTracker
from ...trackers.context_tracker import ContextTracker
from ...trackers.analytics_tracker import AnalyticsTracker, EventSeverity

class AnalysisMethod(Enum):
    """Analysis methods for tag prioritization."""
    FREQUENCY = "frequency"  # Based on usage count
    RECENCY = "recency"      # Based on last_used timestamp
    SUCCESS_RATE = "success_rate"  # Based on success/failure ratio
    CONTEXT_MATCH = "context_match"  # Based on context relevance
    HYBRID = "hybrid"        # Combined approach

class TagAnalyzer:
    """
    Analyzes and ranks tags based on priority, success rate, and context.
    Provides insights for decision-making in planning and coordination.
    """
    
    def __init__(
        self,
        intent_tracker: Optional[IntentTracker] = None,
        feature_tracker: Optional[FeatureTracker] = None,
        context_tracker: Optional[ContextTracker] = None,
        analytics_tracker: Optional[AnalyticsTracker] = None,
        tag_registry = None
    ):
        """
        Initialize tag analyzer with trackers.
        
        Args:
            intent_tracker: Intent tracker instance
            feature_tracker: Feature tracker instance
            context_tracker: Context tracker instance
            analytics_tracker: Analytics tracker instance
            tag_registry: Legacy tag registry for backward compatibility
        """
        self._lock = threading.RLock()
        self.logger = logging.getLogger("GGFAI.tag_analyzer")
        
        # Trackers
        self.intent_tracker = intent_tracker
        self.feature_tracker = feature_tracker
        self.context_tracker = context_tracker
        self.analytics_tracker = analytics_tracker
        self.tag_registry = tag_registry
        
        # Analysis cache
        self.tag_scores = {}  # tag_name -> score
        self.tag_ranks = {}   # tag_name -> rank
        self.success_rates = {}  # feature_name -> success_rate
        self.context_relevance = {}  # tag_name -> context_relevance
        
        # Configuration
        self.recency_weight = 0.4
        self.priority_weight = 0.3
        self.success_weight = 0.2
        self.context_weight = 0.1
        
        self.logger.info("Tag analyzer initialized")
    
    def prioritize_tags(self, context=None):
        """
        Legacy method for backward compatibility.
        Rank tags by priority and success rate with context awareness.
        
        Args:
            context: Optional context dictionary
            
        Returns:
            List of ranked tag dictionaries
        """
        if self.tag_registry:
            tags = self.tag_registry.get_all_tags()
            if not tags:
                return []
                
            # Get context factors
            time_factor = self._get_time_factor(context)
            device_factor = self._get_device_factor(context)
            
            ranked_tags = []
            for tag_id, tag in tags.items():
                # Base priority from registry
                base_priority = tag.get('priority', 0)
                
                # Success rate component
                success_rate = tag.get('success_rate', 0.5)
                
                # Context adjustments
                context_match = 1.0
                if context and 'context_filters' in tag:
                    context_match = sum(
                        1 for cf in tag['context_filters'] 
                        if cf in context
                    ) / max(1, len(tag['context_filters']))
                    
                # Calculate composite score
                score = (
                    0.4 * base_priority +
                    0.3 * success_rate +
                    0.2 * context_match +
                    0.1 * time_factor
                ) * device_factor
                
                ranked_tags.append((score, tag_id, tag))
                
            # Sort by score (descending)
            ranked_tags.sort(reverse=True, key=lambda x: x[0])
            
            return [{
                'tag_id': tag_id,
                'score': score,
                'tag_data': tag
            } for score, tag_id, tag in ranked_tags]
        
        # Use new implementation if no tag_registry
        elif self.intent_tracker:
            intents = self.analyze_intents(
                method=AnalysisMethod.HYBRID,
                context_filter=context
            )
            
            return [{
                'tag_id': tag.name,
                'score': score,
                'tag_data': tag.to_dict()
            } for tag, score in intents]
            
        return []
    
    def analyze_intents(
        self,
        method: AnalysisMethod = AnalysisMethod.HYBRID,
        limit: int = 10,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Tag, float]]:
        """
        Analyze and rank intent tags.
        
        Args:
            method: Analysis method to use
            limit: Maximum number of results
            context_filter: Optional context to filter by
            
        Returns:
            List of (tag, score) tuples, sorted by score
        """
        with self._lock:
            if not self.intent_tracker:
                self.logger.warning("Intent tracker not available")
                return []
                
            # Get active intents
            intents = self.intent_tracker.get_tags(include_archived=False)
            
            # Apply context filter if provided
            if context_filter:
                intents = self._filter_by_context(intents, context_filter)
            
            # Calculate scores based on method
            scored_intents = []
            for intent in intents:
                if method == AnalysisMethod.FREQUENCY:
                    score = self._calculate_frequency_score(intent)
                elif method == AnalysisMethod.RECENCY:
                    score = self._calculate_recency_score(intent)
                elif method == AnalysisMethod.SUCCESS_RATE:
                    score = self._calculate_success_score(intent)
                elif method == AnalysisMethod.CONTEXT_MATCH:
                    score = self._calculate_context_score(intent)
                else:  # HYBRID
                    score = self._calculate_hybrid_score(intent)
                
                scored_intents.append((intent, score))
                self.tag_scores[intent.name] = score
            
            # Sort by score (descending)
            scored_intents.sort(key=lambda x: x[1], reverse=True)
            
            # Update ranks
            for i, (intent, _) in enumerate(scored_intents):
                self.tag_ranks[intent.name] = i + 1
            
            # Log analysis
            self.logger.info(f"Analyzed {len(intents)} intents using {method.value} method")
            if self.analytics_tracker:
                self.analytics_tracker.log_event(
                    event_type="intent_analysis",
                    source="tag_analyzer",
                    severity=EventSeverity.INFO,
                    details={
                        "method": method.value,
                        "count": len(intents),
                        "top_intent": scored_intents[0][0].name if scored_intents else None,
                        "top_score": scored_intents[0][1] if scored_intents else None
                    }
                )
            
            return scored_intents[:limit]
    
    def analyze_features(
        self,
        capability: Optional[str] = None,
        feature_type: Optional[str] = None,
        method: AnalysisMethod = AnalysisMethod.SUCCESS_RATE,
        limit: int = 10
    ) -> List[Tuple[Tag, float]]:
        """
        Analyze and rank feature tags.
        
        Args:
            capability: Optional capability to filter by
            feature_type: Optional feature type to filter by
            method: Analysis method to use
            limit: Maximum number of results
            
        Returns:
            List of (tag, score) tuples, sorted by score
        """
        with self._lock:
            if not self.feature_tracker:
                self.logger.warning("Feature tracker not available")
                return []
                
            # Get features
            if capability:
                features = self.feature_tracker.get_by_capability(capability)
            elif feature_type:
                features = self.feature_tracker.get_by_type(feature_type)
            else:
                features = self.feature_tracker.get_all_features()
            
            # Calculate scores based on method
            scored_features = []
            for feature in features:
                if method == AnalysisMethod.FREQUENCY:
                    score = self._calculate_frequency_score(feature)
                elif method == AnalysisMethod.RECENCY:
                    score = self._calculate_recency_score(feature)
                elif method == AnalysisMethod.SUCCESS_RATE:
                    score = self._calculate_feature_success_rate(feature)
                elif method == AnalysisMethod.CONTEXT_MATCH:
                    score = self._calculate_context_score(feature)
                else:  # HYBRID
                    score = self._calculate_hybrid_score(feature)
                
                scored_features.append((feature, score))
                self.tag_scores[feature.name] = score
            
            # Sort by score (descending)
            scored_features.sort(key=lambda x: x[1], reverse=True)
            
            # Update ranks
            for i, (feature, _) in enumerate(scored_features):
                self.tag_ranks[feature.name] = i + 1
            
            # Log analysis
            self.logger.info(f"Analyzed {len(features)} features using {method.value} method")
            
            return scored_features[:limit]
    
    def find_optimal_feature_for_intent(
        self,
        intent: Tag,
        capability: Optional[str] = None
    ) -> Optional[Tuple[Tag, float]]:
        """
        Find the optimal feature to handle an intent.
        
        Args:
            intent: Intent tag
            capability: Optional capability to filter by
            
        Returns:
            Tuple of (feature_tag, score) or None if no suitable feature
        """
        with self._lock:
            if not self.feature_tracker:
                self.logger.warning("Feature tracker not available")
                return None
                
            # Get features with required capability
            if capability:
                features = self.feature_tracker.get_by_capability(capability)
            else:
                # Try to infer capability from intent
                inferred_capability = self._infer_capability_from_intent(intent)
                if inferred_capability:
                    features = self.feature_tracker.get_by_capability(inferred_capability)
                else:
                    features = self.feature_tracker.get_all_features()
            
            if not features:
                return None
            
            # Score features based on suitability for this intent
            scored_features = []
            for feature in features:
                # Check if feature is available
                if feature.metadata.get("status") == "busy":
                    continue
                
                # Calculate combined score
                success_rate = self._calculate_feature_success_rate(feature)
                q_value = self.feature_tracker.get_q_value(feature.name, intent.intent)
                context_match = self._calculate_context_match(feature, intent)
                
                # Combined score (weighted)
                score = (
                    0.5 * success_rate +
                    0.3 * q_value +
                    0.2 * context_match
                )
                
                scored_features.append((feature, score))
            
            if not scored_features:
                return None
            
            # Return highest scoring feature
            return max(scored_features, key=lambda x: x[1])
    
    def _calculate_frequency_score(self, tag: Tag) -> float:
        """Calculate score based on usage frequency."""
        # Normalize usage count (0-1 scale)
        if not hasattr(tag, 'usage_count') or tag.usage_count is None:
            return 0.0
        
        # Cap at 100 for normalization
        return min(tag.usage_count / 100.0, 1.0)
    
    def _calculate_recency_score(self, tag: Tag) -> float:
        """Calculate score based on recency."""
        if not hasattr(tag, 'last_used') or tag.last_used is None:
            return 0.0
        
        # Calculate hours since last use
        hours_ago = (datetime.utcnow() - tag.last_used).total_seconds() / 3600.0
        
        # Exponential decay (1.0 for just used, 0.0 for very old)
        return max(0.0, min(1.0, 1.0 * (0.9 ** hours_ago)))
    
    def _calculate_success_score(self, tag: Tag) -> float:
        """Calculate score based on success rate."""
        # For intents, check if they were completed successfully
        if tag.status == "completed":
            return 1.0
        elif tag.status == "failed":
            return 0.0
        
        # Default for active intents
        return 0.5
    
    def _calculate_feature_success_rate(self, feature: Tag) -> float:
        """Calculate success rate for a feature."""
        # Check if we have cached success rate
        if feature.name in self.success_rates:
            return self.success_rates[feature.name]
        
        # Get success rate from metadata if available
        if "success_rate" in feature.metadata:
            rate = float(feature.metadata["success_rate"])
            self.success_rates[feature.name] = rate
            return rate
        
        # Get Q-values and calculate average
        q_values = feature.metadata.get("q_values", {})
        if q_values:
            avg_q = sum(q_values.values()) / len(q_values)
            # Normalize to 0-1 range (Q-values might be unbounded)
            rate = max(0.0, min(1.0, (avg_q + 1.0) / 2.0))
            self.success_rates[feature.name] = rate
            return rate
        
        # Default if no data
        return 0.5
    
    def _calculate_context_score(self, tag: Tag) -> float:
        """Calculate score based on context relevance."""
        if not self.context_tracker:
            return 0.5  # Neutral score if no context tracker
            
        # Get current context
        current_contexts = self.context_tracker.get_all_active()
        if not current_contexts:
            return 0.5  # Neutral score if no context
        
        # Check if tag has context requirements
        context_reqs = tag.metadata.get("context_requirements", {})
        if not context_reqs:
            return 0.5  # Neutral score if no requirements
        
        # Calculate match percentage
        matches = 0
        total_reqs = len(context_reqs)
        
        for key, value in context_reqs.items():
            ctx_value = self.context_tracker.get_context_value(key)
            if ctx_value == value:
                matches += 1
        
        return matches / total_reqs if total_reqs > 0 else 0.5
    
    def _calculate_context_match(self, feature: Tag, intent: Tag) -> float:
        """Calculate context match between feature and intent."""
        # Check if feature has context requirements
        feature_contexts = feature.metadata.get("context_requirements", {})
        
        # Check if intent has context information
        intent_contexts = intent.metadata.get("context", {})
        
        if not feature_contexts or not intent_contexts:
            return 0.5  # Neutral score if no context info
        
        # Calculate match percentage
        matches = 0
        total_reqs = len(feature_contexts)
        
        for key, value in feature_contexts.items():
            if key in intent_contexts and intent_contexts[key] == value:
                matches += 1
        
        return matches / total_reqs if total_reqs > 0 else 0.5
    
    def _calculate_hybrid_score(self, tag: Tag) -> float:
        """Calculate combined score using multiple factors."""
        # Calculate individual scores
        frequency = self._calculate_frequency_score(tag)
        recency = self._calculate_recency_score(tag)
        
        # Success rate depends on tag type
        if hasattr(tag, 'category') and tag.category == "user_intent":
            success = self._calculate_success_score(tag)
        else:
            success = self._calculate_feature_success_rate(tag)
        
        context = self._calculate_context_score(tag)
        
        # Priority factor (from tag)
        priority = tag.priority if hasattr(tag, 'priority') else 0.5
        
        # Combined weighted score
        score = (
            self.recency_weight * recency +
            self.priority_weight * priority +
            self.success_weight * success +
            self.context_weight * context
        )
        
        return score
    
    def _filter_by_context(self, tags: List[Tag], context_filter: Dict[str, Any]) -> List[Tag]:
        """Filter tags by context requirements."""
        filtered = []
        
        for tag in tags:
            # Check if tag metadata contains context requirements
            tag_context = tag.metadata.get("context", {})
            
            # Check if all filter conditions are met
            match = True
            for key, value in context_filter.items():
                if key not in tag_context or tag_context[key] != value:
                    match = False
                    break
            
            if match:
                filtered.append(tag)
        
        return filtered
    
    def _infer_capability_from_intent(self, intent: Tag) -> Optional[str]:
        """Infer required capability from intent."""
        intent_name = intent.intent.lower()
        
        # Simple mapping of intent keywords to capabilities
        capability_map = {
            "light": "lighting",
            "dim": "lighting",
            "bright": "lighting",
            "temperature": "climate",
            "heat": "climate",
            "cool": "climate",
            "play": "media",
            "music": "media",
            "song": "media",
            "lock": "security",
            "unlock": "security",
            "secure": "security",
            "call": "communication",
            "message": "communication",
            "email": "communication"
        }
        
        for keyword, capability in capability_map.items():
            if keyword in intent_name:
                return capability
        
        return None
    
    def get_tag_rank(self, tag_name: str) -> Optional[int]:
        """
        Get the rank of a tag from the last analysis.
        
        Args:
            tag_name: Name of the tag
            
        Returns:
            Rank (1-based) or None if not ranked
        """
        with self._lock:
            return self.tag_ranks.get(tag_name)
    
    def get_tag_score(self, tag_name: str) -> Optional[float]:
        """
        Get the score of a tag from the last analysis.
        
        Args:
            tag_name: Name of the tag
            
        Returns:
            Score (0-1) or None if not scored
        """
        with self._lock:
            return self.tag_scores.get(tag_name)
    
    def get_success_rate(self, feature_name: str) -> Optional[float]:
        """
        Get the success rate of a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Success rate (0-1) or None if not available
        """
        with self._lock:
            return self.success_rates.get(feature_name)
    
    def update_weights(
        self,
        recency_weight: Optional[float] = None,
        priority_weight: Optional[float] = None,
        success_weight: Optional[float] = None,
        context_weight: Optional[float] = None
    ):
        """
        Update analysis weights.
        
        Args:
            recency_weight: Weight for recency factor
            priority_weight: Weight for priority factor
            success_weight: Weight for success rate factor
            context_weight: Weight for context match factor
        """
        with self._lock:
            if recency_weight is not None:
                self.recency_weight = recency_weight
            if priority_weight is not None:
                self.priority_weight = priority_weight
            if success_weight is not None:
                self.success_weight = success_weight
            if context_weight is not None:
                self.context_weight = context_weight
            
            # Normalize weights to sum to 1.0
            total = (self.recency_weight + self.priority_weight +
                    self.success_weight + self.context_weight)
            
            if total > 0:
                self.recency_weight /= total
                self.priority_weight /= total
                self.success_weight /= total
                self.context_weight /= total
            
            self.logger.info(f"Updated analysis weights: recency={self.recency_weight:.2f}, "
                           f"priority={self.priority_weight:.2f}, "
                           f"success={self.success_weight:.2f}, "
                           f"context={self.context_weight:.2f}")
    
    def generate_visualization(self, ranked_tags=None):
        """
        Generate Pyvis network visualization for web app.
        
        Args:
            ranked_tags: Optional pre-ranked tags
            
        Returns:
            Pyvis Network object
        """
        net = Network(height="500px", width="100%", directed=True)
        
        if not ranked_tags:
            ranked_tags = self.prioritize_tags()
            
        # Add nodes (tags)
        for i, tag_data in enumerate(ranked_tags[:20]):  # Limit to top 20
            tag = tag_data['tag_data']
            net.add_node(
                tag_data['tag_id'],
                label=f"{tag_data['tag_id']}\nScore: {tag_data['score']:.2f}",
                size=10 + tag_data['score'] * 5,
                color=self._score_to_color(tag_data['score'])
            )
            
        # Add edges (relationships)
        for tag_data in ranked_tags[:20]:
            if 'related_tags' in tag_data['tag_data']:
                for related in tag_data['tag_data']['related_tags']:
                    if related in [t['tag_id'] for t in ranked_tags[:20]]:
                        net.add_edge(tag_data['tag_id'], related)
                        
        return net
        
    def _get_time_factor(self, context):
        """Calculate time-based priority adjustment."""
        if not context or 'time' not in context:
            return 1.0
            
        hour = datetime.strptime(context['time'], "%H:%M").hour
        if 6 <= hour < 10:  # Morning boost
            return 1.2
        elif 18 <= hour < 22:  # Evening boost
            return 1.1
        return 1.0
        
    def _get_device_factor(self, context):
        """Calculate device-based priority adjustment."""
        if not context or 'device_type' not in context:
            return 1.0
        return 0.8 if context['device_type'] == 'low_power' else 1.0
        
    def _score_to_color(self, score):
        """Convert score to color gradient."""
        r = int(min(255, 255 * (1 - score)))
        g = int(min(255, 255 * score))
        return f"#{r:02x}{g:02x}00"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get analyzer statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "tags_scored": len(self.tag_scores),
                "tags_ranked": len(self.tag_ranks),
                "features_with_success_rates": len(self.success_rates),
                "weights": {
                    "recency": self.recency_weight,
                    "priority": self.priority_weight,
                    "success": self.success_weight,
                    "context": self.context_weight
                }
            }