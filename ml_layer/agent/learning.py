"""
GGFAI Learning Service - UCB1 Bandit Learning with Tag Integration
"""
import numpy as np
import time
from typing import List, Dict, Set, Optional
import logging
from threading import Lock
from dataclasses import dataclass, field
from collections import defaultdict
from ...core.tag_registry import TagRegistry

logger = logging.getLogger("GGFAI.learning")

@dataclass
class ArmMetrics:
    value: float = 0.0
    pulls: int = 0
    successes: int = 0
    confidence_bound: float = float('inf')

class LearningService:
    def __init__(self, tag_registry: TagRegistry, exploration_weight: float = 2.0):
        self.tag_registry = tag_registry
        self.exploration_weight = exploration_weight
        self.arms: Dict[str, ArmMetrics] = defaultdict(ArmMetrics)
        self._lock = Lock()
        self.total_pulls = 0
        
    def select_action(self, available_actions: List[str], context: Dict[str, any]) -> str:
        """Select next action using UCB1 with context-aware bounds."""
        if not available_actions:
            raise ValueError("No actions available")
            
        with self._lock:
            # Initialize new arms
            for action in available_actions:
                if action not in self.arms:
                    self.arms[action] = ArmMetrics()
                    
            # Calculate UCB values with context adjustment
            ucb_values = {}
            for action in available_actions:
                metrics = self.arms[action]
                
                # Basic UCB1 calculation
                exploitation = metrics.value
                exploration = self.exploration_weight * np.sqrt(
                    np.log(self.total_pulls + 1) / (metrics.pulls + 1)
                )
                
                # Adjust bounds based on context
                context_multiplier = self._get_context_multiplier(action, context)
                ucb_values[action] = (exploitation + exploration) * context_multiplier
                
            # Select arm with highest UCB value
            selected_action = max(ucb_values.items(), key=lambda x: x[1])[0]
            return selected_action
            
    def update(self, action: str, reward: float, context: Optional[Dict] = None) -> None:
        """Update action value estimates with new reward."""
        with self._lock:
            if action not in self.arms:
                self.arms[action] = ArmMetrics()
                
            metrics = self.arms[action]
            metrics.pulls += 1
            self.total_pulls += 1
            
            # Update running average with learning rate decay
            lr = 1.0 / metrics.pulls  # Natural decay
            metrics.value = (1 - lr) * metrics.value + lr * reward
            
            # Update success tracking
            if reward > 0.5:  # Consider as success if reward > 0.5
                metrics.successes += 1
                
            # Update confidence bound
            metrics.confidence_bound = self._calculate_confidence_bound(metrics)
            
            # Log significant changes
            if metrics.pulls % 10 == 0:  # Log every 10 pulls
                logger.info(f"Action {action} metrics after {metrics.pulls} pulls: "
                          f"value={metrics.value:.3f}, "
                          f"success_rate={metrics.successes/metrics.pulls:.3f}")
                
    def _get_context_multiplier(self, action: str, context: Dict) -> float:
        """Calculate context-based adjustment factor."""
        if not context:
            return 1.0
            
        multiplier = 1.0
        
        # Boost actions that match context tags
        matching_tags = self.tag_registry.get_matching_tags(action, context)
        multiplier *= (1 + 0.1 * len(matching_tags))
        
        # Adjust for time-of-day preference if specified
        if 'time_of_day' in context:
            action_tags = self.tag_registry.get_tags_for_action(action)
            if f"preferred_time_{context['time_of_day']}" in action_tags:
                multiplier *= 1.2
                
        return multiplier
        
    def _calculate_confidence_bound(self, metrics: ArmMetrics) -> float:
        """Calculate 95% confidence interval using Wilson score."""
        if metrics.pulls == 0:
            return float('inf')
            
        z = 1.96  # 95% confidence
        p = metrics.successes / metrics.pulls
        n = metrics.pulls
        
        denominator = 1 + z*z/n
        centre_adj = p + z*z/(2*n)
        spread = z * np.sqrt((p*(1-p) + z*z/(4*n))/n)
        
        return (centre_adj + spread) / denominator

    def get_action_metrics(self, action: str) -> Dict:
        """Get current metrics for an action."""
        if action not in self.arms:
            return None
            
        metrics = self.arms[action]
        return {
            'value': metrics.value,
            'pulls': metrics.pulls,
            'success_rate': metrics.successes/metrics.pulls if metrics.pulls > 0 else 0,
            'confidence_bound': metrics.confidence_bound
        }

    def schedule_method_selection(self, methods: List[Dict], context: Dict) -> List[Dict]:
        """Schedule methods based on context-aware UCB scoring."""
        if not methods:
            return []

        scored_methods = []
        for method in methods:
            method_id = method.get("id", "unknown")
            
            # Get base metrics
            metrics = self.arms.get(method_id, ArmMetrics())
            
            # Calculate UCB components
            exploitation = metrics.value
            exploration = self.exploration_weight * np.sqrt(
                np.log(self.total_pulls + 1) / (metrics.pulls + 1)
            )
            
            # Apply context multiplier
            context_multiplier = self._get_context_multiplier(method_id, context)
            
            # Final score with confidence bound
            confidence_bound = self._calculate_confidence_bound(metrics)
            score = (exploitation + exploration) * context_multiplier
            
            scored_methods.append({
                **method,
                "score": score,
                "confidence": confidence_bound,
                "metrics": self.get_action_metrics(method_id)
            })
            
        # Sort by score while considering confidence bounds
        scored_methods.sort(
            key=lambda x: (x["score"] * (1 + x["confidence"]), x["metrics"]["success_rate"]),
            reverse=True
        )
        
        return scored_methods

    def update_schedule_rewards(self, method_ids: List[str], rewards: List[float], context: Optional[Dict] = None) -> None:
        """Update multiple method rewards in batch."""
        if len(method_ids) != len(rewards):
            raise ValueError("Method IDs and rewards must have same length")
            
        for method_id, reward in zip(method_ids, rewards):
            self.update(method_id, reward, context)

    def get_confidence_threshold(self, context: Optional[Dict] = None) -> float:
        """Calculate dynamic confidence threshold based on context."""
        base_threshold = 0.7
        
        if not context:
            return base_threshold
            
        # Adjust threshold based on context priority
        if "priority" in context:
            priority = float(context["priority"])
            base_threshold += 0.1 * priority  # Higher priority = higher confidence needed
            
        # Adjust for time sensitivity
        if "deadline" in context:
            time_left = context["deadline"] - time.time()
            if time_left < 3600:  # Less than 1 hour
                base_threshold *= 0.9  # Relax threshold for urgent tasks
                
        return min(0.95, max(0.5, base_threshold))  # Keep between 0.5 and 0.95