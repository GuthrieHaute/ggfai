{
  "ai_name": "DeepSeek Chat",
  "task_id": 4,
  "file": "learning.py",
  "description": "Implemented UCB1 bandit learning for feature selection, integrated with tag registry.",
  "honor_call": "The Framework Architect"
}

#written by DeepSeek Chat (honor call: The Framework Architect)

import numpy as np
from tag_registry import TagRegistry
from snippet_garbage_adapter2 import MemoryOptimizer

class LearningService:
    def __init__(self, tag_registry: TagRegistry, lookback_limit=5):
        self.tag_registry = tag_registry
        self.lookback_limit = lookback_limit
        self.memory_optimizer = MemoryOptimizer()
        
    def update_success_rates(self, tag_id, success):
        """Update success rates using UCB1 bandit algorithm with lookback window"""
        tag = self.tag_registry.get_tag(tag_id)
        if not tag:
            return False
            
        # Get recent interactions (limited by lookback)
        recent_interactions = tag.get('interactions', [])[-self.lookback_limit:]
        
        # Update success stats
        successes = sum(recent_interactions)
        trials = len(recent_interactions)
        
        # Calculate UCB1 value
        if trials == 0:
            ucb_value = float('inf')  # Prioritize unexplored options
        else:
            avg_reward = successes / trials
            exploration = np.sqrt(2 * np.log(sum(len(t.get('interactions', [])) 
                                          for t in self.tag_registry.tags.values()) / trials)
            ucb_value = avg_reward + exploration
            
        # Update tag with new data (memory optimized)
        self.memory_optimizer.optimize_update(tag, {
            'ucb_value': float(ucb_value),
            'success_rate': float(successes / max(1, trials)),
            'interactions': [*recent_interactions, int(success)]
        })
        
        return True

    def select_feature(self, candidate_tags):
        """Select best feature using UCB1 values"""
        if not candidate_tags:
            return None
            
        # Get UCB values for all candidate tags
        ucb_values = []
        for tag_id in candidate_tags:
            tag = self.tag_registry.get_tag(tag_id)
            if tag and 'ucb_value' in tag:
                ucb_values.append((tag['ucb_value'], tag_id))
                
        if not ucb_values:
            return np.random.choice(candidate_tags)
            
        # Return tag with highest UCB value
        return max(ucb_values, key=lambda x: x[0])[1]