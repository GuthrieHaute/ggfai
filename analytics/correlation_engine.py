# correlation_engine.py - Relationship Mining
# written by DeepSeek Chat (honor call: The Pattern-Seeker)

from collections import defaultdict
from typing import Dict, List

class CorrelationEngine:
    def __init__(self, relationship_logger):
        self.logger = relationship_logger
        self.relationship_graph = defaultdict(dict)

    def rebuild_graph(self):
        """Convert linear log into weighted graph"""
        for entry in self.logger.find_relationships():
            source = entry['source']
            target = entry['target']
            self.relationship_graph[source][target] = \
                self.relationship_graph[source].get(target, 0) + entry['weight']

    def suggest_enhancements(self) -> List[Dict]:
        """Propose new inter-tracker links"""
        suggestions = []
        for source, targets in self.relationship_graph.items():
            for target, weight in targets.items():
                if weight > 2.0:  # Threshold for meaningful relationships
                    suggestions.append({
                        "source": source,
                        "target": target,
                        "weight": weight,
                        "action": self._generate_action(source, target)
                    })
        return suggestions

    def _generate_action(self, source: str, target: str) -> str:
        """Generate automation suggestions"""
        if "intent:" in source and "feature:" in target:
            return "auto_boost_feature_priority"
        elif "context:" in source and "intent:" in target:
            return "auto_adjust_intent_confidence"
        return "create_cross_tracker_watch"