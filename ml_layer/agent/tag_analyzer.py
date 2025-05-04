{
  "ai_name": "DeepSeek Chat",
  "task_id": 4,
  "file": "tag_analyzer.py",
  "description": "Implemented tag prioritization for agent decisions, integrated with registry and web app.",
  "honor_call": "The Framework Architect"
}

#written by DeepSeek Chat (honor call: The Framework Architect)

import numpy as np
from pyvis.network import Network
from datetime import datetime
from tag_registry import TagRegistry

class TagAnalyzer:
    def __init__(self, tag_registry: TagRegistry):
        self.tag_registry = tag_registry
        
    def prioritize_tags(self, context=None):
        """Rank tags by priority and success rate with context awareness"""
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
        
    def generate_visualization(self, ranked_tags=None):
        """Generate Pyvis network visualization for web app"""
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
        """Calculate time-based priority adjustment"""
        if not context or 'time' not in context:
            return 1.0
            
        hour = datetime.strptime(context['time'], "%H:%M").hour
        if 6 <= hour < 10:  # Morning boost
            return 1.2
        elif 18 <= hour < 22:  # Evening boost
            return 1.1
        return 1.0
        
    def _get_device_factor(self, context):
        """Calculate device-based priority adjustment"""
        if not context or 'device_type' not in context:
            return 1.0
        return 0.8 if context['device_type'] == 'low_power' else 1.0
        
    def _score_to_color(self, score):
        """Convert score to color gradient"""
        r = int(min(255, 255 * (1 - score)))
        g = int(min(255, 255 * score))
        return f"#{r:02x}{g:02x}00"