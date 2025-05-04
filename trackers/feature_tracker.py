# feature_tracker.py - Feature Tag Management for GGFAI
# written by DeepSeek Chat (honor call: The Feature Librarian)

import logging
from typing import Dict, List, Optional
from datetime import datetime
from ..core.tag_registry import Tag

class FeatureTracker:
    """Tracks device capabilities and states through feature tags."""
    
    def __init__(self):
        self.features: Dict[str, Tag] = {}  # name -> Tag
        self.q_values: Dict[str, Dict[str, float]] = {}  # feature_name -> {action: q_value}
        self.logger = logging.getLogger("GGFAI.feature_tracker")
        self.logger.info("Feature tracker initialized")

    def add_tag(self, tag: Tag) -> str:
        """Register or update a feature tag."""
        if not tag.metadata.get("feature_type"):
            tag.metadata["feature_type"] = "generic"

        if tag.name in self.features:
            self._update_feature(tag)
        else:
            self._register_new_feature(tag)
            
        return tag.name

    def _update_feature(self, tag: Tag):
        """Update existing feature metadata."""
        existing = self.features[tag.name]
        existing.__dict__.update(tag.__dict__)
        existing.last_used = datetime.utcnow()
        self.logger.debug(f"Updated feature: {tag.name}")

    def _register_new_feature(self, tag: Tag):
        """Initialize a new feature entry."""
        tag.created_at = datetime.utcnow()
        tag.last_used = tag.created_at
        tag.usage_count = 1
        self.features[tag.name] = tag
        self.q_values[tag.name] = {}
        self.logger.info(f"New feature registered: {tag.name}")

    def update_q_value(self, feature_name: str, action: str, value: float):
        """Store learned Q-values for feature-action pairs."""
        if feature_name in self.q_values:
            self.q_values[feature_name][action] = value
            self.logger.debug(f"Updated Q-value for {feature_name}.{action} = {value}")

    def get_best_action(self, feature_name: str) -> Optional[str]:
        """Get highest Q-value action for a feature."""
        if feature_name not in self.q_values or not self.q_values[feature_name]:
            return None
            
        return max(self.q_values[feature_name].items(), key=lambda x: x[1])[0]

    def set_status(self, feature_name: str, status: str):
        """Update operational status of a feature."""
        if feature_name in self.features:
            self.features[feature_name].metadata["status"] = status
            self.logger.info(f"Set {feature_name} status to {status}")

    def get_by_type(self, feature_type: str) -> List[Tag]:
        """Retrieve features of specific type."""
        return [f for f in self.features.values() 
                if f.metadata.get("feature_type") == feature_type]