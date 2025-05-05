# feature_tracker.py - Feature Tag Management for GGFAI
# written by DeepSeek Chat (honor call: The Feature Librarian)

import logging
import threading
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json

from ..core.tag_registry import Tag

class FeatureTracker:
    """
    Tracks device capabilities and states through feature tags.
    
    Features represent:
    - Device capabilities (e.g., smart_bulb with capability=lighting)
    - Component states (e.g., on/off, busy)
    - Learned parameters (e.g., Q-values for actions)
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self.features: Dict[str, Tag] = {}  # name -> Tag
        self.q_values: Dict[str, Dict[str, float]] = {}  # feature_name -> {action: q_value}
        self.type_index: Dict[str, Set[str]] = {}  # feature_type -> set(feature_names)
        self.capability_index: Dict[str, Set[str]] = {}  # capability -> set(feature_names)
        self.logger = logging.getLogger("GGFAI.feature_tracker")
        self.logger.info("Feature tracker initialized")

    def add_tag(self, tag: Tag) -> str:
        """
        Register or update a feature tag.
        
        Args:
            tag: Feature tag to register/update
            
        Returns:
            Feature name
        """
        with self._lock:
            if not tag.metadata.get("feature_type"):
                tag.metadata["feature_type"] = "generic"

            if tag.name in self.features:
                self._update_feature(tag)
            else:
                self._register_new_feature(tag)
                
            return tag.name

    def _update_feature(self, tag: Tag):
        """Update existing feature metadata."""
        with self._lock:
            existing = self.features[tag.name]
            
            # Update indices if type or capability changes
            old_type = existing.metadata.get("feature_type")
            new_type = tag.metadata.get("feature_type")
            if old_type != new_type:
                self._update_type_index(tag.name, old_type, new_type)
                
            old_capability = existing.metadata.get("capability")
            new_capability = tag.metadata.get("capability")
            if old_capability != new_capability:
                self._update_capability_index(tag.name, old_capability, new_capability)
            
            # Update fields while preserving type
            for key, value in tag.dict().items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
                    
            # Update metadata
            existing.metadata.update(tag.metadata)
            existing.last_used = datetime.utcnow()
            existing.usage_count += 1
            self.logger.debug(f"Updated feature: {tag.name}")

    def _register_new_feature(self, tag: Tag):
        """Initialize a new feature entry."""
        with self._lock:
            # Ensure required fields
            if not hasattr(tag, 'created_at') or tag.created_at is None:
                tag.created_at = datetime.utcnow()
            if not hasattr(tag, 'last_used') or tag.last_used is None:
                tag.last_used = tag.created_at
            if not hasattr(tag, 'usage_count'):
                tag.usage_count = 1
                
            self.features[tag.name] = tag
            self.q_values[tag.name] = {}
            
            # Update indices
            feature_type = tag.metadata.get("feature_type", "generic")
            if feature_type not in self.type_index:
                self.type_index[feature_type] = set()
            self.type_index[feature_type].add(tag.name)
            
            capability = tag.metadata.get("capability")
            if capability:
                if capability not in self.capability_index:
                    self.capability_index[capability] = set()
                self.capability_index[capability].add(tag.name)
                
            self.logger.info(f"New feature registered: {tag.name}")

    def _update_type_index(self, feature_name: str, old_type: Optional[str], new_type: Optional[str]):
        """Update type index when feature type changes."""
        if old_type and old_type in self.type_index and feature_name in self.type_index[old_type]:
            self.type_index[old_type].remove(feature_name)
            
        if new_type:
            if new_type not in self.type_index:
                self.type_index[new_type] = set()
            self.type_index[new_type].add(feature_name)

    def _update_capability_index(self, feature_name: str, old_capability: Optional[str], new_capability: Optional[str]):
        """Update capability index when feature capability changes."""
        if old_capability and old_capability in self.capability_index and feature_name in self.capability_index[old_capability]:
            self.capability_index[old_capability].remove(feature_name)
            
        if new_capability:
            if new_capability not in self.capability_index:
                self.capability_index[new_capability] = set()
            self.capability_index[new_capability].add(feature_name)

    def update_q_value(self, feature_name: str, action: str, value: float):
        """
        Store learned Q-values for feature-action pairs.
        
        Args:
            feature_name: Name of the feature
            action: Action name
            value: Q-value (expected utility)
        """
        with self._lock:
            if feature_name in self.q_values:
                self.q_values[feature_name][action] = value
                
                # Also update in metadata for persistence
                if feature_name in self.features:
                    if "q_values" not in self.features[feature_name].metadata:
                        self.features[feature_name].metadata["q_values"] = {}
                    self.features[feature_name].metadata["q_values"][action] = value
                    
                self.logger.debug(f"Updated Q-value for {feature_name}.{action} = {value}")

    def get_best_action(self, feature_name: str) -> Optional[str]:
        """
        Get highest Q-value action for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Action with highest Q-value, or None if no actions
        """
        with self._lock:
            if feature_name not in self.q_values or not self.q_values[feature_name]:
                return None
                
            return max(self.q_values[feature_name].items(), key=lambda x: x[1])[0]

    def get_q_value(self, feature_name: str, action: str) -> float:
        """
        Get Q-value for a feature-action pair.
        
        Args:
            feature_name: Name of the feature
            action: Action name
            
        Returns:
            Q-value, or 0.0 if not found
        """
        with self._lock:
            if feature_name in self.q_values and action in self.q_values[feature_name]:
                return self.q_values[feature_name][action]
            return 0.0

    def set_status(self, feature_name: str, status: str):
        """
        Update operational status of a feature.
        
        Args:
            feature_name: Name of the feature
            status: Status value (e.g., "on", "off", "busy")
        """
        with self._lock:
            if feature_name in self.features:
                self.features[feature_name].metadata["status"] = status
                self.features[feature_name].last_used = datetime.utcnow()
                self.logger.info(f"Set {feature_name} status to {status}")

    def get_status(self, feature_name: str) -> Optional[str]:
        """
        Get operational status of a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Status value, or None if not found
        """
        with self._lock:
            if feature_name in self.features:
                return self.features[feature_name].metadata.get("status")
            return None

    def get_by_type(self, feature_type: str) -> List[Tag]:
        """
        Retrieve features of specific type.
        
        Args:
            feature_type: Type of features to retrieve
            
        Returns:
            List of matching feature tags
        """
        with self._lock:
            if feature_type in self.type_index:
                return [self.features[name] for name in self.type_index[feature_type] 
                        if name in self.features]
            return []

    def get_by_capability(self, capability: str) -> List[Tag]:
        """
        Retrieve features with specific capability.
        
        Args:
            capability: Capability to match
            
        Returns:
            List of matching feature tags
        """
        with self._lock:
            if capability in self.capability_index:
                return [self.features[name] for name in self.capability_index[capability] 
                        if name in self.features]
            return []

    def get_tag(self, feature_name: str) -> Optional[Tag]:
        """
        Get feature tag by name.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature tag, or None if not found
        """
        with self._lock:
            return self.features.get(feature_name)

    def get_all_features(self) -> List[Tag]:
        """
        Get all registered features.
        
        Returns:
            List of all feature tags
        """
        with self._lock:
            return list(self.features.values())

    def lock_feature(self, feature_name: str, owner: str) -> bool:
        """
        Lock a feature for exclusive use.
        
        Args:
            feature_name: Name of the feature
            owner: ID of the locking entity
            
        Returns:
            True if lock acquired, False if already locked
        """
        with self._lock:
            if feature_name in self.features:
                feature = self.features[feature_name]
                if feature.metadata.get("locked_by"):
                    return False
                    
                feature.metadata["locked_by"] = owner
                feature.metadata["locked_at"] = datetime.utcnow().isoformat()
                feature.metadata["status"] = "busy"
                return True
            return False

    def unlock_feature(self, feature_name: str, owner: str) -> bool:
        """
        Release lock on a feature.
        
        Args:
            feature_name: Name of the feature
            owner: ID of the locking entity
            
        Returns:
            True if lock released, False if not locked by owner
        """
        with self._lock:
            if feature_name in self.features:
                feature = self.features[feature_name]
                if feature.metadata.get("locked_by") == owner:
                    feature.metadata.pop("locked_by", None)
                    feature.metadata.pop("locked_at", None)
                    feature.metadata["status"] = "idle"
                    return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "total_features": len(self.features),
                "types": {t: len(names) for t, names in self.type_index.items()},
                "capabilities": {c: len(names) for c, names in self.capability_index.items()},
                "locked_features": len([f for f in self.features.values() 
                                      if f.metadata.get("locked_by")]),
                "busy_features": len([f for f in self.features.values() 
                                    if f.metadata.get("status") == "busy"])
            }

    def export_state(self) -> str:
        """
        Export state for persistence.
        
        Returns:
            JSON string of state
        """
        with self._lock:
            return json.dumps({
                "features": [f.to_dict() for f in self.features.values()],
                "q_values": self.q_values
            })

    def load_state(self, state: str):
        """
        Load state from persistence.
        
        Args:
            state: JSON string of state
        """
        with self._lock:
            data = json.loads(state)
            
            # Clear existing state
            self.features.clear()
            self.q_values.clear()
            self.type_index.clear()
            self.capability_index.clear()
            
            # Load features
            for feature_data in data["features"]:
                tag = Tag(**feature_data)
                self.features[tag.name] = tag
                
                # Rebuild indices
                feature_type = tag.metadata.get("feature_type", "generic")
                if feature_type not in self.type_index:
                    self.type_index[feature_type] = set()
                self.type_index[feature_type].add(tag.name)
                
                capability = tag.metadata.get("capability")
                if capability:
                    if capability not in self.capability_index:
                        self.capability_index[capability] = set()
                    self.capability_index[capability].add(tag.name)
            
            # Load Q-values
            self.q_values = data["q_values"]