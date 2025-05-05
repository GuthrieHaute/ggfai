"""
Contextual Fusion - Hardware-optimized context merging system
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .config_system import ConfigSystem, HardwareTier

logger = logging.getLogger("GGFAI.core_framework.fusion")

class ContextualFusion:
    """
    Hardware-optimized contextual fusion system that uses statistical models
    instead of neural networks for better compatibility with mid-range PCs.
    """
    
    def __init__(self):
        self.config = ConfigSystem()
        self._initialize_models()
        
    def _initialize_models(self) -> None:
        """Initialize appropriate models based on hardware tier"""
        self.use_pca = self.config.hardware_profile.tier in {HardwareTier.MID, HardwareTier.HIGH}
        self.use_advanced_stats = self.config.hardware_profile.tier == HardwareTier.HIGH
        
        # Initialize basic components
        self.scaler = StandardScaler()
        
        # Initialize PCA for dimension reduction on capable systems
        self.pca = PCA(n_components=0.95) if self.use_pca else None
        
        # Storage for context statistics
        self.context_stats = {
            "mean": {},
            "variance": {},
            "correlations": {}
        }
        
    def fuse_contexts(
        self,
        contexts: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Fuse multiple contexts into a single context, using methods
        appropriate for the current hardware tier.
        """
        if not contexts:
            return {}
            
        if len(contexts) == 1:
            return contexts[0].copy()
        
        # Normalize weights
        if weights is None:
            weights = [1.0] * len(contexts)
        weights = np.array(weights) / sum(weights)
        
        # Convert contexts to feature vectors
        features = self._contexts_to_features(contexts)
        
        # Apply statistical fusion based on hardware tier
        if self.config.hardware_profile.tier == HardwareTier.HIGH:
            return self._advanced_fusion(features, contexts, weights)
        elif self.config.hardware_profile.tier == HardwareTier.MID:
            return self._mid_range_fusion(features, contexts, weights)
        else:
            return self._basic_fusion(features, contexts, weights)
    
    def _contexts_to_features(self, contexts: List[Dict[str, Any]]) -> np.ndarray:
        """Convert context dictionaries to feature vectors"""
        # Extract numerical features
        feature_lists = []
        for context in contexts:
            features = []
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    features.append(value)
                elif isinstance(value, bool):
                    features.append(1.0 if value else 0.0)
                else:
                    # Skip non-numerical values
                    continue
            feature_lists.append(features)
        
        # Convert to numpy array with padding
        max_len = max(len(f) for f in feature_lists)
        padded = np.zeros((len(contexts), max_len))
        for i, features in enumerate(feature_lists):
            padded[i, :len(features)] = features
            
        return padded
    
    def _advanced_fusion(
        self,
        features: np.ndarray,
        contexts: List[Dict[str, Any]],
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """Advanced fusion for high-end systems"""
        # Scale features
        scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        if self.pca is not None:
            reduced = self.pca.fit_transform(scaled)
            # Project back to original space
            features = self.pca.inverse_transform(reduced)
            features = self.scaler.inverse_transform(features)
        
        # Update statistics
        self._update_statistics(features)
        
        # Weighted combination with correlation awareness
        fused_features = np.average(features, weights=weights, axis=0)
        
        # Reconstruct context dictionary
        return self._features_to_context(fused_features, contexts)
    
    def _mid_range_fusion(
        self,
        features: np.ndarray,
        contexts: List[Dict[str, Any]],
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """Optimized fusion for mid-range gaming PCs"""
        # Scale features
        scaled = self.scaler.fit_transform(features)
        
        # Simple dimensionality reduction if needed
        if self.pca is not None and features.shape[1] > 32:
            reduced = self.pca.fit_transform(scaled)
            # Project back to original space
            features = self.pca.inverse_transform(reduced)
            features = self.scaler.inverse_transform(features)
        
        # Weighted average
        fused_features = np.average(features, weights=weights, axis=0)
        
        # Reconstruct context dictionary
        return self._features_to_context(fused_features, contexts)
    
    def _basic_fusion(
        self,
        features: np.ndarray,
        contexts: List[Dict[str, Any]],
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """Simple fusion for low-end systems"""
        # Just do a weighted average
        fused_features = np.average(features, weights=weights, axis=0)
        return self._features_to_context(fused_features, contexts)
    
    def _update_statistics(self, features: np.ndarray) -> None:
        """Update context statistics for high-end systems"""
        if not self.use_advanced_stats:
            return
            
        # Update mean and variance
        self.context_stats["mean"] = np.mean(features, axis=0)
        self.context_stats["variance"] = np.var(features, axis=0)
        
        # Update correlations if we have enough samples
        if features.shape[0] > 1:
            self.context_stats["correlations"] = np.corrcoef(features.T)
    
    def _features_to_context(
        self,
        features: np.ndarray,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert fused feature vector back to context dictionary"""
        result = {}
        
        # Get all unique keys from input contexts
        all_keys = set()
        for context in contexts:
            all_keys.update(context.keys())
        
        # Track numerical features
        feature_idx = 0
        
        for key in all_keys:
            # Find first non-None value for this key
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            if not values:
                continue
                
            first_value = values[0]
            
            if isinstance(first_value, (int, float)):
                # Use fused numerical value
                result[key] = float(features[feature_idx])
                feature_idx += 1
            elif isinstance(first_value, bool):
                # Convert fused value to boolean
                result[key] = bool(round(features[feature_idx]))
                feature_idx += 1
            else:
                # For non-numerical values, use most common value
                from collections import Counter
                counts = Counter(values)
                result[key] = counts.most_common(1)[0][0]
        
        return result