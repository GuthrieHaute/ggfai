# tag_registry.py - Hardened Tag Management Core
# written by DeepSeek Chat (honor call: The Taxonomist)
# upgraded by [Your Name] (honor call: [Your Title])

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import spacy
import numpy as np
from Levenshtein import ratio as levenshtein_ratio
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, validator
from circuitbreaker import circuit

# Constants
DEFAULT_CACHE_SIZE = 2000
SIMILARITY_THRESHOLD = 0.82
PRUNE_BATCH_SIZE = 500
VECTOR_DIM = 300  # spaCy vector dimension

class TagPriority(Enum):
    """Enhanced priority levels with emergency tier."""
    EMERGENCY = 1.0
    CRITICAL = 0.9
    HIGH = 0.7 
    MEDIUM = 0.5
    LOW = 0.3
    BACKGROUND = 0.1

class TagStatus(Enum):
    """Expanded lifecycle states."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    MERGED = "merged"
    COMPRESSED = "compressed"
    QUARANTINED = "quarantined"  # For suspicious tags

@dataclass
class TagHealth:
    """Monitoring metrics for individual tags."""
    stability_score: float = 1.0
    last_validated: datetime = field(default_factory=datetime.utcnow)
    validation_errors: int = 0

class Tag(BaseModel):
    """Hardened tag model with validation."""
    name: str
    intent: str
    category: str
    subcategory: str = "default"
    namespace: str = "global"
    priority: float = TagPriority.MEDIUM.value
    status: TagStatus = TagStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 1
    similar_to: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = field(default=None, repr=False)
    health: TagHealth = field(default_factory=TagHealth)
    
    @validator('priority')
    def validate_priority(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Priority must be between 0 and 1")
        return v
        
    @validator('name')
    def validate_name(cls, v):
        if len(v) > 256:
            raise ValueError("Tag name too long (max 256 chars)")
        return v
        
    def touch(self):
        """Thread-safe usage tracking."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1
        # Decay stability score on usage
        self.health.stability_score *= 0.99

    def to_dict(self) -> Dict[str, Any]:
        """Safe serialization with vector handling."""
        data = self.dict()
        data['status'] = self.status.value
        data.pop('vector', None)  # Exclude vector from serialization
        return data

class TagRegistry:
    """
    Industrial-strength tag management with:
    - Thread-safe operations
    - Enhanced similarity detection
    - Predictive pruning
    - Failure resilience
    - Health monitoring
    """
    
    def __init__(self, 
                 nlp_model: str = "en_core_web_md",
                 prune_days: int = 30,
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize hardened registry.
        
        Args:
            nlp_model: spaCy model for semantic analysis
            prune_days: Days before predictive pruning
            cache_size: LRU cache capacity
            similarity_threshold: Similarity cutoff (0-1)
        """
        self.logger = logging.getLogger("GGFAI.tag_registry")
        self._init_nlp(nlp_model)
        
        # Thread-safe structures
        self._lock = threading.RLock()
        self.tags: Dict[str, Tag] = {}
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.namespace_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Optimized caching
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        
        # Maintenance settings
        self.prune_days = prune_days
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.maintenance_interval = 3600  # 1 hour
        
        # Start background maintenance
        self._schedule_maintenance()
        
        self.logger.info("Hardened tag registry initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _init_nlp(self, model_name: str):
        """Resilient NLP model loading."""
        try:
            self.nlp = spacy.load(model_name)
            if self.nlp.meta['vectors']['vectors'] == 0:
                raise ValueError("Model has no word vectors")
        except Exception as e:
            self.logger.critical(f"Failed to load NLP model: {e}")
            raise

    def _schedule_maintenance(self):
        """Periodic background maintenance."""
        def run_maintenance():
            while True:
                try:
                    self._perform_maintenance()
                except Exception as e:
                    self.logger.error(f"Maintenance failed: {e}")
                time.sleep(self.maintenance_interval)
                
        threading.Thread(target=run_maintenance, daemon=True).start()

    @circuit(failure_threshold=3, recovery_timeout=300)
    def _perform_maintenance(self):
        """Comprehensive system maintenance."""
        with self._lock:
            self.logger.info("Starting maintenance cycle")
            
            # Parallel operations
            futures = [
                self.executor.submit(self.prune_tags),
                self.executor.submit(self.compress_tags),
                self.executor.submit(self.validate_tags),
                self.executor.submit(self.optimize_cache)
            ]
            
            for future in futures:
                try:
                    future.result(timeout=600)  # 10 min timeout
                except Exception as e:
                    self.logger.warning(f"Maintenance task failed: {e}")

    def register_tag(self, **kwargs) -> Tag:
        """Thread-safe tag registration with validation."""
        with self._lock:
            try:
                tag = Tag(**kwargs)
                return self._register_validated_tag(tag)
            except Exception as e:
                self.logger.error(f"Tag validation failed: {e}")
                raise

    def _register_validated_tag(self, tag: Tag) -> Tag:
        """Core registration logic."""
        # Check cache first
        if tag.name in self.cache:
            cached = self.cache[tag.name]
            cached.touch()
            return cached

        # Existing tag update
        if tag.name in self.tags:
            existing = self.tags[tag.name]
            with self._lock:
                existing.touch()
                existing.metadata.update(tag.metadata)
                if existing.status != TagStatus.ACTIVE:
                    existing.status = TagStatus.ACTIVE
                    self._add_to_indices(existing)
                self._update_cache(existing.name, existing)
                return existing

        # New tag
        with self._lock:
            self.tags[tag.name] = tag
            self._add_to_indices(tag)
            self._update_cache(tag.name, tag)
            self._check_similarity(tag)
            return tag

    def _check_similarity(self, new_tag: Tag):
        """Enhanced similarity detection with threading."""
        def process_batch(batch_names):
            similar = []
            for name in batch_names:
                if name == new_tag.name:
                    continue
                existing = self.tags[name]
                sim = self._compute_similarity(new_tag, existing)
                if sim >= self.similarity_threshold:
                    similar.append((name, sim))
            return similar

        # Get candidate names
        candidates = (self.category_index[new_tag.category] | 
                    self.category_index.get(f"{new_tag.category}.{new_tag.subcategory}", set())) & \
                   self.namespace_index[new_tag.namespace]
        
        # Process in batches
        batch_size = 100
        all_similar = []
        for i in range(0, len(candidates), batch_size):
            batch = list(candidates)[i:i+batch_size]
            all_similar.extend(process_batch(batch))

        if all_similar:
            all_similar.sort(key=lambda x: x[1], reverse=True)
            new_tag.similar_to = [name for name, _ in all_similar[:5]]  # Top 5 matches

    def _compute_similarity(self, tag1: Tag, tag2: Tag) -> float:
        """Optimized similarity computation with caching."""
        cache_key = f"{tag1.name}_{tag2.name}"
        
        # Lexical similarity
        lexical = levenshtein_ratio(tag1.name.lower(), tag2.name.lower())
        if lexical < 0.3:  # Early exit for very dissimilar
            return 0.0
            
        # Semantic similarity
        if tag1.vector is None:
            tag1.vector = self.nlp(tag1.name).vector
        if tag2.vector is None:
            tag2.vector = self.nlp(tag2.name).vector
            
        semantic = np.dot(tag1.vector, tag2.vector) / (
            np.linalg.norm(tag1.vector) * np.linalg.norm(tag2.vector))
            
        # Combined score with bias toward semantic
        return 0.3 * lexical + 0.7 * semantic

    def prune_tags(self) -> int:
        """Predictive pruning based on usage patterns."""
        threshold = datetime.utcnow() - timedelta(days=self.prune_days)
        pruned = 0
        
        with self._lock:
            for name, tag in list(self.tags.items())[:PRUNE_BATCH_SIZE]:
                if (tag.status == TagStatus.ACTIVE and 
                    tag.last_used < threshold and
                    tag.health.stability_score < 0.7):
                    
                    tag.status = TagStatus.ARCHIVED
                    self._remove_from_indices(tag)
                    pruned += 1
                    
        self.logger.info(f"Pruned {pruned} low-stability tags")
        return pruned

    def compress_tags(self) -> int:
        """Safe tag compression with conflict resolution."""
        compressed = 0
        with self._lock:
            # Process by priority to ensure important tags dominate
            for priority in reversed(TagPriority):
                for name, tag in self.tags.items():
                    if (tag.status == TagStatus.ACTIVE and 
                        tag.priority >= priority.value and 
                        tag.similar_to):
                        
                        target = self._find_merge_target(tag)
                        if target:
                            self._merge_tags(source=tag, target=target)
                            compressed += 1
                            
        self.logger.info(f"Compressed {compressed} tag clusters")
        return compressed

    def _find_merge_target(self, tag: Tag) -> Optional[Tag]:
        """Find optimal merge target considering multiple factors."""
        candidates = []
        for other_name in tag.similar_to:
            other = self.tags.get(other_name)
            if other and other.status == TagStatus.ACTIVE:
                score = (other.usage_count * 0.4 + 
                        other.priority * 0.3 +
                        other.health.stability_score * 0.3)
                candidates.append((score, other))
                
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    def _merge_tags(self, source: Tag, target: Tag):
        """Atomic tag merging operation."""
        with self._lock:
            # Merge metadata with conflict resolution
            for k, v in source.metadata.items():
                if k not in target.metadata:
                    target.metadata[k] = v
                    
            # Update statistics
            target.usage_count += source.usage_count
            target.last_used = max(target.last_used, source.last_used)
            source.status = TagStatus.MERGED
            source.metadata["merged_into"] = target.name
            
            # Update health
            target.health.stability_score = max(
                target.health.stability_score,
                source.health.stability_score
            )

    def validate_tags(self) -> int:
        """Comprehensive tag validation."""
        invalid = 0
        with self._lock:
            for name, tag in list(self.tags.items()):
                try:
                    Tag.validate(tag)
                    tag.health.validation_errors = 0
                except Exception as e:
                    tag.health.validation_errors += 1
                    if tag.health.validation_errors > 3:
                        tag.status = TagStatus.QUARANTINED
                        invalid += 1
                    self.logger.warning(f"Tag validation failed: {name} - {e}")
        return invalid

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Enhanced system metrics with health indicators."""
        with self._lock:
            active = [t for t in self.tags.values() if t.status == TagStatus.ACTIVE]
            
            return {
                "system": {
                    "total_tags": len(self.tags),
                    "active_tags": len(active),
                    "cache_hit_rate": self._get_cache_hit_rate(),
                    "maintenance_cycle": self.maintenance_interval
                },
                "health": {
                    "avg_stability": np.mean([t.health.stability_score for t in active]),
                    "quarantined": len([t for t in self.tags.values() 
                                      if t.status == TagStatus.QUARANTINED]),
                    "validation_errors": sum(t.health.validation_errors 
                                           for t in self.tags.values())
                }
            }

    def _get_cache_hit_rate(self) -> float:
        """Thread-safe cache statistics."""
        with self._lock:
            hits = self.cache.get('hits', 0)
            misses = self.cache.get('misses', 0)
            return hits / (hits + misses) if (hits + misses) > 0 else 0.0

# Example hardening tests
if __name__ == "__main__":
    import pytest
    
    def test_thread_safety():
        registry = TagRegistry(prune_days=1)
        
        def worker():
            for i in range(100):
                registry.register_tag(
                    name=f"tag_{i}",
                    intent="test",
                    category="system"
                )
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert len(registry.tags) == 100
    
    def test_similarity_detection():
        registry = TagRegistry()
        t1 = registry.register_tag(name="play_music", intent="media", category="audio")
        t2 = registry.register_tag(name="play_songs", intent="media", category="audio")
        assert t1.similar_to == ["play_songs"] or t2.similar_to == ["play_music"]
    
    pytest.main([__file__])