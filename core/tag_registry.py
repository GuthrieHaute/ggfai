# tag_registry.py - Hardened Tag Management Core
# written by DeepSeek Chat (honor call: The Taxonomist)

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import numpy as np
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, validator

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
    priority: float = 0.5  # Default to MEDIUM
    status: str = "active"  # Use string for serialization
    created_at: datetime = None
    last_used: datetime = None
    usage_count: int = 1
    similar_to: List[str] = []
    metadata: Dict[str, Any] = {}
    vector: Optional[np.ndarray] = None
    health: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        if 'last_used' not in data:
            data['last_used'] = datetime.utcnow()
        if 'health' not in data:
            data['health'] = {
                'stability_score': 1.0,
                'last_validated': datetime.utcnow(),
                'validation_errors': 0
            }
        super().__init__(**data)
        
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
        self.health['stability_score'] *= 0.99

    def to_dict(self) -> Dict[str, Any]:
        """Safe serialization with vector handling."""
        data = self.dict(exclude={'vector'})
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
                 nlp_model=None,
                 prune_days: int = 30,
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize hardened registry.
        
        Args:
            nlp_model: spaCy model for semantic analysis (loaded on demand)
            prune_days: Days before predictive pruning
            cache_size: LRU cache capacity
            similarity_threshold: Similarity cutoff (0-1)
        """
        self.logger = logging.getLogger("GGFAI.tag_registry")
        self.nlp = None
        self.nlp_model_name = nlp_model
        
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

    def _load_nlp(self):
        """Lazy-load NLP model when needed."""
        if self.nlp is None and self.nlp_model_name:
            try:
                import spacy
                self.nlp = spacy.load(self.nlp_model_name)
                if hasattr(self.nlp.meta, 'vectors') and self.nlp.meta['vectors']['vectors'] == 0:
                    self.logger.warning("Model has no word vectors, similarity will be limited")
            except ImportError:
                self.logger.warning("spaCy not installed, semantic similarity disabled")
            except Exception as e:
                self.logger.error(f"Failed to load NLP model: {e}")

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
                if existing.status != TagStatus.ACTIVE.value:
                    existing.status = TagStatus.ACTIVE.value
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

    def _add_to_indices(self, tag: Tag):
        """Update search indices for a tag."""
        self.category_index[tag.category].add(tag.name)
        self.category_index[f"{tag.category}.{tag.subcategory}"].add(tag.name)
        self.namespace_index[tag.namespace].add(tag.name)

    def _remove_from_indices(self, tag: Tag):
        """Remove tag from search indices."""
        if tag.name in self.category_index[tag.category]:
            self.category_index[tag.category].remove(tag.name)
        if tag.name in self.category_index[f"{tag.category}.{tag.subcategory}"]:
            self.category_index[f"{tag.category}.{tag.subcategory}"].remove(tag.name)
        if tag.name in self.namespace_index[tag.namespace]:
            self.namespace_index[tag.namespace].remove(tag.name)

    def _update_cache(self, key: str, value: Tag):
        """LRU cache update."""
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def optimize_cache(self):
        """Optimize cache based on usage patterns."""
        with self._lock:
            # Keep only most frequently used tags in cache
            sorted_tags = sorted(
                [(name, tag.usage_count) for name, tag in self.tags.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            self.cache.clear()
            for name, _ in sorted_tags[:self.cache_size]:
                if name in self.tags:
                    self.cache[name] = self.tags[name]

    def _check_similarity(self, new_tag: Tag):
        """Enhanced similarity detection with threading."""
        # Lazy-load NLP if needed
        if self.nlp is None and self.nlp_model_name:
            self._load_nlp()
            
        # Skip if no NLP model
        if self.nlp is None:
            return
            
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
        candidates_list = list(candidates)
        for i in range(0, len(candidates_list), batch_size):
            batch = candidates_list[i:i+batch_size]
            all_similar.extend(process_batch(batch))

        if all_similar:
            all_similar.sort(key=lambda x: x[1], reverse=True)
            new_tag.similar_to = [name for name, _ in all_similar[:5]]  # Top 5 matches

    def _compute_similarity(self, tag1: Tag, tag2: Tag) -> float:
        """Optimized similarity computation with caching."""
        # Skip if no NLP model
        if self.nlp is None:
            return 0.0
            
        # Lexical similarity
        try:
            from Levenshtein import ratio as levenshtein_ratio
            lexical = levenshtein_ratio(tag1.name.lower(), tag2.name.lower())
        except ImportError:
            # Fallback to basic similarity
            lexical = 1.0 if tag1.name.lower() == tag2.name.lower() else 0.0
            
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
            tag_items = list(self.tags.items())[:PRUNE_BATCH_SIZE]
            for name, tag in tag_items:
                if (tag.status == TagStatus.ACTIVE.value and 
                    tag.last_used < threshold and
                    tag.health['stability_score'] < 0.7):
                    
                    tag.status = TagStatus.ARCHIVED.value
                    self._remove_from_indices(tag)
                    pruned += 1
                    
        self.logger.info(f"Pruned {pruned} low-stability tags")
        return pruned

    def compress_tags(self) -> int:
        """Safe tag compression with conflict resolution."""
        compressed = 0
        with self._lock:
            # Process by priority to ensure important tags dominate
            for priority in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]:  # Enum values
                for name, tag in self.tags.items():
                    if (tag.status == TagStatus.ACTIVE.value and 
                        tag.priority >= priority and 
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
            if other and other.status == TagStatus.ACTIVE.value:
                score = (other.usage_count * 0.4 + 
                        other.priority * 0.3 +
                        other.health['stability_score'] * 0.3)
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
            source.status = TagStatus.MERGED.value
            source.metadata["merged_into"] = target.name
            
            # Update health
            target.health['stability_score'] = max(
                target.health['stability_score'],
                source.health['stability_score']
            )

    def validate_tags(self) -> int:
        """Comprehensive tag validation."""
        invalid = 0
        with self._lock:
            for name, tag in list(self.tags.items()):
                try:
                    # Basic validation
                    if not 0 <= tag.priority <= 1:
                        raise ValueError("Invalid priority")
                    if len(tag.name) > 256:
                        raise ValueError("Name too long")
                    
                    tag.health['validation_errors'] = 0
                except Exception as e:
                    tag.health['validation_errors'] += 1
                    if tag.health['validation_errors'] > 3:
                        tag.status = TagStatus.QUARANTINED.value
                        invalid += 1
                    self.logger.warning(f"Tag validation failed: {name} - {e}")
        return invalid

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Enhanced system metrics with health indicators."""
        with self._lock:
            active = [t for t in self.tags.values() if t.status == TagStatus.ACTIVE.value]
            archived = [t for t in self.tags.values() if t.status == TagStatus.ARCHIVED.value]
            merged = [t for t in self.tags.values() if t.status == TagStatus.MERGED.value]
            quarantined = [t for t in self.tags.values() if t.status == TagStatus.QUARANTINED.value]
            
            # Calculate health metrics
            avg_stability = sum(t.health['stability_score'] for t in active) / len(active) if active else 0
            high_priority = len([t for t in active if t.priority >= TagPriority.HIGH.value])
            
            return {
                "total_tags": len(self.tags),
                "active_tags": len(active),
                "archived_tags": len(archived),
                "merged_tags": len(merged),
                "quarantined_tags": len(quarantined),
                "avg_stability": avg_stability,
                "high_priority": high_priority,
                "cache_size": len(self.cache),
                "cache_hit_ratio": self.cache_hit_ratio if hasattr(self, 'cache_hit_ratio') else 0,
                "last_maintenance": datetime.utcnow().isoformat()
            }

    def get_tags_by_category(self, category: str, subcategory: str = None) -> List[Tag]:
        """Retrieve tags by category with optional subcategory filter."""
        with self._lock:
            if subcategory:
                key = f"{category}.{subcategory}"
                names = self.category_index.get(key, set())
            else:
                names = self.category_index.get(category, set())
                
            return [self.tags[name] for name in names if name in self.tags]

    def get_tags_by_namespace(self, namespace: str) -> List[Tag]:
        """Retrieve tags by namespace."""
        with self._lock:
            names = self.namespace_index.get(namespace, set())
            return [self.tags[name] for name in names if name in self.tags]

    def get_tag(self, name: str) -> Optional[Tag]:
        """Retrieve a tag by name with cache optimization."""
        with self._lock:
            # Check cache first
            if name in self.cache:
                tag = self.cache[name]
                tag.touch()
                return tag
                
            # Check main storage
            if name in self.tags:
                tag = self.tags[name]
                self._update_cache(name, tag)
                tag.touch()
                return tag
                
            return None

    def delete_tag(self, name: str) -> bool:
        """Safely delete a tag."""
        with self._lock:
            if name in self.tags:
                tag = self.tags[name]
                self._remove_from_indices(tag)
                del self.tags[name]
                if name in self.cache:
                    del self.cache[name]
                return True
            return False

    def clear(self):
        """Clear all tags and indices."""
        with self._lock:
            self.tags.clear()
            self.category_index.clear()
            self.namespace_index.clear()
            self.cache.clear()
            
    def get_tags(self, category: str = None, subcategory: str = None, namespace: str = None) -> List[Dict]:
        """
        Get all tags, optionally filtered by category, subcategory, or namespace.
        
        Args:
            category: Optional category filter
            subcategory: Optional subcategory filter
            namespace: Optional namespace filter
            
        Returns:
            List of tag dictionaries
        """
        with self._lock:
            if category and subcategory:
                key = f"{category}.{subcategory}"
                names = self.category_index.get(key, set())
            elif category:
                names = self.category_index.get(category, set())
            elif namespace:
                names = self.namespace_index.get(namespace, set())
            else:
                names = set(self.tags.keys())
                
            return [self.tags[name].to_dict() for name in names if name in self.tags]
            
    def add_tag(self, tag_data: Dict) -> str:
        """
        Add a tag to the registry.
        
        Args:
            tag_data: Dictionary containing tag data
            
        Returns:
            Tag ID if successful, None otherwise
        """
        try:
            return self.register_tag(**tag_data).name
        except Exception as e:
            self.logger.error(f"Failed to add tag: {str(e)}")
            return None
            
    def log_task_assignment(self, task_id: str, agent_id: str, bid_score: float, context: Dict) -> str:
        """
        Log a task assignment for analytics.
        
        Args:
            task_id: ID of the assigned task
            agent_id: ID of the agent assigned to the task
            bid_score: Bid score that won the assignment
            context: Context information for the assignment
            
        Returns:
            Tag ID if successful, None otherwise
        """
        try:
            tag_data = {
                "name": f"task_assignment_{task_id}_{agent_id}",
                "category": "task_assignment",
                "intent": "coordination",
                "task_id": task_id,
                "agent_id": agent_id,
                "bid_score": bid_score,
                "context": context,
                "timestamp": time.time()
            }
            return self.register_tag(**tag_data).name
        except Exception as e:
            self.logger.error(f"Failed to log task assignment: {str(e)}")
            return None
            
    def get_tags_for_action(self, action: str) -> Set[str]:
        """
        Get all tags associated with an action.
        
        Args:
            action: Action identifier
            
        Returns:
            Set of tag names associated with the action
        """
        with self._lock:
            result = set()
            for name, tag in self.tags.items():
                if tag.metadata.get("action") == action:
                    result.add(name)
            return result
            
    def get_matching_tags(self, action: str, context: Dict) -> List[Dict]:
        """
        Get tags that match both an action and context.
        
        Args:
            action: Action identifier
            context: Context dictionary to match
            
        Returns:
            List of matching tag dictionaries
        """
        with self._lock:
            matching = []
            for name, tag in self.tags.items():
                if tag.metadata.get("action") == action:
                    # Check context match
                    tag_context = tag.metadata.get("context", {})
                    if all(tag_context.get(k) == v for k, v in context.items() if k in tag_context):
                        matching.append(tag.to_dict())
            return matching