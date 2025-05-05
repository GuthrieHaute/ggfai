# intent_tracker.py - Industrial-Grade Intent Management
# written by DeepSeek Chat (honor call: The Intent Archivist)

import logging
import threading
import time
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

from ..core.tag_registry import Tag

# Constants
MAX_INTENTS = 10000  # Safety limit
PRIORITY_THRESHOLD = 0.7  # Default high-priority cutoff
ARCHIVE_DAYS = 30  # Default inactivity period
INTENT_EXPIRY = 86400  # 24h in seconds

class IntentStatus(Enum):
    """Enhanced intent lifecycle states."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class IntentTracker:
    """
    Hardened intent tracking system with:
    - Thread-safe operations
    - Intent deduplication
    - Priority-based scheduling
    - Expiry and cleanup
    - Failure resilience
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self.logger = logging.getLogger("GGFAI.intent_tracker")
        self.intents: Dict[str, Tag] = {}  # name -> Tag
        self.priority_queue: List[Tuple[float, str]] = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Start maintenance tasks
        self._start_maintenance()
        self.logger.info("Hardened intent tracker initialized")

    def _start_maintenance(self):
        """Background maintenance tasks."""
        def maintenance_loop():
            while True:
                try:
                    self._clean_expired_intents()
                    self._rebalance_priority_queue()
                except Exception as e:
                    self.logger.error(f"Maintenance failed: {e}")
                time.sleep(3600)  # Run hourly
        
        threading.Thread(target=maintenance_loop, daemon=True).start()

    def add_tag(self, tag: Tag) -> str:
        """
        Thread-safe intent registration with deduplication.
        
        Args:
            tag: Intent tag to add/update
            
        Returns:
            Intent ID (name hash)
        """
        with self._lock:
            # Safety checks
            if len(self.intents) >= MAX_INTENTS:
                self._enforce_capacity()
                
            if not tag.name or not tag.intent:
                raise ValueError("Invalid intent tag")

            # Create intent ID
            intent_id = self._generate_id(tag)
            
            # Update existing or add new
            if intent_id in self.intents:
                existing = self.intents[intent_id]
                # Update fields while preserving type
                for key, value in tag.dict().items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.last_used = datetime.utcnow()
                existing.usage_count += 1
                self.logger.debug(f"Updated intent: {tag.name}")
            else:
                # Ensure required fields
                if not hasattr(tag, 'created_at') or tag.created_at is None:
                    tag.created_at = datetime.utcnow()
                if not hasattr(tag, 'last_used') or tag.last_used is None:
                    tag.last_used = tag.created_at
                if not hasattr(tag, 'usage_count'):
                    tag.usage_count = 1
                tag.status = IntentStatus.ACTIVE.value
                self.intents[intent_id] = tag
                self._add_to_priority_queue(tag)
                self.logger.info(f"New intent registered: {tag.name}")

            return intent_id

    def _generate_id(self, tag: Tag) -> str:
        """Create deterministic intent ID."""
        payload = f"{tag.name}:{tag.intent}:{tag.category}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _add_to_priority_queue(self, tag: Tag):
        """Maintain sorted priority queue."""
        with self._lock:
            self.priority_queue.append((tag.priority, self._generate_id(tag)))
            self.priority_queue.sort(reverse=True)

    def _enforce_capacity(self):
        """Auto-clean when reaching capacity limits."""
        with self._lock:
            # Archive oldest low-priority intents first
            to_archive = sorted(
                self.intents.values(),
                key=lambda x: (x.priority, x.last_used)
            )[:100]  # Batch of 100
            
            for intent in to_archive:
                intent.status = IntentStatus.ARCHIVED.value
                
            self.logger.warning(f"Capacity enforcement archived {len(to_archive)} intents")

    def get_tag(self, intent_id: str) -> Optional[Tag]:
        """Thread-safe intent retrieval."""
        with self._lock:
            return self.intents.get(intent_id)

    def get_tags(self, include_archived: bool = False) -> List[Tag]:
        """Get all intents with optional archived ones."""
        with self._lock:
            if include_archived:
                return list(self.intents.values())
            return [t for t in self.intents.values() 
                    if t.status == IntentStatus.ACTIVE.value]

    def get_by_category(self, category: str) -> List[Tag]:
        """Get active intents by category."""
        with self._lock:
            return [t for t in self.intents.values()
                    if t.category == category and 
                    t.status == IntentStatus.ACTIVE.value]

    def get_current_priority_intents(self, 
                                   threshold: float = PRIORITY_THRESHOLD,
                                   limit: int = 100) -> List[Tag]:
        """
        Get high-priority intents with thread-safe queue access.
        
        Args:
            threshold: Minimum priority (0-1)
            limit: Maximum results to return
            
        Returns:
            Sorted list of high-priority intents
        """
        with self._lock:
            results = []
            for priority, intent_id in self.priority_queue:
                if priority < threshold or len(results) >= limit:
                    break
                intent = self.intents.get(intent_id)
                if intent and intent.status == IntentStatus.ACTIVE.value:
                    results.append(intent)
            return results

    def _clean_expired_intents(self):
        """Background cleanup of expired intents."""
        with self._lock:
            threshold = datetime.utcnow() - timedelta(seconds=INTENT_EXPIRY)
            expired = [i for i in self.intents.values() 
                      if i.last_used < threshold and
                      i.status == IntentStatus.ACTIVE.value]
                      
            for intent in expired:
                intent.status = IntentStatus.ARCHIVED.value
                
            if expired:
                self.logger.info(f"Cleaned {len(expired)} expired intents")

    def _rebalance_priority_queue(self):
        """Periodic queue optimization."""
        with self._lock:
            # Remove archived/completed intents
            self.priority_queue = [
                (p, i) for p, i in self.priority_queue
                if i in self.intents and self.intents[i].status == IntentStatus.ACTIVE.value
            ]
            # Re-sort by priority
            self.priority_queue.sort(reverse=True)

    def mark_completed(self, intent_id: str):
        """Safely mark intent as completed."""
        with self._lock:
            if intent_id in self.intents:
                self.intents[intent_id].status = IntentStatus.COMPLETED.value
                self._rebalance_priority_queue()

    def mark_failed(self, intent_id: str, reason: str = ""):
        """Track intent failures with diagnostic info."""
        with self._lock:
            if intent_id in self.intents:
                intent = self.intents[intent_id]
                intent.status = IntentStatus.FAILED.value
                intent.metadata["failure_reason"] = reason
                intent.metadata["failure_time"] = datetime.utcnow().isoformat()
                self._rebalance_priority_queue()

    def get_stats(self) -> Dict[str, Any]:
        """System health metrics."""
        with self._lock:
            active = [t for t in self.intents.values() 
                     if t.status == IntentStatus.ACTIVE.value]
            return {
                "total_intents": len(self.intents),
                "active_intents": len(active),
                "avg_priority": sum(t.priority for t in active) / len(active) if active else 0,
                "priority_distribution": self._get_priority_distribution(),
                "oldest_active": min((t.last_used for t in active), default=None),
                "queue_length": len(self.priority_queue)
            }

    def _get_priority_distribution(self) -> Dict[str, int]:
        """Priority histogram for monitoring."""
        with self._lock:
            active = [t for t in self.intents.values() 
                     if t.status == IntentStatus.ACTIVE.value]
            return {
                "critical": len([t for t in active if t.priority >= 0.9]),
                "high": len([t for t in active if 0.7 <= t.priority < 0.9]),
                "medium": len([t for t in active if 0.5 <= t.priority < 0.7]),
                "low": len([t for t in active if t.priority < 0.5])
            }

    def export_state(self) -> str:
        """Snapshot for persistence (thread-safe)."""
        with self._lock:
            return json.dumps({
                "intents": [t.to_dict() for t in self.intents.values()],
                "priority_queue": self.priority_queue
            })

    def load_state(self, state: str):
        """Restore from snapshot (thread-safe)."""
        with self._lock:
            data = json.loads(state)
            self.intents = {
                self._generate_id(Tag(**t)): Tag(**t) 
                for t in data["intents"]
            }
            self.priority_queue = data["priority_queue"]