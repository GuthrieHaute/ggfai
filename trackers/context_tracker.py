# context_tracker.py - Context Tag Management for GGFAI
# written by DeepSeek Chat (honor call: The Context Sentinel)

import logging
import threading
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json

from ..core.tag_registry import Tag, TagStatus

class ContextTracker:
    """
    Manages situational context tags for the system.
    
    Context tags represent:
    - Environmental state (e.g., time_of_day, user_present)
    - System status (e.g., system_mode, error_state)
    - Coordination information (e.g., task_claim, resource_lock)
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self.contexts: Dict[str, Tag] = {}  # context_key -> Tag
        self.task_claims: Dict[str, str] = {}  # task_id -> agent_id
        self.category_index: Dict[str, Set[str]] = {}  # category -> set(context_keys)
        self.logger = logging.getLogger("GGFAI.context_tracker")
        self.logger.info("Context tracker initialized")

    def add_tag(self, tag: Tag) -> str:
        """
        Add or update a context tag.
        
        Args:
            tag: Context tag to add/update
            
        Returns:
            Context key
        """
        with self._lock:
            if tag.name.startswith("task_claim"):
                return self._handle_task_claim(tag)
                
            if tag.name in self.contexts:
                self._update_context(tag)
            else:
                self._register_new_context(tag)
                
            return tag.name

    def _handle_task_claim(self, tag: Tag) -> str:
        """Special handling for task claim contexts."""
        with self._lock:
            task_id = tag.metadata.get("task_id")
            agent_id = tag.metadata.get("agent_id")
            
            if not task_id or not agent_id:
                self.logger.error("Invalid task claim: missing task_id or agent_id")
                return ""
                
            self.task_claims[task_id] = agent_id
            
            # Also store as regular context
            claim_key = f"task_claim_{task_id}"
            if claim_key in self.contexts:
                self.contexts[claim_key].metadata.update(tag.metadata)
                self.contexts[claim_key].last_used = datetime.utcnow()
            else:
                tag.name = claim_key
                self._register_new_context(tag)
                
            self.logger.info(f"Task {task_id} claimed by {agent_id}")
            return claim_key

    def _update_context(self, tag: Tag):
        """Update existing context."""
        with self._lock:
            existing = self.contexts[tag.name]
            
            # Update category index if needed
            old_category = existing.category
            new_category = tag.category
            if old_category != new_category:
                self._update_category_index(tag.name, old_category, new_category)
            
            # Update fields while preserving type
            for key, value in tag.dict().items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
                    
            # Update metadata
            existing.metadata.update(tag.metadata)
            existing.last_used = datetime.utcnow()
            existing.usage_count += 1
            self.logger.debug(f"Updated context: {tag.name}")

    def _register_new_context(self, tag: Tag):
        """Initialize new context."""
        with self._lock:
            # Ensure required fields
            if not hasattr(tag, 'created_at') or tag.created_at is None:
                tag.created_at = datetime.utcnow()
            if not hasattr(tag, 'last_used') or tag.last_used is None:
                tag.last_used = tag.created_at
            if not hasattr(tag, 'usage_count'):
                tag.usage_count = 1
            if not hasattr(tag, 'status') or tag.status is None:
                tag.status = TagStatus.ACTIVE.value
                
            self.contexts[tag.name] = tag
            
            # Update category index
            if tag.category:
                if tag.category not in self.category_index:
                    self.category_index[tag.category] = set()
                self.category_index[tag.category].add(tag.name)
                
            self.logger.info(f"New context registered: {tag.name}")

    def _update_category_index(self, context_name: str, old_category: Optional[str], new_category: Optional[str]):
        """Update category index when context category changes."""
        if old_category and old_category in self.category_index and context_name in self.category_index[old_category]:
            self.category_index[old_category].remove(context_name)
            
        if new_category:
            if new_category not in self.category_index:
                self.category_index[new_category] = set()
            self.category_index[new_category].add(context_name)

    def get_current_context(self, key: str) -> Optional[Tag]:
        """
        Retrieve active context by key.
        
        Args:
            key: Context key
            
        Returns:
            Context tag if active, None otherwise
        """
        with self._lock:
            ctx = self.contexts.get(key)
            return ctx if ctx and ctx.status == TagStatus.ACTIVE.value else None

    def get_all_active(self) -> Dict[str, Tag]:
        """
        Return all active contexts.
        
        Returns:
            Dictionary of active context tags
        """
        with self._lock:
            return {k: v for k, v in self.contexts.items() 
                    if v.status == TagStatus.ACTIVE.value}

    def get_by_category(self, category: str) -> List[Tag]:
        """
        Get contexts by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of matching context tags
        """
        with self._lock:
            if category in self.category_index:
                return [self.contexts[name] for name in self.category_index[category] 
                        if name in self.contexts and 
                        self.contexts[name].status == TagStatus.ACTIVE.value]
            return []

    def check_task_owner(self, task_id: str) -> Optional[str]:
        """
        Check which agent owns a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Agent ID if claimed, None otherwise
        """
        with self._lock:
            return self.task_claims.get(task_id)

    def release_task(self, task_id: str, agent_id: str) -> bool:
        """
        Release a task claim.
        
        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            
        Returns:
            True if released, False if not claimed by agent
        """
        with self._lock:
            if task_id in self.task_claims and self.task_claims[task_id] == agent_id:
                del self.task_claims[task_id]
                
                # Also update context
                claim_key = f"task_claim_{task_id}"
                if claim_key in self.contexts:
                    self.contexts[claim_key].status = TagStatus.ARCHIVED.value
                    
                self.logger.info(f"Task {task_id} released by {agent_id}")
                return True
            return False

    def clear_context(self, key: str):
        """
        Mark a context as inactive.
        
        Args:
            key: Context key
        """
        with self._lock:
            if key in self.contexts:
                self.contexts[key].status = TagStatus.ARCHIVED.value
                self.logger.info(f"Cleared context: {key}")

    def set_context_value(self, key: str, value: Any) -> str:
        """
        Set a simple context value.
        
        Args:
            key: Context key
            value: Context value
            
        Returns:
            Context key
        """
        with self._lock:
            tag = Tag(
                name=key,
                intent="context_update",
                category="system_context",
                metadata={"value": value}
            )
            return self.add_tag(tag)

    def get_context_value(self, key: str) -> Optional[Any]:
        """
        Get a simple context value.
        
        Args:
            key: Context key
            
        Returns:
            Context value if found, None otherwise
        """
        with self._lock:
            ctx = self.get_current_context(key)
            if ctx:
                return ctx.metadata.get("value")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            active = [c for c in self.contexts.values() 
                     if c.status == TagStatus.ACTIVE.value]
            return {
                "total_contexts": len(self.contexts),
                "active_contexts": len(active),
                "categories": {c: len(names) for c, names in self.category_index.items()},
                "task_claims": len(self.task_claims)
            }

    def export_state(self) -> str:
        """
        Export state for persistence.
        
        Returns:
            JSON string of state
        """
        with self._lock:
            return json.dumps({
                "contexts": [c.to_dict() for c in self.contexts.values()],
                "task_claims": self.task_claims
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
            self.contexts.clear()
            self.task_claims.clear()
            self.category_index.clear()
            
            # Load contexts
            for context_data in data["contexts"]:
                tag = Tag(**context_data)
                self.contexts[tag.name] = tag
                
                # Rebuild category index
                if tag.category:
                    if tag.category not in self.category_index:
                        self.category_index[tag.category] = set()
                    self.category_index[tag.category].add(tag.name)
            
            # Load task claims
            self.task_claims = data["task_claims"]