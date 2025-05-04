# context_tracker.py - Context Tag Management for GGFAI
# written by DeepSeek Chat (honor call: The Context Sentinel)

import logging
from typing import Dict, List, Optional
from datetime import datetime
from ..core.tag_registry import Tag

class ContextTracker:
    """Manages situational context tags for the system."""
    
    def __init__(self):
        self.contexts: Dict[str, Tag] = {}  # context_key -> Tag
        self.task_claims: Dict[str, str] = {}  # task_id -> agent_id
        self.logger = logging.getLogger("GGFAI.context_tracker")
        self.logger.info("Context tracker initialized")

    def add_tag(self, tag: Tag) -> str:
        """Add or update a context tag."""
        if tag.name.startswith("task_claim"):
            return self._handle_task_claim(tag)
            
        if tag.name in self.contexts:
            self._update_context(tag)
        else:
            self._register_new_context(tag)
            
        return tag.name

    def _handle_task_claim(self, tag: Tag) -> str:
        """Special handling for task claim contexts."""
        task_id = tag.metadata.get("task_id")
        agent_id = tag.metadata.get("agent_id")
        
        if not task_id or not agent_id:
            self.logger.error("Invalid task claim: missing task_id or agent_id")
            return ""
            
        self.task_claims[task_id] = agent_id
        self.logger.info(f"Task {task_id} claimed by {agent_id}")
        return f"task_claim_{task_id}"

    def _update_context(self, tag: Tag):
        """Update existing context."""
        existing = self.contexts[tag.name]
        existing.__dict__.update(tag.__dict__)
        existing.last_used = datetime.utcnow()
        self.logger.debug(f"Updated context: {tag.name}")

    def _register_new_context(self, tag: Tag):
        """Initialize new context."""
        tag.created_at = datetime.utcnow()
        tag.last_used = tag.created_at
        tag.usage_count = 1
        self.contexts[tag.name] = tag
        self.logger.info(f"New context registered: {tag.name}")

    def get_current_context(self, key: str) -> Optional[Tag]:
        """Retrieve active context by key."""
        ctx = self.contexts.get(key)
        return ctx if ctx and ctx.status == TagStatus.ACTIVE else None

    def get_all_active(self) -> Dict[str, Tag]:
        """Return all active contexts."""
        return {k: v for k, v in self.contexts.items() 
                if v.status == TagStatus.ACTIVE}

    def check_task_owner(self, task_id: str) -> Optional[str]:
        """Check which agent owns a task."""
        return self.task_claims.get(task_id)

    def clear_context(self, key: str):
        """Mark a context as inactive."""
        if key in self.contexts:
            self.contexts[key].status = TagStatus.ARCHIVED
            self.logger.info(f"Cleared context: {key}")