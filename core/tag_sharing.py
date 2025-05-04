# Filepath: core/tag_sharing.py
import logging
from typing import Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

class TagSharingSystem:
    """
    System for sharing and adopting tags between users/namespaces.
    """
    
    def __init__(self):
        self.shared_tags = {}  # {shared_id: {"tag": tag_data, "namespace": str, "owner": str}}
        self.logger = logging.getLogger(f"{__name__}.TagSharingSystem")
        
    def share_tag(self, tag: Dict, user_id: str, namespace: str = "global") -> Optional[str]:
        """
        Share a tag with other users/namespaces.
        
        Args:
            tag: The tag dictionary to share
            user_id: ID of the sharing user
            namespace: Target namespace for sharing
            
        Returns:
            Shared tag ID if successful, None otherwise
        """
        try:
            if not tag or not isinstance(tag, dict):
                self.logger.warning("Invalid tag data received for sharing")
                return None
                
            shared_id = str(uuid4())
            self.shared_tags[shared_id] = {
                "tag": tag,
                "namespace": namespace,
                "owner": user_id
            }
            
            self.logger.info(f"Tag shared by {user_id} in namespace {namespace}")
            return shared_id
            
        except Exception as e:
            self.logger.error(f"Error sharing tag: {str(e)}")
            return None
            
    def get_shared_tags(self, namespace: str) -> List[Dict]:
        """
        Get all shared tags in a namespace.
        
        Args:
            namespace: Namespace to query
            
        Returns:
            List of shared tag dictionaries
        """
        try:
            return [
                entry["tag"] for entry in self.shared_tags.values()
                if entry["namespace"] == namespace
            ]
        except Exception as e:
            self.logger.error(f"Error getting shared tags: {str(e)}")
            return []
            
    def adopt_tag(self, shared_tag_id: str, adopting_user_id: str) -> Optional[Dict]:
        """
        Adopt a shared tag into a user's personal namespace.
        
        Args:
            shared_tag_id: ID of the shared tag
            adopting_user_id: ID of the adopting user
            
        Returns:
            The adopted tag dictionary if successful, None otherwise
        """
        try:
            if shared_tag_id not in self.shared_tags:
                self.logger.warning(f"Shared tag ID not found: {shared_tag_id}")
                return None
                
            tag_data = self.shared_tags[shared_tag_id]["tag"]
            self.logger.info(f"Tag {shared_tag_id} adopted by {adopting_user_id}")
            return tag_data
            
        except Exception as e:
            self.logger.error(f"Error adopting tag: {str(e)}")
            return None