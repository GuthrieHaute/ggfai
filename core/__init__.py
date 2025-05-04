# Initialize shared relationship logger
from .relationship_logger import RelationshipLogger
REL_LOGGER = RelationshipLogger()

# Make available to all trackers
from .intent_tracker import IntentTracker
INTENT_TRACKER = IntentTracker()
INTENT_TRACKER.relationship_logger = REL_LOGGER  # Inject dependency

# Repeat for other trackers...