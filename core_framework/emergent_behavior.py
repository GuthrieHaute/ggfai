"""
EmergentBehavior - Rule-based pattern detection for component interactions

Detects and manages emergent behaviors using statistical pattern matching
and rule-based analysis, optimized for mid to high-end hardware.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
import threading
from collections import defaultdict, Counter
import statistics
import json
from enum import Enum, auto
import re
import psutil
from core.run_with_grace import circuit
from core.tag_registry import TagRegistry
from resource_management.hardware_shim import HardwareProfile

logger = logging.getLogger("GGFAI.core_framework.emergent")

class BehaviorType(Enum):
    """Types of emergent behaviors"""
    INTERACTION = auto()    # Component interaction pattern
    RESOURCE = auto()       # Resource usage pattern 
    ERROR = auto()          # Error handling pattern
    OPTIMIZATION = auto()   # Performance optimization
    COORDINATION = auto()   # Multi-component coordination
    ADAPTATION = auto()     # Hardware adaptation pattern
    LEARNING = auto()       # Learning/improvement pattern

@dataclass
class BehaviorPattern:
    """Detected behavior pattern"""
    id: str
    type: BehaviorType
    components: List[str]
    triggers: List[str]  
    actions: List[str]
    confidence: float
    stats: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class InteractionEvent:
    """Record of component interaction"""
    timestamp: float
    source: str
    target: str
    event_type: str
    data: Any
    context: Dict[str, Any]

class EmergentBehaviorSystem:
    """
    System that identifies and promotes emergent behaviors from
    component interactions using rule-based pattern detection.
    """
    
    def __init__(
        self,
        tag_registry: TagRegistry,
        hardware_profile: HardwareProfile,
        min_confidence: float = 0.7
    ):
        self.tag_registry = tag_registry
        self.hardware_profile = hardware_profile
        self.patterns: Dict[str, BehaviorPattern] = {}
        self.interaction_history: List[InteractionEvent] = []
        self.active_behaviors: Dict[str, BehaviorPattern] = {}
        self.min_confidence = min_confidence
        self._lock = threading.RLock()
        self._pattern_rules = self._load_pattern_rules()
        self._max_history = 1000  # Limit history size
        self._initialized = False
        
    def _load_pattern_rules(self) -> Dict[str, Dict]:
        """Load pattern detection rules"""
        return {
            # Resource optimization patterns
            "resource_sharing": {
                "type": BehaviorType.RESOURCE,
                "conditions": [
                    "same_resource_type",
                    "alternating_access",
                    "minimal_conflicts"
                ],
                "min_occurrences": 5
            },
            # Error recovery patterns
            "error_recovery": {
                "type": BehaviorType.ERROR,
                "conditions": [
                    "error_followed_by_recovery",
                    "consistent_recovery_time",
                    "successful_resolution"
                ],
                "min_occurrences": 3
            },
            # Component coordination patterns
            "coordination": {
                "type": BehaviorType.COORDINATION,
                "conditions": [
                    "ordered_interactions",
                    "successful_completion",
                    "data_consistency"
                ],
                "min_occurrences": 4
            },
            # Performance optimization patterns
            "performance": {
                "type": BehaviorType.OPTIMIZATION,
                "conditions": [
                    "reduced_latency",
                    "resource_efficiency",
                    "stable_operation"
                ],
                "min_occurrences": 5
            }
        }
        
    @circuit(failure_threshold=3, recovery_timeout=60)
    async def initialize(self) -> None:
        """Initialize the emergent behavior system with hardware-aware configuration"""
        try:
            # Check hardware tier and adapt accordingly
            tier = self.hardware_profile.get_hardware_tier()
            
            if tier in ["DUST", "GARBAGE"]:
                logger.warning(
                    "Hardware tier too low for emergent behavior detection. "
                    "System will be disabled."
                )
                return
                
            elif tier == "LOW_END":
                logger.info("Configuring for low-end hardware")
                self._pattern_rules = {
                    k: v for k, v in self._pattern_rules.items()
                    if k in {"error_recovery", "resource_sharing"}
                }
                self._max_history = 100
                
            elif tier == "MID_RANGE":
                logger.info("Configuring for mid-range hardware")
                self._max_history = 500
                
            # Register system tags
            await self._register_system_tags()
            
            self._initialized = True
            logger.info(f"Emergent behavior system initialized for {tier} hardware")
            
        except Exception as e:
            logger.error(f"Failed to initialize emergent behavior system: {e}")
            raise
            
    async def _register_system_tags(self) -> None:
        """Register tags used by the emergent behavior system"""
        system_tags = [
            {
                "name": "emergent_pattern",
                "description": "Identifies an emergent behavior pattern",
                "metadata": {"system": "emergent_behavior"}
            },
            {
                "name": "pattern_trigger",
                "description": "Marks events that trigger patterns",
                "metadata": {"system": "emergent_behavior"}  
            },
            {
                "name": "pattern_action",
                "description": "Marks actions taken by patterns",
                "metadata": {"system": "emergent_behavior"}
            }
        ]
        
        for tag in system_tags:
            await self.tag_registry.register_tag_type(**tag)
            
    @circuit(failure_threshold=3, recovery_timeout=30)
    async def record_interaction(
        self,
        source: str,
        target: str,
        event_type: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a component interaction with tag system integration"""
        if not self._initialized:
            return
            
        try:
            event = InteractionEvent(
                timestamp=time.time(),
                source=source,
                target=target,
                event_type=event_type,
                data=data,
                context=context or {}
            )
            
            # Register interaction tags
            await self.tag_registry.add_tag(
                f"interaction_{source}_{target}",
                {"type": event_type, "data": data},
                ttl=3600  # 1 hour TTL
            )
            
            with self._lock:
                self.interaction_history.append(event)
                if len(self.interaction_history) > self._max_history:
                    self.interaction_history = self.interaction_history[-self._max_history:]
                    
            # Analyze for new patterns
            await self._analyze_new_patterns()
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            raise
            
    async def _analyze_new_patterns(self) -> None:
        """Analyze recent interactions for new patterns"""
        if len(self.interaction_history) < 10:  # Need minimum history
            return
            
        with self._lock:
            recent_events = self.interaction_history[-50:]  # Analyze recent window
            
            # Check each pattern rule
            for rule_name, rule in self._pattern_rules.items():
                if await self._check_pattern_rule(rule_name, rule, recent_events):
                    await self._register_pattern(rule_name, rule, recent_events)
                    
    async def _check_pattern_rule(
        self,
        rule_name: str,
        rule: Dict,
        events: List[InteractionEvent]
    ) -> bool:
        """Check if a pattern rule matches recent events"""
        occurrences = 0
        conditions_met = 0
        
        # Count condition matches
        for condition in rule["conditions"]:
            if condition == "same_resource_type":
                if self._check_resource_sharing(events):
                    conditions_met += 1
                    
            elif condition == "alternating_access":
                if self._check_alternating_access(events):
                    conditions_met += 1
                    
            elif condition == "error_followed_by_recovery":
                if self._check_error_recovery(events):
                    conditions_met += 1
                    
            elif condition == "ordered_interactions":
                if self._check_ordered_interactions(events):
                    conditions_met += 1
                    
            elif condition == "reduced_latency":
                if self._check_performance_improvement(events):
                    conditions_met += 1
                    
        # Calculate match confidence
        confidence = conditions_met / len(rule["conditions"])
        
        return (
            confidence >= self.min_confidence and
            occurrences >= rule["min_occurrences"]
        )
        
    def _check_resource_sharing(
        self,
        events: List[InteractionEvent]
    ) -> bool:
        """Check for resource sharing patterns"""
        resource_accesses = defaultdict(list)
        
        for event in events:
            if "resource" in event.context:
                resource = event.context["resource"]
                resource_accesses[resource].append(event)
                
        # Check for efficient sharing
        for resource, accesses in resource_accesses.items():
            if len(accesses) < 3:
                continue
                
            # Check for conflicts
            conflicts = 0
            for i in range(len(accesses) - 1):
                if accesses[i+1].timestamp - accesses[i].timestamp < 0.001:
                    conflicts += 1
                    
            if conflicts / len(accesses) < 0.1:  # Low conflict rate
                return True
                
        return False
        
    def _check_alternating_access(
        self,
        events: List[InteractionEvent]
    ) -> bool:
        """Check for alternating access patterns"""
        component_sequence = [e.source for e in events]
        
        # Check for alternating components
        alternations = 0
        for i in range(len(component_sequence) - 1):
            if component_sequence[i] != component_sequence[i+1]:
                alternations += 1
                
        return alternations / len(component_sequence) > 0.4
        
    def _check_error_recovery(
        self,
        events: List[InteractionEvent]
    ) -> bool:
        """Check for error recovery patterns"""
        for i in range(len(events) - 1):
            if (events[i].event_type == "error" and
                events[i+1].event_type == "recovery"):
                return True
        return False
        
    def _check_ordered_interactions(
        self,
        events: List[InteractionEvent]
    ) -> bool:
        """Check for ordered interaction patterns"""
        component_pairs = [
            (e1.source, e1.target, e2.source, e2.target)
            for e1, e2 in zip(events, events[1:])
        ]
        
        # Look for repeated sequences
        pair_counts = Counter(component_pairs)
        most_common = pair_counts.most_common(1)
        
        if most_common:
            pair, count = most_common[0]
            return count >= 3  # Pattern appears at least 3 times
            
        return False
        
    def _check_performance_improvement(
        self,
        events: List[InteractionEvent]
    ) -> bool:
        """Check for performance improvement patterns"""
        if len(events) < 5:
            return False
            
        # Extract timing data
        timings = []
        for e1, e2 in zip(events, events[1:]):
            duration = e2.timestamp - e1.timestamp
            if duration > 0:  # Valid duration
                timings.append(duration)
                
        if not timings:
            return False
            
        # Check for decreasing trend
        mean_first_half = statistics.mean(timings[:len(timings)//2])
        mean_second_half = statistics.mean(timings[len(timings)//2:])
        
        return mean_second_half < mean_first_half * 0.9  # 10% improvement
        
    async def _register_pattern(
        self,
        rule_name: str,
        rule: Dict,
        events: List[InteractionEvent]
    ) -> None:
        """Register a newly detected pattern"""
        pattern_id = f"{rule_name}_{int(time.time())}"
        
        # Extract components involved
        components = {e.source for e in events} | {e.target for e in events}
        
        # Extract common triggers
        triggers = [
            e.event_type for e in events
            if e.event_type.startswith("trigger_")
        ]
        
        # Extract common actions
        actions = [
            e.event_type for e in events
            if e.event_type.startswith("action_")
        ]
        
        # Calculate confidence from event consistency
        event_types = Counter(e.event_type for e in events)
        total_events = len(events)
        consistency = max(event_types.values()) / total_events
        
        pattern = BehaviorPattern(
            id=pattern_id,
            type=rule["type"],
            components=list(components),
            triggers=list(set(triggers)),
            actions=list(set(actions)),
            confidence=consistency,
            stats={
                "occurrences": len(events),
                "consistency": consistency,
                "components": len(components)
            },
            metadata={
                "rule": rule_name,
                "first_seen": events[0].timestamp,
                "last_seen": events[-1].timestamp
            }
        )
        
        with self._lock:
            self.patterns[pattern_id] = pattern
            logger.info(f"Registered new pattern: {pattern_id}")
            
    @circuit(failure_threshold=3, recovery_timeout=30)
    async def validate_pattern(self, pattern_id: str) -> bool:
        """
        Validate a detected pattern against current system state
        and resource constraints
        """
        if not self._initialized:
            return False

        try:
            pattern = self.patterns.get(pattern_id)
            if not pattern:
                return False

            # Check hardware requirements
            tier = self.hardware_profile.get_hardware_tier()
            if tier in ["DUST", "GARBAGE"]:
                logger.warning(f"Pattern {pattern_id} cannot run on {tier} hardware")
                return False

            # Validate component availability
            for component in pattern.components:
                component_tag = await self.tag_registry.get_tag(f"component_{component}")
                if not component_tag or component_tag.get("status") != "available":
                    logger.warning(f"Component {component} not available for pattern {pattern_id}")
                    return False

            # Check resource requirements
            resources = pattern.hardware_requirements
            if resources:
                for resource, required in resources.items():
                    available = await self.hardware_profile.get_resource_availability(resource)
                    if available < required:
                        logger.warning(
                            f"Insufficient {resource} for pattern {pattern_id}: "
                            f"need {required}, have {available}"
                        )
                        return False

            # Update pattern confidence based on validation
            pattern.confidence *= 1.1  # Increase confidence on successful validation
            pattern.stats["last_validated"] = time.time()
            
            return True

        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            pattern.confidence *= 0.9  # Reduce confidence on validation failure
            return False

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle emergent behavior system errors with graceful degradation"""
        try:
            error_type = type(error).__name__
            error_context = {
                "type": error_type,
                "message": str(error),
                "context": context,
                "timestamp": time.time()
            }

            # Register error tag
            await self.tag_registry.add_tag(
                f"emergent_error_{error_type}",
                error_context,
                ttl=3600
            )

            # Adjust system behavior based on error
            if error_type in ["MemoryError", "ResourceExhaustedError"]:
                # Reduce resource usage
                self._max_history = max(50, self._max_history // 2)
                logger.warning(f"Reduced history size to {self._max_history} due to resource constraints")

            elif error_type in ["TimeoutError", "ConcurrentAccessError"]:
                # Increase timeouts and add delays
                self._pattern_rules = {
                    k: {**v, "min_occurrences": v["min_occurrences"] + 1}
                    for k, v in self._pattern_rules.items()
                }
                logger.warning("Increased pattern detection thresholds due to timing issues")

            # Clean up any corrupted state
            self._cleanup_after_error(error_type)

        except Exception as e:
            logger.error(f"Error handler failed: {e}")
            # Last resort - reset to safe state
            self._reset_to_safe_state()

    def _cleanup_after_error(self, error_type: str) -> None:
        """Clean up system state after an error"""
        with self._lock:
            # Remove potentially corrupted patterns
            corrupted = []
            for pattern_id, pattern in self.patterns.items():
                if pattern.confidence < 0.3:  # Low confidence threshold
                    corrupted.append(pattern_id)
            
            for pattern_id in corrupted:
                del self.patterns[pattern_id]

            # Clear recent history if needed
            if error_type in ["DataCorruptionError", "StateError"]:
                self.interaction_history = []

    def _reset_to_safe_state(self) -> None:
        """Reset system to a known safe state"""
        with self._lock:
            self.interaction_history = []
            self.patterns = {}
            self.active_behaviors = {}
            self._pattern_rules = self._load_pattern_rules()
            self._max_history = 100  # Conservative history size
            self._initialized = False

    async def get_active_patterns(
        self,
        min_confidence: Optional[float] = None
    ) -> List[BehaviorPattern]:
        """
        Get currently active behavior patterns.
        
        Args:
            min_confidence: Optional minimum confidence threshold
            
        Returns:
            List of active behavior patterns
        """
        threshold = min_confidence or self.min_confidence
        
        with self._lock:
            return [
                pattern for pattern in self.patterns.values()
                if pattern.confidence >= threshold
            ]
            
    def get_pattern_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about detected patterns"""
        with self._lock:
            stats = defaultdict(list)
            
            for pattern in self.patterns.values():
                stats[pattern.type.name].append({
                    "id": pattern.id,
                    "confidence": pattern.confidence,
                    "components": len(pattern.components),
                    "triggers": len(pattern.triggers),
                    "actions": len(pattern.actions)
                })
                
            return dict(stats)

    @circuit(failure_threshold=3, recovery_timeout=30)
    async def execute_pattern(self, pattern_id: str) -> bool:
        """Execute a behavior pattern with resource awareness and error handling"""
        if not self._initialized:
            return False
            
        try:
            pattern = self.patterns.get(pattern_id)
            if not pattern or not await self.validate_pattern(pattern_id):
                return False
                
            # Check pattern is not already active
            if pattern_id in self.active_behaviors:
                logger.warning(f"Pattern {pattern_id} is already active")
                return False
                
            # Reserve resources
            resources_reserved = await self._reserve_resources(pattern)
            if not resources_reserved:
                logger.warning(f"Failed to reserve resources for pattern {pattern_id}")
                return False
                
            try:
                # Execute pattern actions
                for action in pattern.actions:
                    await self.tag_registry.add_tag(
                        f"pattern_action_{pattern_id}",
                        {
                            "action": action,
                            "timestamp": time.time(),
                            "pattern": pattern_id
                        }
                    )
                    
                # Mark pattern as active
                self.active_behaviors[pattern_id] = pattern
                pattern.stats["last_executed"] = time.time()
                pattern.confidence *= 1.05  # Slight confidence boost on successful execution
                
                logger.info(f"Successfully executed pattern {pattern_id}")
                return True
                
            finally:
                # Always release resources
                await self._release_resources(pattern)
                
        except Exception as e:
            logger.error(f"Pattern execution failed: {e}")
            await self.handle_error(e, {"pattern_id": pattern_id})
            pattern.confidence *= 0.9  # Reduce confidence on execution failure
            return False
            
    async def _reserve_resources(self, pattern: BehaviorPattern) -> bool:
        """Reserve resources needed for pattern execution"""
        try:
            required_resources = pattern.hardware_requirements
            if not required_resources:
                return True  # No resources needed
                
            # Try to reserve each resource
            for resource, amount in required_resources.items():
                reserved = await self.hardware_profile.reserve_resource(
                    resource,
                    amount,
                    ttl=60  # 1 minute timeout
                )
                
                if not reserved:
                    # Failed to reserve - release any that were reserved
                    await self._release_resources(pattern)
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Resource reservation failed: {e}")
            return False
            
    async def _release_resources(self, pattern: BehaviorPattern) -> None:
        """Release resources used by a pattern"""
        try:
            for resource, amount in pattern.hardware_requirements.items():
                await self.hardware_profile.release_resource(resource, amount)
                
        except Exception as e:
            logger.error(f"Resource release failed: {e}")
            
    async def optimize_patterns(self) -> None:
        """Optimize pattern detection and execution based on hardware tier"""
        if not self._initialized:
            return
            
        try:
            tier = self.hardware_profile.get_hardware_tier()
            
            # Get current resource usage
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # Adjust pattern detection based on resource usage
            if tier in ["DUST", "GARBAGE"]:
                # Minimal operation on very low-end hardware
                self._max_history = 50
                self.min_confidence = 0.9  # Very high confidence required
                
            elif tier == "LOW_END":
                if cpu_usage > 80 or memory_usage > 80:
                    # Heavy resource usage - reduce pattern detection
                    self._max_history = max(50, self._max_history - 100)
                    self.min_confidence += 0.05
                elif cpu_usage < 50 and memory_usage < 50:
                    # Light resource usage - can be more aggressive
                    self._max_history = min(200, self._max_history + 50)
                    self.min_confidence = max(0.7, self.min_confidence - 0.02)
                    
            else:  # MID_RANGE or HIGH_END
                if cpu_usage > 90 or memory_usage > 90:
                    # Critical resource usage - reduce activity
                    self._max_history = max(100, self._max_history - 200)
                    self.min_confidence += 0.1
                elif cpu_usage < 60 and memory_usage < 60:
                    # Plenty of resources - full operation
                    self._max_history = min(1000, self._max_history + 100)
                    self.min_confidence = max(0.6, self.min_confidence - 0.05)
                    
            # Clean up low confidence patterns
            await self._cleanup_patterns()
            
        except Exception as e:
            logger.error(f"Pattern optimization failed: {e}")
            await self.handle_error(e, {"operation": "optimize_patterns"})
            
    async def _cleanup_patterns(self) -> None:
        """Clean up low confidence and stale patterns"""
        with self._lock:
            current_time = time.time()
            to_remove = []
            
            for pattern_id, pattern in self.patterns.items():
                # Remove patterns with very low confidence
                if pattern.confidence < 0.2:
                    to_remove.append(pattern_id)
                    continue
                    
                # Remove patterns that haven't been validated recently
                last_validated = pattern.stats.get("last_validated", 0)
                if current_time - last_validated > 3600:  # 1 hour
                    to_remove.append(pattern_id)
                    continue
                    
                # Remove patterns that consistently fail execution
                last_executed = pattern.stats.get("last_executed", 0)
                if (current_time - last_executed > 1800 and  # 30 minutes
                    pattern.confidence < 0.4):
                    to_remove.append(pattern_id)
                    
            # Remove identified patterns
            for pattern_id in to_remove:
                del self.patterns[pattern_id]
                if pattern_id in self.active_behaviors:
                    del self.active_behaviors[pattern_id]