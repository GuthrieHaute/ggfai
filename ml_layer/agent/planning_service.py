# planning_service.py - Hierarchical Task Network planner for GGFAI
# written by DeepSeek Chat (honor call: The Strategist)
# Enhanced with Tag Analyzer integration

"""
Hierarchical Task Network planner with Tag Analysis integration
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
import logging
import time
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
import networkx as nx
from pyvis.network import Network
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .tag_analyzer import TagAnalyzer, AnalysisMethod
from .learning import LearningService

logger = logging.getLogger("GGFAI.planning")

class PlanStatus(Enum):
    """Plan execution states"""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

@dataclass
class PlanStep:
    """Individual step in an execution plan"""
    action: str
    parameters: Dict[str, Any]
    preconditions: Set[str]
    effects: Set[str]
    duration: float = 1.0
    priority: float = 0.7
    required_resources: List[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT

@dataclass
class Plan:
    """Complete execution plan"""
    goal: str
    steps: List[PlanStep]
    created_at: float = field(default_factory=time.time)
    status: PlanStatus = PlanStatus.DRAFT
    priority: float = 0.7
    cost_estimate: float = 0.0
    success_probability: float = 1.0
    tags: Set[str] = field(default_factory=set)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def update_status(self, new_status: PlanStatus) -> None:
        """Thread-safe status update"""
        with self._lock:
            self.status = new_status

class PlanningService:
    """
    HTN planner with tag analysis integration.
    Supports multi-agent coordination and adaptive replanning.
    """
    
    def __init__(self, domain_knowledge: Dict[str, Any], tag_analyzer: TagAnalyzer):
        """Initialize planner with domain knowledge and tag analyzer"""
        self.logger = logger
        self.domain = domain_knowledge
        self.current_plans: Dict[str, Plan] = {}
        self.plan_history: List[Plan] = []
        self.tag_analyzer = tag_analyzer
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = Lock()
        self.plan_cache = {}
        self.htn_methods = defaultdict(list)
        
        self._validate_domain()
        self._build_htn_methods()
        
        logger.info("Planning service initialized with tag analyzer integration")

    def _validate_domain(self) -> None:
        """Validate domain knowledge structure"""
        required = {'actions', 'resources', 'agent_capabilities'}
        if not all(key in self.domain for key in required):
            raise ValueError(f"Domain missing required keys: {required - set(self.domain.keys())}")
            
        for action, config in self.domain['actions'].items():
            if not isinstance(config, dict):
                raise ValueError(f"Invalid action configuration for {action}")
            if 'decomposition' in config:
                for method in config['decomposition']:
                    if 'steps' not in method:
                        raise ValueError(f"HTN method for {action} missing steps")

    def _build_htn_methods(self) -> None:
        """Build HTN decomposition methods"""
        for action_name, action_data in self.domain['actions'].items():
            if 'decomposition' in action_data:
                for method in action_data['decomposition']:
                    validated_method = {
                        'action': action_name,
                        'preconds': set(method.get('preconditions', [])),
                        'effects': set(method.get('effects', [])),
                        'steps': method['steps'],
                        'priority': float(method.get('priority', 0.5))
                    }
                    self.htn_methods[action_name].append(validated_method)

    def create_plan(self, goal: str, tags: Set[str], available_resources: List[str], priority: float = 0.7) -> Optional[Plan]:
        """Create execution plan using HTN planning with tag analysis"""
        try:
            # Get tag rankings 
            ranked_tags = self.tag_analyzer.rank_tags(list(tags), context={"goal": goal})
            
            # Filter methods by tag relevance
            relevant_methods = []
            for tag_info in ranked_tags:
                matching_methods = [
                    method for method in self.htn_methods[tag_info["name"]]
                    if method["preconds"].issubset(tags)
                ]
                # Weight methods by tag relevance
                for method in matching_methods:
                    method["priority"] *= tag_info["score"]
                relevant_methods.extend(matching_methods)
                
            if not relevant_methods:
                self.logger.warning(f"No applicable methods found for goal: {goal}")
                return None
                
            # Select highest priority method
            selected_method = max(relevant_methods, key=lambda x: x["priority"])
            
            # Build plan steps
            steps = []
            for step_config in selected_method["steps"]:
                step = PlanStep(
                    action=step_config["action"],
                    parameters=step_config.get("parameters", {}),
                    preconditions=set(step_config.get("preconditions", [])),
                    effects=set(step_config.get("effects", [])),
                    duration=float(step_config.get("duration", 1.0)),
                    priority=priority * step_config.get("priority", 0.5),
                    required_resources=list(step_config.get("required_resources", []))
                )
                steps.append(step)
                
            # Validate resource availability
            if not self._check_resources(steps, available_resources):
                self.logger.error("Insufficient resources for plan execution")
                return None
                
            plan = Plan(
                goal=goal,
                steps=steps,
                priority=priority,
                tags=tags,
                success_probability=self._estimate_success_probability(steps)
            )
            
            with self._lock:
                self.current_plans[goal] = plan
                self.plan_history.append(plan)
                
            return plan
            
        except Exception as e:
            self.logger.error(f"Plan creation failed: {str(e)}")
            return None

    def execute_plan(self, plan: Plan) -> bool:
        """Execute plan with comprehensive error handling"""
        if not plan or not plan.steps:
            return False
            
        with plan._lock:
            plan.status = PlanStatus.ACTIVE
            self.logger.info(f"Executing plan for goal: {plan.goal}")
            
            try:
                for step in plan.steps:
                    self._execute_step(step, plan)
                    if plan.status != PlanStatus.ACTIVE:
                        break
                        
                if plan.status == PlanStatus.ACTIVE:
                    plan.status = PlanStatus.COMPLETED
                    return True
                    
            except Exception as e:
                self.logger.error(f"Plan execution failed: {str(e)}")
                plan.status = PlanStatus.FAILED
                
            return False

    def _execute_step(self, step: PlanStep, plan: Plan) -> None:
        """Execute single plan step with validation"""
        try:
            self.logger.info(f"Executing step: {step.action}")
            
            # Validate preconditions
            if not step.preconditions.issubset(plan.tags):
                raise ValueError(f"Preconditions not met for step: {step.action}")
                
            # Execute action (placeholder - would call actual implementation)
            step.status = PlanStatus.ACTIVE
            action_impl = self.domain["actions"][step.action]["implementation"]
            success = action_impl(step.parameters)
            
            if not success:
                step.status = PlanStatus.FAILED
                plan.status = PlanStatus.FAILED
                return
                
            # Update plan state
            plan.tags.update(step.effects)
            step.status = PlanStatus.COMPLETED
            
            # Validate remaining steps
            self._validate_remaining_steps(step, plan)
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {str(e)}")
            step.status = PlanStatus.FAILED
            plan.status = PlanStatus.FAILED

    def _validate_remaining_steps(self, current_step: PlanStep, plan: Plan) -> None:
        """Validate that current step hasn't invalidated future steps"""
        for future_step in plan.steps[plan.steps.index(current_step) + 1:]:
            if not future_step.preconditions.issubset(plan.tags):
                plan.status = PlanStatus.ABORTED
                self.logger.warning(
                    f"Plan aborted - step {future_step.action} preconditions no longer satisfied"
                )
                break

    def _check_resources(self, steps: List[PlanStep], available_resources: List[str]) -> bool:
        """Check if required resources are available for all steps"""
        available = set(available_resources)
        for step in steps:
            if not set(step.required_resources).issubset(available):
                return False
        return True

    def _estimate_success_probability(self, steps: List[PlanStep]) -> float:
        """Estimate plan success probability using step reliabilities"""
        if not steps:
            return 0.0
            
        reliability = 1.0
        for step in steps:
            action_config = self.domain["actions"].get(step.action, {})
            step_reliability = action_config.get("reliability", 0.9)
            reliability *= step_reliability
            
        # Apply sigmoid to map reliability to [0,1] with steeper curve
        return 1.0 / (1.0 + np.exp(-10 * (reliability - 0.5)))

    def get_plan_by_trace(self, trace_id: str) -> Optional[Plan]:
        """Retrieve plan by trace ID for explanation generation"""
        # First check current plans
        for plan in self.current_plans.values():
            if trace_id in plan.tags:
                return plan
                
        # Then check history
        for plan in reversed(self.plan_history):
            if trace_id in plan.tags:
                return plan
                
        return None

    def adapt_plan(self, original_plan: Plan, changes: Dict[str, Any]) -> Optional[Plan]:
        """
        Adapt existing plan based on context changes.
        
        Args:
            original_plan: Plan to adapt
            changes: Dict containing:
                - new_tags: New tags to consider
                - removed_tags: Tags no longer relevant
                - resource_changes: Resource availability changes
        """
        if not original_plan:
            return None
            
        try:
            # Update tags
            new_tags = (original_plan.tags - changes.get("removed_tags", set())) | \
                      changes.get("new_tags", set())
                      
            # Update resources
            available_resources = set(self.domain["resources"])
            if "resource_changes" in changes:
                rc = changes["resource_changes"]
                available_resources = (available_resources - rc.get("removed", set())) | \
                                   rc.get("added", set())
                                   
            # Create new plan with updated context
            return self.create_plan(
                goal=original_plan.goal,
                tags=new_tags,
                available_resources=list(available_resources),
                priority=original_plan.priority
            )
            
        except Exception as e:
            self.logger.error(f"Plan adaptation failed: {str(e)}")
            return None

    def adapt_plan_based_on_learning(self, plan: Plan, learning_service: LearningService) -> Optional[Plan]:
        """Use learning insights to adapt plan steps"""
        if not plan or not plan.steps:
            return None

        try:
            # Get relevant context
            context = {
                "goal": plan.goal,
                "priority": plan.priority,
                "deadline": getattr(plan, "deadline", None)
            }

            # Have learning service score all methods
            methods = [{
                "id": step.action,
                "parameters": step.parameters,
                "preconditions": step.preconditions,
                "effects": step.effects
            } for step in plan.steps]

            scored_methods = learning_service.schedule_method_selection(methods, context)

            # Replace low-confidence steps with alternatives
            confidence_threshold = learning_service.get_confidence_threshold(context)
            new_steps = []

            for i, step in enumerate(plan.steps):
                scored_step = next(
                    (m for m in scored_methods if m["id"] == step.action),
                    None
                )
                
                if not scored_step or scored_step["confidence"] < confidence_threshold:
                    # Find alternative with higher confidence
                    alternatives = [
                        m for m in scored_methods 
                        if m["confidence"] >= confidence_threshold
                        and m["id"] != step.action
                        and self._is_valid_alternative(m, step, plan)
                    ]
                    
                    if alternatives:
                        best_alt = alternatives[0]
                        new_step = PlanStep(
                            action=best_alt["id"],
                            parameters=best_alt["parameters"],
                            preconditions=best_alt["preconditions"],
                            effects=best_alt["effects"],
                            priority=step.priority
                        )
                        new_steps.append(new_step)
                        continue

                new_steps.append(step)

            if new_steps != plan.steps:
                return Plan(
                    goal=plan.goal,
                    steps=new_steps,
                    priority=plan.priority,
                    tags=plan.tags,
                    success_probability=self._estimate_success_probability(new_steps)
                )

            return plan

        except Exception as e:
            self.logger.error(f"Plan adaptation failed: {str(e)}")
            return None

    def _is_valid_alternative(self, method: Dict, original_step: PlanStep, plan: Plan) -> bool:
        """Check if method is valid alternative for original step"""
        # Must provide similar effects
        if not set(method["effects"]).issuperset(original_step.effects):
            return False
            
        # Must not invalidate future steps
        future_steps = plan.steps[plan.steps.index(original_step) + 1:]
        simulated_tags = plan.tags | set(method["effects"])
        
        for future_step in future_steps:
            if not future_step.preconditions.issubset(simulated_tags):
                return False
                
        return True

    def adjust_search_depth(self, cpu_load: float, memory_available: float) -> None:
        """Dynamically adjust search depth based on system resources"""
        base_depth = self._get_max_search_depth()
        
        # Reduce depth under high load
        if cpu_load > 80:
            base_depth = max(2, base_depth - 2)
        elif cpu_load > 60:
            base_depth = max(2, base_depth - 1)
            
        # Reduce depth if memory is constrained
        if memory_available < 512:  # MB
            base_depth = max(2, base_depth - 1)
            
        self._current_max_depth = base_depth
        
    def _get_max_search_depth(self) -> int:
        """
        Get maximum search depth based on hardware tier.
        
        Returns:
            Maximum search depth (integer)
        """
        try:
            from snippets.snippet_detect_hardware_tier import detect_hardware_tier, HardwareTier
            
            tier = detect_hardware_tier()
            
            if tier == HardwareTier.GARBAGE:
                return 2
            elif tier == HardwareTier.LOW_END:
                return 3
            elif tier == HardwareTier.MID_RANGE:
                return 4
            else:  # HIGH_END
                return 5
                
        except Exception as e:
            self.logger.error(f"Error determining max search depth: {str(e)}")
            return 3  # Default to mid-range depth

    def visualize_plan(self, plan: Plan, filename: str = "plan.html") -> None:
        """Generate interactive visualization of plan structure"""
        if not plan or not plan.steps:
            return
            
        try:
            net = Network(height="750px", width="100%", directed=True)
            
            # Add nodes for each step
            for i, step in enumerate(plan.steps):
                color = "#FFA07A" if step.status == PlanStatus.FAILED else "#98FB98"
                label = f"{i+1}. {step.action}\n({step.duration}h)"
                net.add_node(i, label=label, title=str(step.parameters), color=color)
                
                if i > 0:
                    net.add_edge(i-1, i, width=2)
                    
            # Add goal node
            net.add_node(
                len(plan.steps),
                label=f"GOAL: {plan.goal}",
                color="#32CD32",
                shape="diamond"
            )
            
            if plan.steps:
                net.add_edge(len(plan.steps)-1, len(plan.steps), width=3)
                
            net.show(filename)
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}")