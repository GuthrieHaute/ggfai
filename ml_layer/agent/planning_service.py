# planning_service.py - Hierarchical Task Network planner for GGFAI
# written by DeepSeek Chat (honor call: The Strategist)

from typing import Dict, List, Optional, Tuple, Set
import logging
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
import networkx as nx
from pyvis.network import Network
from collections import defaultdict

class PlanStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

@dataclass
class PlanStep:
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
    goal: str
    steps: List[PlanStep]
    created_at: float = field(default_factory=time.time)
    status: PlanStatus = PlanStatus.DRAFT
    priority: float = 0.7
    cost_estimate: float = 0.0
    success_probability: float = 1.0
    tags: Set[str] = field(default_factory=set)

class PlanningService:
    """
    Hierarchical Task Network planner with PDDL-inspired capabilities.
    Supports multi-agent coordination, plan explanation, and adaptive replanning.
    """
    
    def __init__(self, domain_knowledge: Dict[str, Any]):
        """
        Initialize with domain knowledge about actions, resources, and agents.
        
        Args:
            domain_knowledge: Dictionary containing:
                - actions: Available actions and their parameters
                - resources: Available resources in the system
                - agent_capabilities: What each agent can do
        """
        self.logger = logging.getLogger("GGFAI.planning")
        self.domain = domain_knowledge
        self.current_plans: Dict[str, Plan] = {}
        self.plan_history: List[Plan] = []
        
        # Initialize HTN methods
        self._build_htn_methods()
        
        # Planning heuristics
        self.heuristics = {
            'resource_usage': self._resource_usage_heuristic,
            'tag_priority': self._tag_priority_heuristic
        }
        
        self.logger.info("Planning service initialized")

    def _build_htn_methods(self):
        """Precompile HTN decomposition methods from domain knowledge."""
        self.htn_methods = defaultdict(list)
        
        for action_name, action_data in self.domain['actions'].items():
            if 'decomposition' in action_data:
                for method in action_data['decomposition']:
                    self.htn_methods[action_name].append({
                        'preconds': set(method.get('preconditions', [])),
                        'steps': method['steps'],
                        'priority': method.get('priority', 0.5)
                    })

    def create_plan(self, 
                   goal: str,
                   tags: Set[str],
                   available_resources: List[str],
                   priority: float = 0.7) -> Optional[Plan]:
        """
        Create a plan to achieve given goal using HTN planning.
        
        Args:
            goal: The goal to achieve (must match an HTN task)
            tags: Relevant tags for contextual planning
            available_resources: Currently available resources
            priority: Priority of this planning request
            
        Returns:
            A Plan object or None if planning fails
        """
        if goal not in self.htn_methods:
            self.logger.error(f"No HTN methods available for goal: {goal}")
            return None
            
        # Select best HTN method based on preconditions and priority
        applicable_methods = []
        for method in self.htn_methods[goal]:
            if method['preconds'].issubset(tags):
                applicable_methods.append(method)
                
        if not applicable_methods:
            self.logger.error(f"No applicable methods for goal: {goal}")
            return None
            
        # Select method with highest priority
        selected_method = max(applicable_methods, key=lambda x: x['priority'])
        
        # Build plan steps
        steps = []
        for step_desc in selected_method['steps']:
            action = self.domain['actions'].get(step_desc['action'])
            if not action:
                continue
                
            step = PlanStep(
                action=step_desc['action'],
                parameters=step_desc.get('parameters', {}),
                preconditions=set(action.get('preconditions', [])),
                effects=set(action.get('effects', [])),
                duration=action.get('duration', 1.0),
                priority=priority,
                required_resources=action.get('required_resources', [])
            )
            steps.append(step)
        
        # Verify resource availability
        if not self._check_resource_availability(steps, available_resources):
            self.logger.error("Insufficient resources for plan")
            return None
            
        plan = Plan(
            goal=goal,
            steps=steps,
            priority=priority,
            tags=tags,
            cost_estimate=sum(step.duration for step in steps),
            success_probability=self._estimate_success_probability(steps)
        )
        
        self.current_plans[goal] = plan
        return plan

    def _check_resource_availability(self, 
                                  steps: List[PlanStep],
                                  available_resources: List[str]) -> bool:
        """Verify all required resources are available."""
        required = set()
        for step in steps:
            required.update(step.required_resources)
            
        return required.issubset(available_resources)

    def _estimate_success_probability(self, steps: List[PlanStep]) -> float:
        """Estimate plan success probability based on step reliability."""
        if not steps:
            return 0.0
            
        # Simple product of step reliabilities (could be enhanced with ML)
        reliability = 1.0
        for step in steps:
            action_data = self.domain['actions'].get(step.action, {})
            reliability *= action_data.get('reliability', 0.9)
            
        return reliability

    def execute_plan(self, plan: Plan) -> bool:
        """Execute a plan and handle step transitions."""
        plan.status = PlanStatus.ACTIVE
        self.logger.info(f"Executing plan for goal: {plan.goal}")
        
        for i, step in enumerate(plan.steps):
            if plan.status != PlanStatus.ACTIVE:
                break
                
            self._execute_step(step, plan)
            
        if plan.status == PlanStatus.ACTIVE:
            plan.status = PlanStatus.COMPLETED
            self.logger.info(f"Plan completed: {plan.goal}")
            return True
            
        return False

    def _execute_step(self, step: PlanStep, plan: Plan):
        """Execute a single plan step with error handling."""
        try:
            self.logger.info(f"Executing step: {step.action}")
            # TODO: Actual execution would interface with agents
            step.status = PlanStatus.COMPLETED
            
            # Check if any effects invalidate remaining steps
            for future_step in plan.steps[plan.steps.index(step)+1:]:
                if not future_step.preconditions.issubset(plan.tags | step.effects):
                    plan.status = PlanStatus.ABORTED
                    self.logger.warning(
                        f"Plan aborted - step {future_step.action} preconditions not met")
                    break
                    
        except Exception as e:
            self.logger.error(f"Step failed: {step.action} - {str(e)}")
            step.status = PlanStatus.FAILED
            plan.status = PlanStatus.FAILED

    def visualize_plan(self, plan: Plan, filename: str = "plan.html"):
        """Generate interactive visualization of the plan."""
        net = Network(height="750px", width="100%", directed=True)
        
        # Add nodes for each step
        for i, step in enumerate(plan.steps):
            label = f"{i+1}. {step.action}\nDuration: {step.duration}h"
            net.add_node(i, label=label, title=str(step.parameters))
            
            # Add edges for dependencies
            if i > 0:
                net.add_edge(i-1, i)
                
        # Add goal node
        net.add_node(len(plan.steps), 
                   label=f"GOAL: {plan.goal}", 
                   color="green")
        if plan.steps:
            net.add_edge(len(plan.steps)-1, len(plan.steps))
            
        net.show(filename)

    def adapt_plan(self, 
                  original_plan: Plan,
                  changes: Dict[str, Any]) -> Optional[Plan]:
        """
        Adapt an existing plan based on changes in context.
        
        Args:
            original_plan: The plan to adapt
            changes: Dictionary describing changes:
                - new_tags: Set of new relevant tags
                - removed_tags: Set of no-longer-relevant tags
                - resource_changes: Changes in resource availability
                
        Returns:
            Adapted Plan or None if replanning fails
        """
        new_tags = (original_plan.tags - changes.get('removed_tags', set())) | \
                   changes.get('new_tags', set())
                   
        available_resources = self._get_current_resources()
        if 'resource_changes' in changes:
            available_resources = (available_resources - 
                                  changes['resource_changes'].get('removed', set())) | \
                                 changes['resource_changes'].get('added', set())
                                 
        return self.create_plan(
            goal=original_plan.goal,
            tags=new_tags,
            available_resources=available_resources,
            priority=original_plan.priority
        )

    def _get_current_resources(self) -> Set[str]:
        """Get currently available resources (simplified)."""
        return set(self.domain['resources'])

# learning.py - Bandit learning for dynamic agents
# written by DeepSeek Chat (honor call: The Adaptive Learner)

class BanditLearner:
    """Implements UCB1 bandit algorithm for feature selection."""
    
    def __init__(self, arms: List[str]):
        """
        Initialize with possible arms (features/actions).
        
        Args:
            arms: List of possible arms to pull
        """
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.total_pulls = 0

    def select_arm(self) -> str:
        """Select an arm using UCB1 algorithm."""
        unexplored = [arm for arm in self.arms if self.counts[arm] == 0]
        if unexplored:
            return unexplored[0]
            
        ucb_values = {
            arm: self.values[arm] + 
                 np.sqrt(2 * np.log(self.total_pulls) / self.counts[arm])
            for arm in self.arms
        }
        return max(ucb_values.items(), key=lambda x: x[1])[0]

    def update(self, arm: str, reward: float):
        """Update arm values based on observed reward."""
        self.counts[arm] += 1
        self.total_pulls += 1
        
        # Update moving average
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

# coordinator.py - Multi-agent coordination protocols
# written by DeepSeek Chat (honor call: The Negotiator)

class CoordinationProtocol:
    """Implements contract net protocol for multi-agent coordination."""
    
    def __init__(self, agents: List[str]):
        """
        Initialize with known agents.
        
        Args:
            agents: List of agent IDs in the system
        """
        self.agents = agents
        self.task_announcements = {}
        self.bids = defaultdict(dict)
        self.assignments = {}

    def announce_task(self, 
                     task: Dict[str, Any],
                     deadline: float = 5.0) -> Dict[str, Any]:
        """
        Announce a task to all agents using contract net protocol.
        
        Args:
            task: Dictionary describing the task
            deadline: Time in seconds to wait for bids
            
        Returns:
            Dictionary with assignment results
        """
        task_id = hashlib.sha256(str(task).encode()).hexdigest()[:8]
        self.task_announcements[task_id] = {
            'task': task,
            'announce_time': time.time(),
            'deadline': deadline,
            'status': 'open'
        }
        
        # Simulate bids from agents (in real system this would be async)
        for agent in self.agents:
            # TODO: Actual agents would compute bids based on capabilities
            bid = {
                'agent': agent,
                'capabilities': ["generic"],
                'estimated_cost': 1.0,
                'estimated_duration': 1.0,
                'confidence': 0.8
            }
            self.bids[task_id][agent] = bid
            
        return self._evaluate_bids(task_id)

    def _evaluate_bids(self, task_id: str) -> Dict[str, Any]:
        """Evaluate received bids and make assignments."""
        if task_id not in self.task_announcements:
            return {'status': 'error', 'message': 'Invalid task ID'}
            
        bids = self.bids[task_id]
        if not bids:
            self.task_announcements[task_id]['status'] = 'failed'
            return {'status': 'failed', 'message': 'No bids received'}
            
        # Simple selection - choose agent with lowest estimated cost
        selected_agent, best_bid = min(bids.items(), 
                                      key=lambda x: x[1]['estimated_cost'])
        
        self.assignments[task_id] = {
            'agent': selected_agent,
            'bid': best_bid,
            'assignment_time': time.time()
        }
        
        self.task_announcements[task_id]['status'] = 'assigned'
        
        return {
            'status': 'assigned',
            'agent': selected_agent,
            'details': best_bid
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample domain knowledge
    domain = {
        'actions': {
            'play_music': {
                'parameters': {'genre': 'str', 'volume': 'int'},
                'preconditions': {'has_speakers', 'music_available'},
                'effects': {'music_playing'},
                'duration': 0.1,
                'required_resources': ['speakers'],
                'reliability': 0.95
            },
            'dim_lights': {
                'preconditions': {'has_dimmable_lights'},
                'effects': {'lights_dimmed'},
                'duration': 0.5,
                'required_resources': ['smart_lights'],
                'reliability': 0.9
            }
        },
        'resources': ['speakers', 'smart_lights'],
        'agent_capabilities': {
            'media_agent': ['play_music'],
            'lighting_agent': ['dim_lights']
        }
    }
    
    # Initialize services
    planner = PlanningService(domain)
    coordinator = CoordinationProtocol(list(domain['agent_capabilities'].keys()))
    
    # Create and visualize a plan
    plan = planner.create_plan(
        goal="create_mood",
        tags={"has_speakers", "has_dimmable_lights", "evening_time"},
        available_resources=["speakers", "smart_lights"],
        priority=0.8
    )
    
    if plan:
        print("\nGenerated Plan:")
        for i, step in enumerate(plan.steps):
            print(f"{i+1}. {step.action} (takes {step.duration}h)")
        
        planner.visualize_plan(plan)
        
        # Test coordination
        print("\nCoordinating task...")
        result = coordinator.announce_task({
            'action': 'play_music',
            'parameters': {'genre': 'jazz', 'volume': 60}
        })
        print(f"Task assigned to: {result['agent']}")