# planning_service.py - Advanced HTN/A* Planner for GGFAI
# written by DeepSeek Chat (honor call: The Strategist)
# Enhanced with Tag Registry, Intent Tracker, Feature Tracker, Context Tracker, Analytics Tracker, and Model Adapter

"""
Advanced Hierarchical Task Network (HTN) Planner with A* Optimization

Features:
- Sophisticated HTN decomposition with multiple abstraction levels
- A* pathfinding for optimal action sequences
- Dynamic resource-aware planning
- Multi-threaded plan generation
- Intelligent plan caching
- Hardware-aware constraints
- Active learning integration
- Plan adaptation and repair
"""

from __future__ import annotations
import logging
import time
import asyncio
import psutil
import numpy as np
from typing import Dict, List, Optional, Set, Any, Tuple, FrozenSet
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from heapq import heappush, heappop
from collections import defaultdict
import networkx as nx
from pyvis.network import Network

from core.run_with_grace import run_with_grace
from core.tag_registry import TagRegistry
from trackers.intent_tracker import IntentTracker 
from trackers.feature_tracker import FeatureTracker
from trackers.context_tracker import ContextTracker
from trackers.analytics_tracker import AnalyticsTracker
from ml_layer.model_adapter import ModelAdapter
from resource_management.resource_predictor import ResourcePredictor

logger = logging.getLogger("GGFAI.planning")

@dataclass
class HTNMethod:
    """HTN decomposition method"""
    name: str
    preconditions: Set[str]
    effects: Set[str]
    subtasks: List[str]
    ordering_constraints: List[Tuple[int, int]]
    variables: Dict[str, Any]
    priority: float = 1.0
    reliability: float = 0.9

@dataclass
class PlanNode:
    """Node in planning search space"""
    state: FrozenSet[str]
    action: Optional[str] = None
    parent: Optional[PlanNode] = None
    depth: int = 0
    path_cost: float = 0.0
    heuristic_cost: float = 0.0
    
    @property
    def total_cost(self) -> float:
        return self.path_cost + self.heuristic_cost
    
    def __lt__(self, other: PlanNode) -> bool:
        return self.total_cost < other.total_cost

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
    trace_id: Optional[str] = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def update_status(self, new_status: PlanStatus) -> None:
        """Thread-safe status update"""
        with self._lock:
            self.status = new_status
            
class PlanningService:
    """
    Advanced HTN/A* planner with sophisticated features:
    - Multi-level task decomposition
    - Optimal action sequence planning
    - Resource-aware execution
    - Learning-based adaptation
    - Hardware-aware constraints
    """
    
    def __init__(self, 
                 domain_knowledge: Dict[str, Any],
                 tag_registry: TagRegistry,
                 intent_tracker: IntentTracker,
                 feature_tracker: FeatureTracker,
                 context_tracker: ContextTracker,
                 analytics_tracker: AnalyticsTracker,
                 model_adapter: Optional[ModelAdapter] = None):
        """Initialize planner with all required components"""
        self.domain = domain_knowledge
        self.tag_registry = tag_registry
        self.intent_tracker = intent_tracker
        self.feature_tracker = feature_tracker  
        self.context_tracker = context_tracker
        self.analytics_tracker = analytics_tracker
        self.model_adapter = model_adapter
        
        # Planning components
        self.htn_methods: Dict[str, List[HTNMethod]] = defaultdict(list)
        self.current_plans: Dict[str, Plan] = {}
        self.plan_history: List[Plan] = []
        self.plan_cache: Dict[str, Any] = {}
        
        # Resource management
        self.resource_predictor = ResourcePredictor()
        
        # Performance optimization
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = Lock()
        self._plan_graph = nx.DiGraph()
        
        # Initialize
        self._validate_domain()
        self._build_htn_methods()
        self._init_resource_monitoring()
        
        logger.info("Planning service initialized with all components")
        
    def _validate_domain(self) -> None:
        """Validate domain knowledge structure and constraints"""
        required_keys = {'actions', 'methods', 'resources', 'constraints'}
        if not all(key in self.domain for key in required_keys):
            raise ValueError(f"Domain missing required keys: {required_keys - set(self.domain.keys())}")
            
        # Validate actions
        for action, spec in self.domain['actions'].items():
            required_action_keys = {'preconditions', 'effects', 'parameters'}
            if not all(key in spec for key in required_action_keys):
                raise ValueError(f"Action {action} missing required keys")
                
        # Validate methods
        for method in self.domain['methods']:
            if 'subtasks' not in method or 'ordering' not in method:
                raise ValueError(f"Invalid method specification: {method}")

    def _build_htn_methods(self) -> None:
        """Build HTN decomposition methods with validation"""
        for method_spec in self.domain['methods']:
            method = HTNMethod(
                name=method_spec['name'],
                preconditions=set(method_spec['preconditions']),
                effects=set(method_spec['effects']),
                subtasks=method_spec['subtasks'],
                ordering_constraints=method_spec['ordering'],
                variables=method_spec.get('variables', {}),
                priority=float(method_spec.get('priority', 1.0)),
                reliability=float(method_spec.get('reliability', 0.9))
            )
            
            # Validate method
            self._validate_method(method)
            
            # Add to methods
            self.htn_methods[method.name].append(method)
            
    def _validate_method(self, method: HTNMethod) -> None:
        """Validate HTN method consistency"""
        # Check subtask existence
        for subtask in method.subtasks:
            if subtask not in self.domain['actions'] and not any(
                m.name == subtask for methods in self.htn_methods.values() for m in methods
            ):
                raise ValueError(f"Unknown subtask {subtask} in method {method.name}")
                
        # Validate ordering constraints
        task_indices = set(range(len(method.subtasks)))
        for i, j in method.ordering_constraints:
            if i not in task_indices or j not in task_indices:
                raise ValueError(f"Invalid ordering constraint ({i},{j}) in {method.name}")
                
        # Check effect consistency
        if not method.effects.issubset(
            set().union(*(
                self.domain['actions'][task]['effects'] 
                for task in method.subtasks 
                if task in self.domain['actions']
            ))
        ):
            logger.warning(f"Method {method.name} effects may be inconsistent with subtasks")

    def _init_resource_monitoring(self) -> None:
        """Initialize resource monitoring"""
        self.resource_stats = {
            'cpu_history': [],
            'memory_history': [],
            'resource_usage': defaultdict(list)
        }
        self.resource_thresholds = {
            'cpu_warning': 80.0,
            'memory_warning': 85.0,
            'plan_timeout': 30.0
        }
        
    @run_with_grace(max_attempts=2, timeout=10.0)
    async def create_plan(self, 
                         goal: str,
                         initial_state: Set[str],
                         available_resources: List[str],
                         priority: float = 0.7,
                         deadline: Optional[float] = None) -> Optional[Plan]:
        """
        Create execution plan using HTN planning with A* optimization
        
        Args:
            goal: Goal to achieve
            initial_state: Initial world state
            available_resources: Available resources
            priority: Plan priority (0-1)
            deadline: Optional deadline for plan completion
            
        Returns:
            Complete execution plan or None if planning fails
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(goal, initial_state)
            if cache_key in self.plan_cache:
                cached_plan = self.plan_cache[cache_key]
                if self._is_plan_valid(cached_plan, available_resources):
                    return cached_plan
                    
            # Get relevant methods
            applicable_methods = self._get_applicable_methods(goal, initial_state)
            if not applicable_methods:
                logger.warning(f"No applicable methods found for goal: {goal}")
                return None
                
            # Initialize search
            initial_node = PlanNode(state=frozenset(initial_state))
            goal_state = self._get_goal_state(goal)
            
            # Perform A* search through HTN decompositions
            plan_steps = await self._astar_search(
                initial_node,
                goal_state,
                applicable_methods,
                available_resources,
                deadline
            )
            
            if not plan_steps:
                return None
                
            # Create plan
            plan = Plan(
                goal=goal,
                steps=plan_steps,
                priority=priority,
                tags=initial_state,
                success_probability=self._estimate_success_probability(plan_steps)
            )
            
            # Cache plan
            self.plan_cache[cache_key] = plan
            
            # Update trackers
            self._update_trackers(plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"Plan creation failed: {str(e)}")
            return None
            
    async def execute_plan(self, plan: Plan) -> bool:
        """Execute plan with comprehensive monitoring and adaptation"""
        if not plan or not plan.steps:
            return False
            
        with plan._lock:
            plan.status = PlanStatus.ACTIVE
            logger.info(f"Executing plan for goal: {plan.goal}")
            
            try:
                execution_start = time.time()
                
                for step_idx, step in enumerate(plan.steps):
                    # Check resource availability
                    if not self._check_step_resources(step):
                        if not await self._attempt_plan_repair(plan, step_idx):
                            plan.status = PlanStatus.FAILED
                            return False
                    
                    # Execute step with monitoring
                    success = await self._execute_monitored_step(step, plan)
                    if not success:
                        # Attempt repair or adaptation
                        if not await self._handle_step_failure(plan, step_idx):
                            plan.status = PlanStatus.FAILED
                            return False
                            
                    # Update state and validate
                    self._update_world_state(step.effects)
                    if not self._validate_plan_state(plan):
                        if not await self._replan_from_current(plan, step_idx + 1):
                            plan.status = PlanStatus.FAILED
                            return False
                            
                    # Check timeout
                    if time.time() - execution_start > self.resource_thresholds['plan_timeout']:
                        logger.warning("Plan execution timeout")
                        plan.status = PlanStatus.FAILED
                        return False
                        
                plan.status = PlanStatus.COMPLETED
                self._log_plan_success(plan)
                return True
                
            except Exception as e:
                logger.error(f"Plan execution failed: {str(e)}")
                plan.status = PlanStatus.FAILED
                return False

    async def _execute_monitored_step(self, step: PlanStep, plan: Plan) -> bool:
        """Execute single step with resource monitoring"""
        try:
            # Start resource monitoring
            monitoring_task = asyncio.create_task(
                self._monitor_resources(step.action)
            )
            
            # Execute action
            logger.info(f"Executing step: {step.action}")
            step.status = PlanStatus.ACTIVE
            
            action_impl = self.domain["actions"][step.action]["implementation"]
            success = await self._run_with_timeout(
                action_impl(step.parameters),
                timeout=step.duration
            )
            
            # Stop monitoring
            monitoring_task.cancel()
            
            if not success:
                step.status = PlanStatus.FAILED
                return False
                
            # Update step status
            step.status = PlanStatus.COMPLETED
            
            # Record metrics
            self._record_step_metrics(step)
            
            return True
            
        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            step.status = PlanStatus.FAILED
            return False

    async def _run_with_timeout(self, coro, timeout: float):
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout)
        except asyncio.TimeoutError:
            return False

    async def _monitor_resources(self, action: str) -> None:
        """Monitor resource usage during action execution"""
        try:
            while True:
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                
                self.resource_stats['cpu_history'].append(cpu)
                self.resource_stats['memory_history'].append(memory)
                self.resource_stats['resource_usage'][action].append(
                    (cpu, memory, time.time())
                )
                
                # Check thresholds
                if cpu > self.resource_thresholds['cpu_warning']:
                    logger.warning(f"High CPU usage during {action}: {cpu}%")
                if memory > self.resource_thresholds['memory_warning']:
                    logger.warning(f"High memory usage during {action}: {memory}%")
                    
                await asyncio.sleep(0.1)  # Sample every 100ms
                
        except asyncio.CancelledError:
            pass

    async def _attempt_plan_repair(self, plan: Plan, failed_step: int) -> bool:
        """Attempt to repair plan after step failure"""
        try:
            # Get current state
            current_state = self._get_current_state()
            
            # Try to find alternative method
            remaining_goals = self._extract_remaining_goals(plan, failed_step)
            
            alternative_plan = await self.create_plan(
                goal=plan.goal,
                initial_state=current_state,
                available_resources=self._get_available_resources(),
                priority=plan.priority
            )
            
            if alternative_plan:
                # Replace remaining steps
                plan.steps[failed_step:] = alternative_plan.steps
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Plan repair failed: {str(e)}")
            return False

    async def _handle_step_failure(self, plan: Plan, failed_step: int) -> bool:
        """Handle step failure with multiple recovery strategies"""
        # Try simpler alternative
        if await self._try_simpler_alternative(plan, failed_step):
            return True
            
        # Try resource reallocation
        if await self._try_resource_reallocation(plan, failed_step):
            return True
            
        # Try decomposing step
        if await self._try_step_decomposition(plan, failed_step):
            return True
            
        # Try full replanning
        return await self._attempt_plan_repair(plan, failed_step)

    async def _try_simpler_alternative(self, plan: Plan, step_idx: int) -> bool:
        """Try to find simpler alternative for failed step"""
        failed_step = plan.steps[step_idx]
        
        # Get alternative methods sorted by complexity
        alternatives = self._get_alternative_methods(failed_step.action)
        alternatives.sort(key=lambda m: len(m.subtasks))
        
        for alt_method in alternatives:
            if self._is_valid_alternative(alt_method, failed_step, plan):
                # Replace step with alternative
                new_steps = self._decompose_method(alt_method)
                plan.steps[step_idx:step_idx+1] = new_steps
                return True
                
        return False

    def _decompose_method(self, method: HTNMethod) -> List[PlanStep]:
        """Decompose HTN method into primitive steps"""
        steps = []
        for subtask in method.subtasks:
            action = self.domain['actions'][subtask]
            step = PlanStep(
                action=subtask,
                parameters=action['parameters'],
                preconditions=set(action['preconditions']),
                effects=set(action['effects']),
                duration=float(action.get('duration', 1.0)),
                priority=method.priority,
                required_resources=action.get('required_resources', [])
            )
            steps.append(step)
        return steps

    async def _try_resource_reallocation(self, plan: Plan, step_idx: int) -> bool:
        """Try to reallocate resources to enable step execution"""
        step = plan.steps[step_idx]
        
        # Get current resource usage
        current_usage = self._get_resource_usage()
        
        # Find resources that could be freed
        freeable_resources = self._find_freeable_resources(current_usage)
        
        if freeable_resources:
            # Release less critical resources
            await self._release_resources(freeable_resources)
            
            # Retry step execution
            return await self._execute_monitored_step(step, plan)
            
        return False

    async def _try_step_decomposition(self, plan: Plan, step_idx: int) -> bool:
        """Try to decompose failed step into smaller steps"""
        step = plan.steps[step_idx]
        
        # Get decomposition methods
        decompositions = self._get_step_decompositions(step.action)
        
        for decomposition in decompositions:
            # Validate decomposition
            if self._validate_decomposition(decomposition, step, plan):
                # Replace step with decomposed steps
                plan.steps[step_idx:step_idx+1] = decomposition
                return True
                
        return False

    def _validate_plan_state(self, plan: Plan) -> bool:
        """Validate that plan state remains achievable"""
        current_state = self._get_current_state()
        
        for step in plan.steps:
            if not step.preconditions.issubset(current_state):
                return False
            current_state.update(step.effects)
            
        return True

    def _record_step_metrics(self, step: PlanStep) -> None:
        """Record metrics for completed step"""
        metrics = {
            'duration': time.time() - step.start_time,
            'cpu_usage': np.mean(self.resource_stats['cpu_history'][-10:]),
            'memory_usage': np.mean(self.resource_stats['memory_history'][-10:])
        }
        
        self.analytics_tracker.track_event(
            'step_completed',
            {
                'step': step.action,
                'metrics': metrics
            }
        )

    def _log_plan_success(self, plan: Plan) -> None:
        """Log successful plan completion"""
        duration = time.time() - plan.created_at
        
        self.analytics_tracker.track_event(
            'plan_completed',
            {
                'goal': plan.goal,
                'steps': len(plan.steps),
                'duration': duration,
                'success': True
            }
        )
        
        # Update learning
        if self.model_adapter:
            asyncio.create_task(
                self.model_adapter.learn_from_success(
                    plan.goal, 
                    [step.action for step in plan.steps],
                    duration
                )
            )

    def _check_step_resources(self, step: PlanStep) -> bool:
        """Check if required resources are available for step"""
        available = self._get_available_resources()
        return set(step.required_resources).issubset(set(available))

    def _get_current_state(self) -> Set[str]:
        """Get current world state"""
        state = set()
        
        # Get active context
        state.update(self.context_tracker.get_active_contexts())
        
        # Get feature states
        for feature in self.feature_tracker.get_all_features():
            if feature['status'] == 'available':
                state.add(feature['name'])
                
        return state

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage levels"""
        usage = {}
        
        try:
            # System resources
            usage['cpu'] = psutil.cpu_percent()
            usage['memory'] = psutil.virtual_memory().percent
            
            # Custom resources
            for resource in self.domain['resources']:
                usage[resource] = self.resource_predictor.get_usage(resource)
                
        except Exception as e:
            logger.error(f"Error getting resource usage: {str(e)}")
            
        return usage

    def visualize_plan(self, plan: Plan, filename: str = "plan_visualization.html") -> None:
        """Generate interactive visualization of plan structure"""
        if not plan or not plan.steps:
            return
            
        try:
            net = Network(height="750px", width="100%", directed=True)
            
            # Add nodes for each step
            for i, step in enumerate(plan.steps):
                color = {
                    PlanStatus.COMPLETED: "#98FB98",
                    PlanStatus.FAILED: "#FFA07A",
                    PlanStatus.ACTIVE: "#87CEEB",
                    PlanStatus.DRAFT: "#DCDCDC"
                }.get(step.status, "#FFFFFF")
                
                label = f"{i+1}. {step.action}\n({step.duration:.1f}s)"
                net.add_node(
                    i, 
                    label=label,
                    title=str(step.parameters),
                    color=color
                )
                
                # Add edge to previous step
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
                net.add_edge(
                    len(plan.steps)-1,
                    len(plan.steps),
                    width=3
                )
            
            # Save visualization
            net.show(filename)
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")

    def get_plan_metrics(self, plan: Plan) -> Dict[str, Any]:
        """Get comprehensive metrics for plan"""
        return {
            'steps': len(plan.steps),
            'completed_steps': sum(1 for s in plan.steps if s.status == PlanStatus.COMPLETED),
            'failed_steps': sum(1 for s in plan.steps if s.status == PlanStatus.FAILED),
            'duration': time.time() - plan.created_at,
            'success_probability': plan.success_probability,
            'resource_usage': {
                action: np.mean(usage) 
                for action, usage in self.resource_stats['resource_usage'].items()
                if usage
            },
            'state_changes': len(plan.tags),
            'adaptations': getattr(plan, 'adaptation_count', 0)
        }

    async def _astar_search(self,
                           initial_node: PlanNode,
                           goal_state: Set[str],
                           methods: List[HTNMethod],
                           available_resources: List[str],
                           deadline: Optional[float] = None) -> Optional[List[PlanStep]]:
        """
        A* search through HTN decomposition space
        
        Features:
        - Optimal path finding through state space
        - Dynamic resource allocation
        - Deadline-aware processing
        - Parallel method evaluation
        - Learning-based heuristics
        """
        open_set: List[PlanNode] = [initial_node]
        closed_set: Set[FrozenSet[str]] = set()
        came_from: Dict[FrozenSet[str], PlanNode] = {}
        g_score: Dict[FrozenSet[str], float] = {initial_node.state: 0}
        f_score: Dict[FrozenSet[str], float] = {
            initial_node.state: self._heuristic(initial_node.state, goal_state)
        }
        
        # Enable parallel evaluation for large method sets
        parallel_threshold = 10
        executor = ThreadPoolExecutor(max_workers=4)
        
        while open_set:
            # Check deadline
            if deadline and time.time() > deadline:
                logger.warning("A* search deadline exceeded")
                return self._extract_partial_plan(current, came_from)
            
            # Get most promising node
            current = min(open_set, key=lambda n: f_score.get(n.state, float('inf')))
            
            # Check if goal reached
            if goal_state.issubset(current.state):
                return self._extract_plan(current, came_from)
            
            open_set.remove(current)
            closed_set.add(current.state)
            
            # Get applicable methods and evaluate in parallel if many
            applicable = [m for m in methods if self._is_method_applicable(m, current.state)]
            
            if len(applicable) > parallel_threshold:
                # Parallel evaluation
                futures = []
                for method in applicable:
                    futures.append(
                        executor.submit(
                            self._evaluate_method,
                            method, 
                            current,
                            available_resources,
                            goal_state
                        )
                    )
                successors = []
                for future in futures:
                    if result := future.result():
                        successors.append(result)
            else:
                # Sequential evaluation
                successors = [
                    self._evaluate_method(m, current, available_resources, goal_state)
                    for m in applicable
                ]
                successors = [s for s in successors if s]
            
            # Process successors
            for successor, cost in successors:
                if successor.state in closed_set:
                    continue
                
                tentative_g = g_score[current.state] + cost
                
                if successor.state not in g_score or tentative_g < g_score[successor.state]:
                    came_from[successor.state] = current
                    g_score[successor.state] = tentative_g
                    f_score[successor.state] = tentative_g + successor.heuristic_cost
                    
                    if successor not in open_set:
                        heappush(open_set, successor)
                        
                        # Update search metrics
                        self.analytics_tracker.track_metric(
                            'search_expansion',
                            {
                                'depth': successor.depth,
                                'branching': len(applicable),
                                'heuristic': successor.heuristic_cost
                            }
                        )
        
        return None
        
    def _evaluate_method(self,
                        method: HTNMethod,
                        current: PlanNode,
                        available_resources: List[str],
                        goal_state: Set[str]) -> Optional[Tuple[PlanNode, float]]:
        """Evaluate HTN method application"""
        try:
            # Check resource constraints
            if not self._check_method_resources(method, available_resources):
                return None
            
            # Apply method effects
            new_state = set(current.state)
            new_state.update(method.effects)
            
            # Create successor node
            successor = PlanNode(
                state=frozenset(new_state),
                action=method.name,
                parent=current,
                depth=current.depth + 1
            )
            
            # Calculate costs
            path_cost = self._calculate_path_cost(method, current.state)
            heuristic_cost = self._heuristic(successor.state, goal_state)
            
            successor.path_cost = current.path_cost + path_cost
            successor.heuristic_cost = heuristic_cost
            
            return successor, path_cost
            
        except Exception as e:
            logger.error(f"Method evaluation failed: {str(e)}")
            return None
            
    def _calculate_path_cost(self, method: HTNMethod, current_state: FrozenSet[str]) -> float:
        """Calculate cost of applying method"""
        base_cost = 1.0
        
        # Adjust for complexity
        complexity_factor = len(method.subtasks) / 5.0  # Normalize by typical decomposition size
        
        # Adjust for reliability
        reliability_cost = -np.log(method.reliability)  # Higher cost for less reliable methods
        
        # Adjust for precondition satisfaction
        unsatisfied = len(method.preconditions - set(current_state))
        precondition_cost = unsatisfied * 0.5
        
        # Get learning-based cost adjustment if available
        if self.model_adapter:
            success_rate = self.model_adapter.get_success_rate(method.name)
            learning_factor = 1.0 / (0.1 + 0.9 * success_rate)  # Avoid division by zero
        else:
            learning_factor = 1.0
            
        return base_cost * (
            1.0 + 
            0.3 * complexity_factor +
            0.3 * reliability_cost + 
            0.2 * precondition_cost
        ) * learning_factor
        
    def _check_method_resources(self, method: HTNMethod, available: List[str]) -> bool:
        """Check if method's resource requirements can be met"""
        required = set()
        
        # Get resource requirements for all subtasks
        for task in method.subtasks:
            if task in self.domain['actions']:
                action = self.domain['actions'][task]
                required.update(action.get('required_resources', []))
                
        return required.issubset(set(available))
        
    def _extract_plan(self, goal_node: PlanNode, came_from: Dict[FrozenSet[str], PlanNode]) -> List[PlanStep]:
        """Extract plan steps from A* search result"""
        steps = []
        current = goal_node
        
        while current.parent:
            if current.action:
                method = next(
                    (m for m in self.htn_methods[current.action]), 
                    None
                )
                if method:
                    # Convert method to primitive steps
                    primitive_steps = self._decompose_to_primitive(
                        method,
                        current.state
                    )
                    steps.extend(primitive_steps)
            current = came_from[current.state]
            
        return list(reversed(steps))
        
    def _decompose_to_primitive(self, method: HTNMethod, state: FrozenSet[str]) -> List[PlanStep]:
        """Decompose HTN method into primitive steps"""
        primitive_steps = []
        
        # Create topological ordering of subtasks
        ordering = self._topological_sort(
            method.subtasks,
            method.ordering_constraints
        )
        
        # Convert each subtask to primitive step
        for task in ordering:
            if task in self.domain['actions']:
                # Primitive action
                action = self.domain['actions'][task]
                step = PlanStep(
                    action=task,
                    parameters=self._bind_parameters(
                        action['parameters'],
                        method.variables,
                        state
                    ),
                    preconditions=set(action['preconditions']),
                    effects=set(action['effects']),
                    duration=float(action.get('duration', 1.0)),
                    priority=method.priority,
                    required_resources=action.get('required_resources', [])
                )
                primitive_steps.append(step)
            else:
                # Compound task - recursive decomposition
                subtask_method = self._find_applicable_method(task, state)
                if subtask_method:
                    subtask_steps = self._decompose_to_primitive(
                        subtask_method,
                        state
                    )
                    primitive_steps.extend(subtask_steps)
                    
                    # Update state
                    state = frozenset(
                        set(state) | subtask_method.effects
                    )
                    
        return primitive_steps
        
    def _topological_sort(self,
                         tasks: List[str],
                         ordering: List[Tuple[int, int]]) -> List[str]:
        """Create valid topological ordering of tasks"""
        # Build dependency graph
        graph = defaultdict(set)
        for i, j in ordering:
            graph[tasks[i]].add(tasks[j])
            
        # Perform topological sort
        visited = set()
        temp_mark = set()
        order = []
        
        def visit(task):
            if task in temp_mark:
                raise ValueError("Cycle detected in task ordering")
            if task in visited:
                return
                
            temp_mark.add(task)
            
            for dep in graph[task]:
                visit(dep)
                
            temp_mark.remove(task)
            visited.add(task)
            order.append(task)
            
        for task in tasks:
            if task not in visited:
                visit(task)
                
        return list(reversed(order))
        
    def _bind_parameters(self,
                        params: Dict[str, Any],
                        variables: Dict[str, Any],
                        state: FrozenSet[str]) -> Dict[str, Any]:
        """Bind method variables to action parameters"""
        bound = {}
        
        for name, spec in params.items():
            if isinstance(spec, str) and spec.startswith('?'):
                # Variable reference
                if spec[1:] in variables:
                    bound[name] = variables[spec[1:]]
                else:
                    # Try to bind from state
                    candidates = [
                        s for s in state
                        if s.startswith(f"{name}=")
                    ]
                    if candidates:
                        bound[name] = candidates[0].split('=')[1]
            else:
                # Constant value
                bound[name] = spec
                
        return bound