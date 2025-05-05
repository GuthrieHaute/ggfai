{
  "ai_name": "DeepSeek Chat",
  "task_id": 4,
  "file": "coordinator.py",
  "description": "Implemented negotiation protocols for multi-agent task coordination.",
  "honor_call": "The Framework Architect"
}

#written by DeepSeek Chat (honor call: The Framework Architect)

"""
Multi-Agent Coordinator with Learning Integration
"""
import numpy as np
import time
from typing import Dict, List, Optional, Set
from threading import Lock

from ...core.tag_registry import TagRegistry
from ...resource_management.hardware_shim import HardwareMonitor
from .learning import LearningService

# Constants
MAX_TASKS = 10  # Maximum tasks per agent
TOTAL_MEMORY = 1024  # Default total memory in MB

class Coordinator:
    def __init__(self, tag_registry: TagRegistry, learning_service: LearningService = None):
        self.tag_registry = tag_registry
        self.learning_service = learning_service
        self.hardware_monitor = HardwareMonitor()
        self.agents = {}
        self._tasks_in_progress = {}
        self._lock = Lock()
        
    def register_agent(self, agent_id: str, capabilities: Set[str]) -> bool:
        """Register an agent with capabilities"""
        with self._lock:
            if agent_id in self.agents:
                return False
                
            self.agents[agent_id] = {
                'capabilities': capabilities,
                'current_tasks': [],
                'success_rate': 1.0,
                'failure_count': 0,
                'last_heartbeat': time.time()
            }
            return True
        
    def claim_task(self, task: Dict, bidding_agents: List[str]) -> Optional[str]:
        """Coordinate task assignment using priority-based bidding and learning"""
        if not bidding_agents:
            return None
            
        with self._lock:
            # Get system load info
            cpu_load = self.hardware_monitor.get_cpu_load()
            mem_avail = self.hardware_monitor.get_available_memory()
            
            # Build context for learning service
            context = {
                'cpu_load': cpu_load,
                'memory_available': mem_avail,
                'time': time.time(),
                'priority': task.get('priority', 0.5),
                'deadline': task.get('deadline')
            }
            
            bids = []
            for agent_id in bidding_agents:
                agent = self.agents.get(agent_id)
                if not agent:
                    continue
                    
                # Calculate bid components
                capability_match = sum(
                    1 for req in task['requirements'] 
                    if req in agent['capabilities']
                )
                current_load = len(agent['current_tasks'])
                
                # Get learning adjustment if available
                learning_score = 1.0
                if self.learning_service:
                    metrics = self.learning_service.get_action_metrics(agent_id)
                    if metrics:
                        learning_score = metrics['success_rate']
                
                # Calculate final bid score
                load_factor = 0.5 * (cpu_load / 100) + 0.5 * (1 - mem_avail / TOTAL_MEMORY)
                bid_score = (
                    capability_match * 0.4 +
                    (1 - current_load/MAX_TASKS) * 0.3 +
                    learning_score * 0.3
                ) * (1 - load_factor)
                
                bids.append((bid_score, agent_id))
            
            if not bids:
                return None
                
            # Select winner and update state
            winning_bid, winning_agent = max(bids, key=lambda x: x[0])
            
            # Update agent state
            agent = self.agents[winning_agent]
            agent['current_tasks'].append(task['id'])
            agent['last_heartbeat'] = time.time()
            
            # Track task for completion monitoring
            self._tasks_in_progress[task['id']] = {
                'agent': winning_agent,
                'start_time': time.time(),
                'priority': task.get('priority', 0.5)
            }
            
            # Log assignment
            self.tag_registry.log_task_assignment(
                task_id=task['id'],
                agent_id=winning_agent,
                bid_score=winning_bid,
                context=context
            )
            
            return winning_agent
            
    def complete_task(self, task_id: str, agent_id: str, success: bool = True) -> bool:
        """Handle task completion and update learning"""
        with self._lock:
            agent = self.agents.get(agent_id)
            if not agent or task_id not in agent['current_tasks']:
                return False
                
            # Update agent state
            agent['current_tasks'].remove(task_id)
            if success:
                agent['success_rate'] = 0.9 * agent['success_rate'] + 0.1
                agent['failure_count'] = max(0, agent['failure_count'] - 1)
            else:
                agent['success_rate'] = 0.9 * agent['success_rate']
                agent['failure_count'] += 1
                
            # Update learning service
            if self.learning_service and task_id in self._tasks_in_progress:
                task_info = self._tasks_in_progress[task_id]
                reward = 1.0 if success else 0.0
                
                # Adjust reward based on timing if deadline was specified
                if 'deadline' in task_info:
                    time_taken = time.time() - task_info['start_time']
                    time_available = task_info['deadline'] - task_info['start_time']
                    if time_taken > time_available:
                        reward *= 0.5
                
                self.learning_service.update(
                    action=agent_id,
                    reward=reward,
                    context={'priority': task_info['priority']}
                )
                
                del self._tasks_in_progress[task_id]
            
            return True
            
    def heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat timestamp"""
        with self._lock:
            if agent_id not in self.agents:
                return False
            self.agents[agent_id]['last_heartbeat'] = time.time()
            return True
            
    def check_agent_health(self) -> List[str]:
        """Check for stale/failed agents"""
        stale_agents = []
        current_time = time.time()
        
        with self._lock:
            for agent_id, agent in self.agents.items():
                if current_time - agent['last_heartbeat'] > 300:  # 5 minutes
                    stale_agents.append(agent_id)
                elif agent['failure_count'] > 3:
                    stale_agents.append(agent_id)
                    
        return stale_agents