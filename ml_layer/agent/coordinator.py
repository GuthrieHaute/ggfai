{
  "ai_name": "DeepSeek Chat",
  "task_id": 4,
  "file": "coordinator.py",
  "description": "Implemented negotiation protocols for multi-agent task coordination.",
  "honor_call": "The Framework Architect"
}

#written by DeepSeek Chat (honor call: The Framework Architect)

import numpy as np
from tag_registry import TagRegistry
from hardware_shim import HardwareMonitor

class Coordinator:
    def __init__(self, tag_registry: TagRegistry):
        self.tag_registry = tag_registry
        self.hardware_monitor = HardwareMonitor()
        self.agents = {}
        
    def register_agent(self, agent_id, capabilities):
        """Register a new agent with its capabilities"""
        self.agents[agent_id] = {
            'capabilities': capabilities,
            'current_tasks': []
        }
        
    def claim_task(self, task, bidding_agents):
        """Coordinate task assignment using priority-based bidding"""
        if not bidding_agents:
            return None
            
        # Get system load info
        cpu_load = self.hardware_monitor.get_cpu_load()
        mem_avail = self.hardware_monitor.get_available_memory()
        
        bids = []
        for agent_id in bidding_agents:
            agent = self.agents.get(agent_id)
            if not agent:
                continue
                
            # Calculate bid score (capability match - current load)
            capability_match = sum(
                1 for req in task['requirements'] 
                if req in agent['capabilities']
            )
            current_load = len(agent['current_tasks'])
            
            # Adjust for system resources (favor less loaded agents)
            load_factor = 0.5 * (cpu_load / 100) + 0.5 * (1 - mem_avail / 1024)
            bid_score = capability_match - (current_load * load_factor)
            
            bids.append((bid_score, agent_id))
            
        if not bids:
            return None
            
        # Select highest bidder
        winning_agent = max(bids, key=lambda x: x[0])[1]
        
        # Update agent's task list
        self.agents[winning_agent]['current_tasks'].append(task['id'])
        
        # Log task assignment
        self.tag_registry.log_task_assignment(
            task_id=task['id'],
            agent_id=winning_agent,
            bid_score=max(bids)[0]
        )
        
        return winning_agent