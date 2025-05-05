# Filename: generate_explanation.py
# Description: Generates text-based plan explanations and enhanced pyvis visualizations for GGFAI.
#written by GuthrieHaute (honor call: The Visionary)

import numpy as np
import networkx as nx
from pyvis.network import Network
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, List
import json

from planning_service import PlanningService
from tag_analyzer import TagAnalyzer
from analytics_tracker import AnalyticsTracker

class ExplanationLevel(Enum):
    SIMPLE = auto()      # Basic summary
    STANDARD = auto()    # Default narrative with key details
    TECHNICAL = auto()   # Includes confidence, alternatives
    DEVELOPER = auto()   # Includes raw decision data

@dataclass
class ExplanationConfig:
    level: ExplanationLevel = ExplanationLevel.STANDARD
    use_physics_viz: bool = True

class ExplanationGenerator:
    def __init__(self, planning_service: PlanningService, tag_analyzer: TagAnalyzer, analytics_tracker: Optional[AnalyticsTracker] = None):
        self.planning_service = planning_service
        self.tag_analyzer = tag_analyzer
        self.analytics = analytics_tracker or AnalyticsTracker()

    def _generate_base_narrative(self, trace_id: str, context: Dict, goal_data: Dict, plan_data: Dict) -> tuple:
        if not plan_data:
            return "No plan data available.", 0.0, [], 0, "unknown", plan_data

        ranked_tags = self.tag_analyzer.prioritize_tags(context)
        top_tag_info = f"'{ranked_tags[0]['tag_id']}' (Score: {ranked_tags[0]['score']:.2f})" if ranked_tags else "general context"

        intent = goal_data.get('intent', 'the request')
        actions = getattr(plan_data, 'actions', plan_data.get('actions', []))
        primary_action = next((a for a in actions if getattr(a, 'primary', a.get('primary', False))), None)
        primary_action_name = primary_action.name if primary_action else "take action"

        confidence = getattr(plan_data, 'confidence', 0.75)
        exec_time = getattr(plan_data, 'execution_time_ms', 0)
        hw_impact = getattr(plan_data, 'hardware_impact', 'unknown')
        alternatives = getattr(plan_data, 'alternatives', [])

        narrative = (
            f"To fulfill '{intent}', the agent chose to '{primary_action_name}'.\n"
            f"Key factors:\n"
            f"- Context: {top_tag_info} was most relevant.\n"
            f"- Confidence: Action selected with {confidence*100:.0f}% confidence."
        )
        return narrative, confidence, alternatives, exec_time, hw_impact, plan_data

    def generate_narrative(self, trace_id: str, context: Dict, goal_data: Dict, config: ExplanationConfig = ExplanationConfig()) -> str:
        plan_data = self.planning_service.get_plan(trace_id)
        narrative, confidence, alternatives, exec_time, hw_impact, plan_data = self._generate_base_narrative(trace_id, context, goal_data, plan_data)

        if config.level == ExplanationLevel.SIMPLE:
            return narrative.split('\n')[0]
        elif config.level >= ExplanationLevel.TECHNICAL:
            narrative += (
                f"\n- Alternatives: {len(alternatives)} other options considered."
                f"\n- Performance: Took {exec_time}ms with '{hw_impact}' hardware impact."
            )
        if config.level == ExplanationLevel.DEVELOPER:
            narrative += f"\n- Trace ID: {trace_id}"
            debug_data = {
                "decision_matrix": getattr(plan_data, 'decision_matrix', None),
                "q_values": getattr(plan_data, 'q_values', None),
                "entropy": getattr(plan_data, 'entropy', None),
                "reward_calculation": getattr(plan_data, 'reward_calculation', None)
            }
            debug_data = {k: v for k, v in debug_data.items() if v is not None}
            narrative += f"\n- Debug Data: {debug_data if debug_data else 'No detailed data available.'}"

        self.analytics.log_event(
            event_type="explanation_generated",
            data={
                "trace_id": trace_id,
                "explanation_level": config.level.name,
                "narrative_length": len(narrative)
            }
        )
        return narrative

    def generate_visualization(self, trace_id: str, context: Dict, goal_data: Dict, config: ExplanationConfig = ExplanationConfig()) -> Network:
        plan_data = self.planning_service.get_plan(trace_id)
        ranked_tags = self.tag_analyzer.prioritize_tags(context)
        actions = getattr(plan_data, 'actions', plan_data.get('actions', [])) if plan_data else []

        net = Network(height="600px", width="100%", directed=True, notebook=False)
        nodes, edges = [], []

        nodes.append({"id": "goal", "label": f"Goal: {goal_data.get('intent', 'Unknown')}", "color": "#FF6B6B", "size": 25, "shape": "diamond"})
        nodes.append({"id": "decision", "label": "Agent Decision", "color": "#4ECDC4", "size": 20, "shape": "star"})
        edges.append({"from": "goal", "to": "decision", "arrows": "to"})

        for key, value in context.items():
            nodes.append({"id": f"ctx_{key}", "label": f"Context: {key}={value}", "color": "#FFE66D", "size": 15})
            edges.append({"from": f"ctx_{key}", "to": "decision", "dashes": True})

        for tag in ranked_tags[:5]:
            nodes.append({"id": f"tag_{tag['tag_id']}", "label": f"Tag: {tag['tag_id']}\nScore: {tag['score']:.2f}", "color": "#45B7D1", "size": 10 + tag['score'] * 5})
            edges.append({"from": f"tag_{tag['tag_id']}", "to": "decision", "arrows": "to"})

        for i, action in enumerate(actions):
            action_name = getattr(action, 'name', action.get('name', f'Action {i}'))
            is_primary = getattr(action, 'primary', action.get('primary', False))
            nodes.append({"id": f"action_{i}", "label": f"Action: {action_name}", "color": "#8AC926" if is_primary else "#A0CFA0", "size": 15 if is_primary else 12, "shape": "box"})
            edges.append({"from": "decision", "to": f"action_{i}", "arrows": "to"})

        for node in nodes:
            net.add_node(node['id'], label=node['label'], color=node['color'], size=node['size'], shape=node['shape'])
        for edge in edges:
            net.add_edge(edge['from'], edge['to'], arrows=edge['arrows'], dashes=edge.get('dashes', False))

        options = {
            "physics": {
                "solver": "barnesHut" if config.use_physics_viz else "forceAtlas2Based",
                "barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.4, "springLength": 95, "springConstant": 0.04, "damping": 0.09, "avoidOverlap": 0.2},
                "forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08, "damping": 0.4, "avoidOverlap": 0.1},
                "minVelocity": 0.75,
                "stabilization": {"iterations": 150}
            }
        }
        net.set_options(json.dumps(options))

        self.analytics.log_event(
            event_type="visualization_generated",
            data={
                "trace_id": trace_id,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "physics_used": options["physics"]["solver"]
            }
        )
        return net

    def generate_for_web(self, trace_id: str, context: Dict, goal_data: Dict, config: ExplanationConfig = ExplanationConfig()) -> Dict:
        narrative = self.generate_narrative(trace_id, context, goal_data, config)
        vis_net = self.generate_visualization(trace_id, context, goal_data, config)
        vis_data = {"nodes": vis_net.get_nodes(), "edges": vis_net.get_edges(), "options": json.loads(vis_net.options)}
        return {"narrative": narrative, "visualization_data": vis_data}