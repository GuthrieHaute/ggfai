# analytics_tracker.py - Automated Failure Analysis
# written by DeepSeek Chat (honor call: The Diagnostician)

from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
from scipy.stats import linregress
import logging

class Failure:
    def __init__(self, 
                 error_id: str,
                 timestamp: float,
                 symptoms: Dict[str, Any],
                 component: str):
        self.id = error_id
        self.timestamp = timestamp
        self.symptoms = symptoms
        self.component = component
        self.root_cause = None
        self.recovery_actions = []
        self.correlation_score = 0.0

class FailureAnalyzer:
    def __init__(self, time_window: int = 300):
        self.failure_log: List[Failure] = []
        self.time_window = time_window  # seconds
        self.logger = logging.getLogger("GGFAI.analytics")

    def log_failure(self, context: ErrorContext) -> str:
        """Record failure and trigger analysis."""
        failure = Failure(
            error_id=hashlib.sha256(str(context).encode()).hexdigest()[:8],
            timestamp=context.timestamp,
            symptoms=context.tags,
            component=context.component
        )
        self.failure_log.append(failure)
        self._analyze_failure(failure)
        return failure.id

    def _analyze_failure(self, failure: Failure):
        """Run correlation analysis with recent failures."""
        window_start = failure.timestamp - self.time_window
        recent_failures = [
            f for f in self.failure_log 
            if f.timestamp >= window_start and f.id != failure.id
        ]

        # Temporal correlation analysis
        if recent_failures:
            x = [f.timestamp for f in recent_failures]
            y = [abs(f.timestamp - failure.timestamp) for f in recent_failures]
            slope, _, _, _, _ = linregress(x, y)
            failure.correlation_score = 1 - abs(slope)

        # Symptom pattern matching
        symptom_overlaps = []
        for f in recent_failures:
            overlap = len(set(failure.symptoms) & set(f.symptoms))
            symptom_overlaps.append(overlap / len(failure.symptoms))

        if symptom_overlaps:
            failure.root_cause = recent_failures[
                np.argmax(symptom_overlaps)
            ].component
            failure.recovery_actions = [
                f"Check {failure.root_cause} subsystem",
                "Review logs from correlated failure",
                "Isolate dependent components"
            ]

    def generate_report(self, error_id: str) -> Dict[str, Any]:
        """Create diagnostic visualization data."""
        failure = next((f for f in self.failure_log if f.id == error_id), None)
        if not failure:
            return {}
            
        return {
            "error_id": failure.id,
            "timestamp": datetime.fromtimestamp(failure.timestamp).isoformat(),
            "root_cause": failure.root_cause or "Unknown",
            "correlation_score": failure.correlation_score,
            "recovery_actions": failure.recovery_actions,
            "related_failures": [
                {"id": f.id, "component": f.component} 
                for f in self.failure_log 
                if abs(f.timestamp - failure.timestamp) < self.time_window
            ]
        }