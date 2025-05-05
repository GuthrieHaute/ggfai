# analytics_tracker.py - Analytics and Logging System for GGFAI
# written by DeepSeek Chat (honor call: The Data Historian)

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from collections import deque, defaultdict
import statistics
from enum import Enum

from ..core.tag_registry import Tag

# Constants
MAX_EVENT_HISTORY = 10000  # Maximum events to keep in memory
RETENTION_DAYS = 7  # Default retention period for events
ANOMALY_THRESHOLD = 2.0  # Standard deviations for anomaly detection

class EventSeverity(Enum):
    """Event severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class AnalyticsTracker:
    """
    Tracks system events, errors, and performance metrics.
    Provides automated failure correlation analysis.
    """
    
    def __init__(self, max_events: int = MAX_EVENT_HISTORY):
        self._lock = threading.RLock()
        self.logger = logging.getLogger("GGFAI.analytics_tracker")
        
        # Event storage
        self.events = deque(maxlen=max_events)
        self.event_types = defaultdict(int)  # type -> count
        self.event_sources = defaultdict(int)  # source -> count
        
        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))  # metric_name -> values
        self.metric_stats = {}  # metric_name -> {mean, stddev, min, max}
        
        # Correlation tracking
        self.error_sequences = defaultdict(list)  # error_type -> [preceding_events]
        self.correlation_scores = {}  # (event_type, error_type) -> correlation_score
        
        # Start background analysis
        self._start_analysis_thread()
        self.logger.info("Analytics tracker initialized")

    def _start_analysis_thread(self):
        """Start background analysis thread."""
        def analysis_loop():
            while True:
                try:
                    self._update_metric_stats()
                    self._analyze_correlations()
                except Exception as e:
                    self.logger.error(f"Analysis failed: {e}")
                time.sleep(3600)  # Run hourly
        
        threading.Thread(target=analysis_loop, daemon=True).start()

    def log_event(self, 
                 event_type: str, 
                 source: str, 
                 severity: EventSeverity = EventSeverity.INFO,
                 details: Dict[str, Any] = None) -> str:
        """
        Log a system event.
        
        Args:
            event_type: Type of event
            source: Component that generated the event
            severity: Event severity level
            details: Additional event details
            
        Returns:
            Event ID
        """
        with self._lock:
            event_id = f"{int(time.time())}_{source}_{event_type}"
            event = {
                "id": event_id,
                "timestamp": datetime.utcnow(),
                "type": event_type,
                "source": source,
                "severity": severity.value,
                "details": details or {}
            }
            
            self.events.append(event)
            self.event_types[event_type] += 1
            self.event_sources[source] += 1
            
            # Log to system logger as well
            log_message = f"{event_type} from {source}"
            if details:
                log_message += f": {json.dumps(details)}"
                
            if severity == EventSeverity.DEBUG:
                self.logger.debug(log_message)
            elif severity == EventSeverity.INFO:
                self.logger.info(log_message)
            elif severity == EventSeverity.WARNING:
                self.logger.warning(log_message)
            elif severity == EventSeverity.ERROR:
                self.logger.error(log_message)
                self._track_error_sequence(event_type)
            elif severity == EventSeverity.CRITICAL:
                self.logger.critical(log_message)
                self._track_error_sequence(event_type)
                
            return event_id

    def _track_error_sequence(self, error_type: str):
        """Track events preceding an error for correlation analysis."""
        with self._lock:
            # Get recent events (last 20)
            recent_events = list(self.events)[-20:-1] if len(self.events) > 20 else list(self.events)[:-1]
            
            # Store sequence
            sequence = [(e["type"], e["source"]) for e in recent_events]
            self.error_sequences[error_type].append(sequence)
            
            # Limit sequence history
            if len(self.error_sequences[error_type]) > 100:
                self.error_sequences[error_type].pop(0)

    def log_metric(self, name: str, value: float, source: str = "system"):
        """
        Log a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            source: Component that generated the metric
        """
        with self._lock:
            self.metrics[name].append((datetime.utcnow(), value))
            
            # Check for anomalies if we have stats
            if name in self.metric_stats:
                stats = self.metric_stats[name]
                if abs(value - stats["mean"]) > stats["stddev"] * ANOMALY_THRESHOLD:
                    self.log_event(
                        event_type="metric_anomaly",
                        source=source,
                        severity=EventSeverity.WARNING,
                        details={
                            "metric": name,
                            "value": value,
                            "mean": stats["mean"],
                            "stddev": stats["stddev"],
                            "deviation": abs(value - stats["mean"]) / stats["stddev"]
                        }
                    )

    def add_tag(self, tag: Tag) -> str:
        """
        Add an analytics tag.
        
        Args:
            tag: Analytics tag
            
        Returns:
            Event ID
        """
        with self._lock:
            event_type = tag.intent
            source = tag.category
            severity = EventSeverity[tag.metadata.get("severity", "INFO").upper()]
            
            return self.log_event(
                event_type=event_type,
                source=source,
                severity=severity,
                details=tag.metadata
            )

    def get_recent_events(self, 
                         limit: int = 100, 
                         event_type: str = None,
                         source: str = None,
                         min_severity: EventSeverity = None) -> List[Dict[str, Any]]:
        """
        Get recent events with optional filtering.
        
        Args:
            limit: Maximum events to return
            event_type: Filter by event type
            source: Filter by source
            min_severity: Minimum severity level
            
        Returns:
            List of matching events
        """
        with self._lock:
            filtered = list(self.events)
            
            if event_type:
                filtered = [e for e in filtered if e["type"] == event_type]
            if source:
                filtered = [e for e in filtered if e["source"] == source]
            if min_severity:
                filtered = [e for e in filtered if e["severity"] >= min_severity.value]
                
            return list(reversed(filtered))[:limit]

    def get_metric_history(self, 
                          name: str, 
                          hours: int = 24) -> List[Tuple[datetime, float]]:
        """
        Get history for a specific metric.
        
        Args:
            name: Metric name
            hours: Hours of history to return
            
        Returns:
            List of (timestamp, value) tuples
        """
        with self._lock:
            if name not in self.metrics:
                return []
                
            threshold = datetime.utcnow() - timedelta(hours=hours)
            return [(ts, val) for ts, val in self.metrics[name] if ts >= threshold]

    def _update_metric_stats(self):
        """Update statistical summaries of metrics."""
        with self._lock:
            for name, values in self.metrics.items():
                if not values:
                    continue
                    
                # Extract just the values (not timestamps)
                just_values = [v for _, v in values]
                
                try:
                    self.metric_stats[name] = {
                        "mean": statistics.mean(just_values),
                        "stddev": statistics.stdev(just_values) if len(just_values) > 1 else 0,
                        "min": min(just_values),
                        "max": max(just_values),
                        "count": len(just_values),
                        "updated_at": datetime.utcnow()
                    }
                except Exception as e:
                    self.logger.error(f"Failed to calculate stats for {name}: {e}")

    def _analyze_correlations(self):
        """Analyze event correlations to identify potential causes of errors."""
        with self._lock:
            # Reset correlation scores
            self.correlation_scores = {}
            
            # For each error type
            for error_type, sequences in self.error_sequences.items():
                if not sequences:
                    continue
                    
                # Count occurrences of each event type before this error
                event_counts = defaultdict(int)
                total_sequences = len(sequences)
                
                for sequence in sequences:
                    seen_events = set()
                    for event_type, source in sequence:
                        event_key = f"{event_type}:{source}"
                        if event_key not in seen_events:
                            event_counts[event_key] += 1
                            seen_events.add(event_key)
                
                # Calculate correlation scores (0-1)
                for event_key, count in event_counts.items():
                    correlation = count / total_sequences
                    if correlation > 0.5:  # Only store significant correlations
                        self.correlation_scores[(event_key, error_type)] = correlation

    def get_error_correlations(self, error_type: str = None) -> Dict[str, float]:
        """
        Get potential causes of errors based on correlation analysis.
        
        Args:
            error_type: Specific error type to analyze, or None for all
            
        Returns:
            Dictionary of event -> correlation_score
        """
        with self._lock:
            if error_type:
                return {
                    event_key: score
                    for (event_key, err_type), score in self.correlation_scores.items()
                    if err_type == error_type
                }
            else:
                # Group by error type
                result = defaultdict(dict)
                for (event_key, err_type), score in self.correlation_scores.items():
                    result[err_type][event_key] = score
                return dict(result)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            severity_counts = defaultdict(int)
            for event in self.events:
                severity_counts[event["severity"]] += 1
                
            return {
                "total_events": len(self.events),
                "event_types": dict(self.event_types),
                "event_sources": dict(self.event_sources),
                "severity_distribution": {
                    EventSeverity(level).name: count 
                    for level, count in severity_counts.items()
                },
                "metrics_tracked": len(self.metrics),
                "correlation_patterns": len(self.correlation_scores)
            }

    def clear_old_events(self, days: int = RETENTION_DAYS):
        """
        Clear events older than specified days.
        
        Args:
            days: Retention period in days
            
        Returns:
            Number of events cleared
        """
        with self._lock:
            threshold = datetime.utcnow() - timedelta(days=days)
            original_count = len(self.events)
            
            # Filter events
            self.events = deque(
                [e for e in self.events if e["timestamp"] >= threshold],
                maxlen=self.events.maxlen
            )
            
            # Update counts
            self.event_types.clear()
            self.event_sources.clear()
            for event in self.events:
                self.event_types[event["type"]] += 1
                self.event_sources[event["source"]] += 1
                
            cleared = original_count - len(self.events)
            if cleared > 0:
                self.logger.info(f"Cleared {cleared} old events")
                
            return cleared

    def export_state(self) -> str:
        """
        Export state for persistence.
        
        Returns:
            JSON string of state
        """
        with self._lock:
            # Convert datetime objects to strings
            events_serializable = []
            for event in self.events:
                event_copy = event.copy()
                event_copy["timestamp"] = event_copy["timestamp"].isoformat()
                events_serializable.append(event_copy)
                
            metrics_serializable = {}
            for name, values in self.metrics.items():
                metrics_serializable[name] = [(ts.isoformat(), val) for ts, val in values]
                
            return json.dumps({
                "events": events_serializable,
                "event_types": dict(self.event_types),
                "event_sources": dict(self.event_sources),
                "metrics": metrics_serializable,
                "metric_stats": self.metric_stats
            })

    def load_state(self, state: str):
        """
        Load state from persistence.
        
        Args:
            state: JSON string of state
        """
        with self._lock:
            data = json.loads(state)
            
            # Clear existing state
            self.events.clear()
            self.event_types.clear()
            self.event_sources.clear()
            self.metrics.clear()
            self.metric_stats.clear()
            
            # Convert string timestamps back to datetime
            for event in data["events"]:
                event["timestamp"] = datetime.fromisoformat(event["timestamp"])
                self.events.append(event)
                
            self.event_types.update(data["event_types"])
            self.event_sources.update(data["event_sources"])
            
            for name, values in data["metrics"].items():
                self.metrics[name] = deque(
                    [(datetime.fromisoformat(ts), val) for ts, val in values],
                    maxlen=100
                )
                
            self.metric_stats = data["metric_stats"]