"""
ML-based Resource Predictor for GGFAI Framework.

This module provides advanced resource usage prediction using machine learning
models to forecast CPU, memory, and other resource demands based on historical
usage patterns and current system state.
"""

import logging
import time
import threading
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from enum import Enum
from dataclasses import dataclass, field
import pickle
import numpy as np
from datetime import datetime, timedelta
import collections

# Configure logging
logger = logging.getLogger("GGFAI.resource.predictor")

# Check for optional dependencies
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Using fallback prediction methods.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Some features will be limited.")


class ResourceType(Enum):
    """Types of resources that can be predicted."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    BATTERY = "battery"


class PredictionMethod(Enum):
    """Methods for resource prediction."""
    MOVING_AVERAGE = "moving_average"  # Simple moving average
    EXPONENTIAL = "exponential"  # Exponential smoothing
    RANDOM_FOREST = "random_forest"  # ML-based random forest
    ARIMA = "arima"  # ARIMA time series model
    LSTM = "lstm"  # LSTM neural network


@dataclass
class ResourcePredictionConfig:
    """Configuration for resource prediction."""
    # General settings
    prediction_method: PredictionMethod = PredictionMethod.MOVING_AVERAGE
    history_window: int = 60  # Number of data points to keep in history
    prediction_horizon: int = 10  # Number of steps to predict ahead
    update_interval: int = 5  # Seconds between updates
    
    # Model settings
    model_path: Optional[str] = None
    retrain_interval: int = 3600  # Seconds between model retraining
    min_samples_for_training: int = 100
    
    # Resource types to monitor
    resource_types: List[ResourceType] = field(default_factory=lambda: [
        ResourceType.CPU, 
        ResourceType.MEMORY
    ])
    
    # Feature settings
    use_time_features: bool = True  # Use time of day, day of week as features
    use_load_features: bool = True  # Use system load as features
    
    # Thresholds for alerts
    cpu_threshold: float = 0.8  # 80% CPU usage
    memory_threshold: float = 0.8  # 80% memory usage
    disk_threshold: float = 0.9  # 90% disk usage
    
    # Cache settings
    cache_dir: str = "cache/resource_prediction"
    
    def __post_init__(self):
        """Initialize default values and create directories."""
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set default model path if not provided
        if self.model_path is None:
            self.model_path = os.path.join(self.cache_dir, "resource_model.pkl")


@dataclass
class PredictionResult:
    """Result of a resource prediction."""
    resource_type: ResourceType
    current_value: float
    predicted_values: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    prediction_timestamps: Optional[List[float]] = None
    exceeds_threshold: bool = False
    threshold: float = 0.0
    method_used: str = "unknown"


class MLResourcePredictor:
    """
    ML-based resource predictor for GGFAI Framework that forecasts
    resource usage based on historical patterns and current state.
    """
    
    def __init__(self, config: Optional[ResourcePredictionConfig] = None):
        """
        Initialize resource predictor with configuration.
        
        Args:
            config: Resource prediction configuration
        """
        self.config = config or ResourcePredictionConfig()
        self.logger = logging.getLogger("GGFAI.resource.predictor")
        
        # Initialize state
        self._lock = threading.RLock()
        self._running = False
        self._last_update_time = 0
        self._last_train_time = 0
        
        # Initialize data structures
        self._history = {
            resource_type: collections.deque(maxlen=self.config.history_window)
            for resource_type in self.config.resource_types
        }
        
        # Initialize models
        self._models = {}
        self._scalers = {}
        
        # Load existing models if available
        self._load_models()
        
        self.logger.info("ML Resource Predictor initialized")
    
    def _load_models(self):
        """Load existing prediction models if available."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available. Cannot load ML models.")
            return
        
        try:
            model_path = Path(self.config.model_path)
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    
                    self._models = saved_data.get('models', {})
                    self._scalers = saved_data.get('scalers', {})
                    
                    self.logger.info(f"Loaded models for {len(self._models)} resource types")
            else:
                self.logger.info("No existing models found. Will train new models.")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            self._models = {}
            self._scalers = {}
    
    def _save_models(self):
        """Save prediction models to disk."""
        if not SKLEARN_AVAILABLE or not self._models:
            return
        
        try:
            model_path = Path(self.config.model_path)
            
            # Create parent directory if it doesn't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save models and scalers
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'models': self._models,
                    'scalers': self._scalers
                }, f)
                
            self.logger.info(f"Saved models for {len(self._models)} resource types")
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def start(self) -> bool:
        """
        Start the resource prediction service.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if self._running:
                self.logger.warning("Resource predictor already running")
                return True
            
            try:
                self._running = True
                
                # Start update thread
                self._update_thread = threading.Thread(
                    target=self._update_loop,
                    daemon=True
                )
                self._update_thread.start()
                
                self.logger.info("Resource predictor started")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to start resource predictor: {e}")
                self._running = False
                return False
    
    def stop(self) -> bool:
        """
        Stop the resource prediction service.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if not self._running:
                self.logger.warning("Resource predictor not running")
                return True
            
            try:
                self._running = False
                
                # Wait for update thread to end
                if hasattr(self, '_update_thread') and self._update_thread.is_alive():
                    self._update_thread.join(timeout=2.0)
                
                # Save models before stopping
                self._save_models()
                
                self.logger.info("Resource predictor stopped")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to stop resource predictor: {e}")
                return False
    
    def _update_loop(self):
        """Update loop running in a separate thread."""
        while self._running:
            try:
                current_time = time.time()
                
                # Check if it's time to update
                if current_time - self._last_update_time >= self.config.update_interval:
                    self._update_history()
                    self._last_update_time = current_time
                
                # Check if it's time to retrain models
                if (SKLEARN_AVAILABLE and 
                    self.config.prediction_method == PredictionMethod.RANDOM_FOREST and
                    current_time - self._last_train_time >= self.config.retrain_interval):
                    
                    self._train_models()
                    self._last_train_time = current_time
                
                # Sleep for a short time
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(5.0)  # Sleep longer after error
    
    def _update_history(self):
        """Update resource usage history."""
        try:
            # Get current resource usage
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            disk_usage = self._get_disk_usage()
            
            # Get current timestamp
            timestamp = time.time()
            
            # Update history for each resource type
            if ResourceType.CPU in self.config.resource_types:
                self._history[ResourceType.CPU].append({
                    'timestamp': timestamp,
                    'value': cpu_usage,
                    'hour': datetime.now().hour,
                    'day': datetime.now().weekday()
                })
            
            if ResourceType.MEMORY in self.config.resource_types:
                self._history[ResourceType.MEMORY].append({
                    'timestamp': timestamp,
                    'value': memory_usage,
                    'hour': datetime.now().hour,
                    'day': datetime.now().weekday()
                })
            
            if ResourceType.DISK in self.config.resource_types:
                self._history[ResourceType.DISK].append({
                    'timestamp': timestamp,
                    'value': disk_usage,
                    'hour': datetime.now().hour,
                    'day': datetime.now().weekday()
                })
            
            # Add other resource types as needed
            
            self.logger.debug(f"Updated resource history (CPU: {cpu_usage:.2f}, Memory: {memory_usage:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to update resource history: {e}")
    
    def _get_cpu_usage(self) -> float:
        """
        Get current CPU usage as a fraction (0.0 to 1.0).
        
        Returns:
            CPU usage as a fraction
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1) / 100.0
        except ImportError:
            # Fallback if psutil is not available
            try:
                if os.name == 'posix':  # Linux/Mac
                    # Use 'top' command
                    import subprocess
                    output = subprocess.check_output(['top', '-bn1']).decode('utf-8')
                    cpu_line = [line for line in output.split('\n') if 'Cpu(s)' in line][0]
                    cpu_usage = float(cpu_line.split()[1].replace('%', '')) / 100.0
                    return cpu_usage
                elif os.name == 'nt':  # Windows
                    # Use 'wmic' command
                    import subprocess
                    output = subprocess.check_output('wmic cpu get loadpercentage').decode('utf-8')
                    cpu_usage = float(output.strip().split('\n')[1]) / 100.0
                    return cpu_usage
                else:
                    return 0.5  # Default value if platform not supported
            except Exception:
                return 0.5  # Default value if command fails
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage as a fraction (0.0 to 1.0).
        
        Returns:
            Memory usage as a fraction
        """
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Fallback if psutil is not available
            try:
                if os.name == 'posix':  # Linux/Mac
                    # Use 'free' command
                    import subprocess
                    output = subprocess.check_output(['free', '-m']).decode('utf-8')
                    memory_line = output.split('\n')[1]
                    values = memory_line.split()
                    total = float(values[1])
                    used = float(values[2])
                    return used / total
                elif os.name == 'nt':  # Windows
                    # Use 'wmic' command
                    import subprocess
                    total_output = subprocess.check_output('wmic ComputerSystem get TotalPhysicalMemory').decode('utf-8')
                    total = float(total_output.strip().split('\n')[1])
                    free_output = subprocess.check_output('wmic OS get FreePhysicalMemory').decode('utf-8')
                    free = float(free_output.strip().split('\n')[1])
                    return 1.0 - (free / total)
                else:
                    return 0.5  # Default value if platform not supported
            except Exception:
                return 0.5  # Default value if command fails
    
    def _get_disk_usage(self) -> float:
        """
        Get current disk usage as a fraction (0.0 to 1.0).
        
        Returns:
            Disk usage as a fraction
        """
        try:
            import psutil
            return psutil.disk_usage('/').percent / 100.0
        except ImportError:
            # Fallback if psutil is not available
            try:
                if os.name == 'posix':  # Linux/Mac
                    # Use 'df' command
                    import subprocess
                    output = subprocess.check_output(['df', '-h', '/']).decode('utf-8')
                    disk_line = output.split('\n')[1]
                    usage = disk_line.split()[4].replace('%', '')
                    return float(usage) / 100.0
                elif os.name == 'nt':  # Windows
                    # Use 'wmic' command
                    import subprocess
                    output = subprocess.check_output('wmic logicaldisk get size,freespace,caption').decode('utf-8')
                    lines = output.strip().split('\n')[1:]
                    total_usage = 0.0
                    count = 0
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    free = float(parts[1])
                                    total = float(parts[2])
                                    usage = 1.0 - (free / total)
                                    total_usage += usage
                                    count += 1
                                except (ValueError, IndexError):
                                    pass
                    return total_usage / count if count > 0 else 0.5
                else:
                    return 0.5  # Default value if platform not supported
            except Exception:
                return 0.5  # Default value if command fails
    
    def _train_models(self):
        """Train prediction models using historical data."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available. Cannot train ML models.")
            return
        
        try:
            for resource_type in self.config.resource_types:
                history = list(self._history[resource_type])
                
                # Check if we have enough data
                if len(history) < self.config.min_samples_for_training:
                    self.logger.info(f"Not enough data to train model for {resource_type.value} "
                                    f"({len(history)}/{self.config.min_samples_for_training})")
                    continue
                
                # Prepare features and target
                X, y = self._prepare_training_data(history)
                
                if len(X) == 0 or len(y) == 0:
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_scaled, y)
                
                # Save model and scaler
                self._models[resource_type] = model
                self._scalers[resource_type] = scaler
                
                self.logger.info(f"Trained model for {resource_type.value} with {len(X)} samples")
            
            # Save models to disk
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Failed to train models: {e}")
    
    def _prepare_training_data(self, history: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from history.
        
        Args:
            history: List of historical data points
            
        Returns:
            Tuple of (features, targets)
        """
        if not history:
            return np.array([]), np.array([])
        
        # Convert to pandas DataFrame if available
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(history)
            
            # Create lagged features
            for lag in range(1, min(10, len(df))):
                df[f'value_lag_{lag}'] = df['value'].shift(lag)
            
            # Create time features if enabled
            if self.config.use_time_features:
                # One-hot encode hour and day
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
                df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            if df.empty:
                return np.array([]), np.array([])
            
            # Select features and target
            feature_cols = [col for col in df.columns if col.startswith('value_lag_') or 
                           col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']]
            
            X = df[feature_cols].values
            y = df['value'].values
            
            return X, y
        else:
            # Fallback if pandas is not available
            X = []
            y = []
            
            for i in range(10, len(history)):
                # Create feature vector with lagged values
                features = [history[i-lag]['value'] for lag in range(1, 11)]
                
                # Add time features if enabled
                if self.config.use_time_features:
                    hour = history[i]['hour']
                    day = history[i]['day']
                    features.extend([
                        np.sin(2 * np.pi * hour / 24),
                        np.cos(2 * np.pi * hour / 24),
                        np.sin(2 * np.pi * day / 7),
                        np.cos(2 * np.pi * day / 7)
                    ])
                
                X.append(features)
                y.append(history[i]['value'])
            
            return np.array(X), np.array(y)
    
    def predict_resource_usage(self, resource_type: ResourceType, 
                              steps: int = None) -> PredictionResult:
        """
        Predict future resource usage.
        
        Args:
            resource_type: Type of resource to predict
            steps: Number of steps to predict ahead (default: config.prediction_horizon)
            
        Returns:
            PredictionResult with predicted values and confidence intervals
        """
        if steps is None:
            steps = self.config.prediction_horizon
        
        with self._lock:
            # Check if we have history for this resource type
            if resource_type not in self._history or not self._history[resource_type]:
                self.logger.warning(f"No history available for {resource_type.value}")
                return PredictionResult(
                    resource_type=resource_type,
                    current_value=0.0,
                    predicted_values=[0.0] * steps,
                    method_used="no_data"
                )
            
            # Get current value
            current_value = self._history[resource_type][-1]['value']
            
            # Get threshold for this resource type
            threshold = self._get_threshold(resource_type)
            
            # Choose prediction method
            if (self.config.prediction_method == PredictionMethod.RANDOM_FOREST and
                SKLEARN_AVAILABLE and resource_type in self._models):
                # Use ML model
                result = self._predict_with_ml(resource_type, steps)
                result.threshold = threshold
                result.exceeds_threshold = any(val > threshold for val in result.predicted_values)
                return result
            elif self.config.prediction_method == PredictionMethod.EXPONENTIAL:
                # Use exponential smoothing
                result = self._predict_with_exponential(resource_type, steps)
                result.threshold = threshold
                result.exceeds_threshold = any(val > threshold for val in result.predicted_values)
                return result
            else:
                # Use moving average as fallback
                result = self._predict_with_moving_average(resource_type, steps)
                result.threshold = threshold
                result.exceeds_threshold = any(val > threshold for val in result.predicted_values)
                return result
    
    def _get_threshold(self, resource_type: ResourceType) -> float:
        """
        Get threshold for a resource type.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Threshold value
        """
        if resource_type == ResourceType.CPU:
            return self.config.cpu_threshold
        elif resource_type == ResourceType.MEMORY:
            return self.config.memory_threshold
        elif resource_type == ResourceType.DISK:
            return self.config.disk_threshold
        else:
            return 0.8  # Default threshold
    
    def _predict_with_ml(self, resource_type: ResourceType, steps: int) -> PredictionResult:
        """
        Predict resource usage using ML model.
        
        Args:
            resource_type: Type of resource to predict
            steps: Number of steps to predict ahead
            
        Returns:
            PredictionResult with predicted values and confidence intervals
        """
        try:
            history = list(self._history[resource_type])
            current_value = history[-1]['value']
            
            # Get model and scaler
            model = self._models[resource_type]
            scaler = self._scalers[resource_type]
            
            # Prepare initial features
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(history)
                
                # Create lagged features
                for lag in range(1, min(10, len(df))):
                    df[f'value_lag_{lag}'] = df['value'].shift(lag)
                
                # Create time features if enabled
                if self.config.use_time_features:
                    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
                    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
                
                # Drop rows with NaN values
                df = df.dropna()
                
                if df.empty:
                    return self._predict_with_moving_average(resource_type, steps)
                
                # Select features
                feature_cols = [col for col in df.columns if col.startswith('value_lag_') or 
                               col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']]
                
                # Get latest feature values
                latest_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
                
            else:
                # Fallback if pandas is not available
                if len(history) < 10:
                    return self._predict_with_moving_average(resource_type, steps)
                
                # Create feature vector with lagged values
                latest_features = [history[-lag]['value'] for lag in range(1, 11)]
                
                # Add time features if enabled
                if self.config.use_time_features:
                    hour = history[-1]['hour']
                    day = history[-1]['day']
                    latest_features.extend([
                        np.sin(2 * np.pi * hour / 24),
                        np.cos(2 * np.pi * hour / 24),
                        np.sin(2 * np.pi * day / 7),
                        np.cos(2 * np.pi * day / 7)
                    ])
                
                latest_features = np.array(latest_features).reshape(1, -1)
            
            # Scale features
            latest_features_scaled = scaler.transform(latest_features)
            
            # Make predictions
            predicted_values = []
            confidence_intervals = []
            prediction_timestamps = []
            
            # Current feature vector
            current_features = latest_features_scaled.copy()
            
            # Current timestamp and datetime
            current_timestamp = time.time()
            current_dt = datetime.now()
            
            for i in range(steps):
                # Predict next value
                predictions = []
                for estimator in model.estimators_:
                    predictions.append(estimator.predict(current_features)[0])
                
                # Calculate mean and confidence interval
                mean_prediction = np.mean(predictions)
                std_prediction = np.std(predictions)
                lower_bound = max(0, mean_prediction - 1.96 * std_prediction)
                upper_bound = min(1, mean_prediction + 1.96 * std_prediction)
                
                # Ensure prediction is between 0 and 1
                mean_prediction = max(0, min(1, mean_prediction))
                
                # Add to results
                predicted_values.append(mean_prediction)
                confidence_intervals.append((lower_bound, upper_bound))
                
                # Calculate timestamp for this prediction
                future_timestamp = current_timestamp + (i + 1) * self.config.update_interval
                prediction_timestamps.append(future_timestamp)
                
                # Update features for next prediction
                if PANDAS_AVAILABLE:
                    # Shift lagged values
                    for lag in range(9, 0, -1):
                        lag_idx = feature_cols.index(f'value_lag_{lag}')
                        next_lag_idx = feature_cols.index(f'value_lag_{lag+1}') if lag < 9 else -1
                        
                        if next_lag_idx >= 0:
                            current_features[0, next_lag_idx] = current_features[0, lag_idx]
                    
                    # Set newest value
                    current_features[0, feature_cols.index('value_lag_1')] = mean_prediction
                    
                    # Update time features if enabled
                    if self.config.use_time_features:
                        future_dt = current_dt + timedelta(seconds=(i + 1) * self.config.update_interval)
                        hour = future_dt.hour
                        day = future_dt.weekday()
                        
                        hour_sin_idx = feature_cols.index('hour_sin')
                        hour_cos_idx = feature_cols.index('hour_cos')
                        day_sin_idx = feature_cols.index('day_sin')
                        day_cos_idx = feature_cols.index('day_cos')
                        
                        current_features[0, hour_sin_idx] = np.sin(2 * np.pi * hour / 24)
                        current_features[0, hour_cos_idx] = np.cos(2 * np.pi * hour / 24)
                        current_features[0, day_sin_idx] = np.sin(2 * np.pi * day / 7)
                        current_features[0, day_cos_idx] = np.cos(2 * np.pi * day / 7)
                else:
                    # Shift lagged values
                    for j in range(9, 0, -1):
                        current_features[0, j] = current_features[0, j-1]
                    
                    # Set newest value
                    current_features[0, 0] = mean_prediction
                    
                    # Update time features if enabled
                    if self.config.use_time_features:
                        future_dt = current_dt + timedelta(seconds=(i + 1) * self.config.update_interval)
                        hour = future_dt.hour
                        day = future_dt.weekday()
                        
                        current_features[0, 10] = np.sin(2 * np.pi * hour / 24)
                        current_features[0, 11] = np.cos(2 * np.pi * hour / 24)
                        current_features[0, 12] = np.sin(2 * np.pi * day / 7)
                        current_features[0, 13] = np.cos(2 * np.pi * day / 7)
            
            return PredictionResult(
                resource_type=resource_type,
                current_value=current_value,
                predicted_values=predicted_values,
                confidence_intervals=confidence_intervals,
                prediction_timestamps=prediction_timestamps,
                method_used="random_forest"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to predict with ML: {e}")
            return self._predict_with_moving_average(resource_type, steps)
    
    def _predict_with_exponential(self, resource_type: ResourceType, steps: int) -> PredictionResult:
        """
        Predict resource usage using exponential smoothing.
        
        Args:
            resource_type: Type of resource to predict
            steps: Number of steps to predict ahead
            
        Returns:
            PredictionResult with predicted values
        """
        try:
            history = list(self._history[resource_type])
            values = [entry['value'] for entry in history]
            current_value = values[-1]
            
            # Calculate exponential smoothing parameters
            alpha = 0.3  # Smoothing factor
            
            # Calculate initial level
            level = values[0]
            trend = 0
            
            # Calculate level and trend
            for value in values[1:]:
                prev_level = level
                level = alpha * value + (1 - alpha) * (level + trend)
                trend = 0.1 * (level - prev_level) + (1 - 0.1) * trend
            
            # Make predictions
            predicted_values = []
            prediction_timestamps = []
            current_timestamp = time.time()
            
            for i in range(steps):
                # Predict next value
                prediction = level + (i + 1) * trend
                
                # Ensure prediction is between 0 and 1
                prediction = max(0, min(1, prediction))
                
                predicted_values.append(prediction)
                prediction_timestamps.append(current_timestamp + (i + 1) * self.config.update_interval)
            
            return PredictionResult(
                resource_type=resource_type,
                current_value=current_value,
                predicted_values=predicted_values,
                prediction_timestamps=prediction_timestamps,
                method_used="exponential"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to predict with exponential smoothing: {e}")
            return self._predict_with_moving_average(resource_type, steps)
    
    def _predict_with_moving_average(self, resource_type: ResourceType, steps: int) -> PredictionResult:
        """
        Predict resource usage using moving average.
        
        Args:
            resource_type: Type of resource to predict
            steps: Number of steps to predict ahead
            
        Returns:
            PredictionResult with predicted values
        """
        try:
            history = list(self._history[resource_type])
            values = [entry['value'] for entry in history]
            current_value = values[-1]
            
            # Calculate moving average
            window_size = min(10, len(values))
            if window_size == 0:
                avg_value = 0.5  # Default if no history
            else:
                avg_value = sum(values[-window_size:]) / window_size
            
            # Calculate trend
            if len(values) >= 2 * window_size:
                prev_avg = sum(values[-2*window_size:-window_size]) / window_size
                trend = (avg_value - prev_avg) / window_size
            else:
                trend = 0
            
            # Make predictions
            predicted_values = []
            prediction_timestamps = []
            current_timestamp = time.time()
            
            for i in range(steps):
                # Predict next value with trend
                prediction = avg_value + trend * (i + 1)
                
                # Ensure prediction is between 0 and 1
                prediction = max(0, min(1, prediction))
                
                predicted_values.append(prediction)
                prediction_timestamps.append(current_timestamp + (i + 1) * self.config.update_interval)
            
            return PredictionResult(
                resource_type=resource_type,
                current_value=current_value,
                predicted_values=predicted_values,
                prediction_timestamps=prediction_timestamps,
                method_used="moving_average"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to predict with moving average: {e}")
            
            # Return default prediction
            current_value = 0.5  # Default if everything fails
            predicted_values = [current_value] * steps
            prediction_timestamps = [time.time() + (i + 1) * self.config.update_interval for i in range(steps)]
            
            return PredictionResult(
                resource_type=resource_type,
                current_value=current_value,
                predicted_values=predicted_values,
                prediction_timestamps=prediction_timestamps,
                method_used="default"
            )
    
    def predict_all_resources(self, steps: int = None) -> Dict[ResourceType, PredictionResult]:
        """
        Predict usage for all configured resource types.
        
        Args:
            steps: Number of steps to predict ahead (default: config.prediction_horizon)
            
        Returns:
            Dictionary of PredictionResults by resource type
        """
        results = {}
        
        for resource_type in self.config.resource_types:
            results[resource_type] = self.predict_resource_usage(resource_type, steps)
        
        return results
    
    def check_resource_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for resource usage alerts based on predictions.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Predict all resources
        predictions = self.predict_all_resources()
        
        for resource_type, prediction in predictions.items():
            # Check if any predicted value exceeds threshold
            if prediction.exceeds_threshold:
                # Find when it will exceed threshold
                for i, value in enumerate(prediction.predicted_values):
                    if value > prediction.threshold:
                        # Calculate time until threshold exceeded
                        time_until = (prediction.prediction_timestamps[i] - time.time()) if prediction.prediction_timestamps else (i + 1) * self.config.update_interval
                        
                        alerts.append({
                            'resource_type': resource_type.value,
                            'current_value': prediction.current_value,
                            'predicted_value': value,
                            'threshold': prediction.threshold,
                            'time_until_exceeded': time_until,
                            'steps_until_exceeded': i + 1,
                            'timestamp': time.time()
                        })
                        
                        # Only report first occurrence
                        break
        
        return alerts
    
    def get_resource_history(self, resource_type: ResourceType, 
                            max_points: int = None) -> List[Dict[str, Any]]:
        """
        Get historical data for a resource type.
        
        Args:
            resource_type: Type of resource
            max_points: Maximum number of data points to return
            
        Returns:
            List of historical data points
        """
        with self._lock:
            if resource_type not in self._history:
                return []
            
            history = list(self._history[resource_type])
            
            if max_points is not None:
                history = history[-max_points:]
            
            return history
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        self.stop()
        
        # Save models before destruction
        if hasattr(self, '_models') and self._models:
            self._save_models()