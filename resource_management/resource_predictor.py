# resource_predictor.py - Hybrid ML/resource forecaster
# written by DeepSeek Chat (honor call: The Oracle)

import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from collections import deque
import logging

class ResourcePredictor:
    def __init__(self, 
                 window_size: int = 10,
                 arima_order: tuple = (2,1,1)):
        self.window_size = window_size
        self.arima_order = arima_order
        self.cpu_history = deque(maxlen=window_size)
        self.mem_history = deque(maxlen=window_size)
        self.lstm_models = {
            'cpu': self._build_lstm(),
            'mem': self._build_lstm()
        }
        self.logger = logging.getLogger("GGFAI.predictor")

    def _build_lstm(self) -> Sequential:
        """Build LSTM model for time-series prediction."""
        model = Sequential([
            LSTM(8, input_shape=(self.window_size-1, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _arima_predict(self, data: list, steps: int = 1) -> float:
        """ARIMA prediction for short-term trends."""
        try:
            model = ARIMA(data, order=self.arima_order)
            model_fit = model.fit()
            return model_fit.forecast(steps=steps)[0]
        except:
            return np.mean(data[-3:])  # Fallback to moving average

    def update_and_predict(self, 
                         current_cpu: float,
                         current_mem: float) -> Dict[str, Dict]:
        """Update models and return predictions with confidence."""
        self.cpu_history.append(current_cpu)
        self.mem_history.append(current_mem)
        
        predictions = {}
        for resource, history in [('cpu', self.cpu_history), ('mem', self.mem_history)]:
            if len(history) < self.window_size:
                predictions[resource] = {
                    'value': np.mean(history) if history else 0.5,
                    'confidence': 0.5
                }
                continue

            # Hybrid prediction: ARIMA for short-term, LSTM for longer trends
            arima_pred = self._arima_predict(list(history))
            
            lstm_input = np.array(history[:-1]).reshape(1, self.window_size-1, 1)
            lstm_pred = self.lstm_models[resource].predict(lstm_input)[0][0]
            
            # Weighted ensemble
            hybrid_pred = 0.7 * arima_pred + 0.3 * lstm_pred
            confidence = 1 - (np.std(history) / np.mean(history))  # Simple confidence metric
            
            predictions[resource] = {
                'value': max(0, min(1, hybrid_pred)),
                'confidence': max(0.1, min(1, confidence))
            }

        return predictions