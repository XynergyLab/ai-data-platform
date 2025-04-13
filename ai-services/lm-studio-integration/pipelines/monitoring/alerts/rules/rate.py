from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .base import AlertRule
import logging

class RateRule(AlertRule):
    """Alert rule based on rate of change"""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.rate_threshold = config.get('rate_threshold')
        self.window_size = config.get('window_size', 300)  # 5 minutes default
        self.historical_values = []
        self.last_values = []
    
    def check_condition(self, value: Any) -> bool:
        self.update_check_time()
        
        try:
            current_value = float(value)
            self.update_historical_values(current_value)
            
            if len(self.historical_values) < 2:
                return False
            
            rate = self.calculate_rate()
            if rate is None:
                return False
            
            violated = abs(rate) > float(self.rate_threshold)
            
            if violated:
                self.record_trigger()
                if self.duration > 0:
                    duration_exceeded = (
                        datetime.now() - self.violation_start
                    ).total_seconds() >= self.duration
                    return duration_exceeded
                return True
            else:
                self.clear_violation()
                return False
                
        except (TypeError, ValueError) as e:
            logging.error(f"Error checking rate for {self.name}: {str(e)}")
            return False
    
    def update_historical_values(self, value: float):
        """Update the historical values list"""
        current_time = datetime.now()
        
        # Add new value with timestamp
        self.historical_values.append((current_time, value))
        
        # Remove values outside the window
        window_start = current_time - timedelta(seconds=self.window_size)
        self.historical_values = [
            (t, v) for t, v in self.historical_values
            if t >= window_start
        ]
    
    def calculate_rate(self) -> Optional[float]:
        """Calculate rate of change over the window"""
        if len(self.historical_values) < 2:
            return None
        
        # Get first and last values in window
        first_time, first_value = self.historical_values[0]
        last_time, last_value = self.historical_values[-1]
        
        # Calculate time difference in seconds
        time_diff = (last_time - first_time).total_seconds()
        
        if time_diff == 0:
            return None
        
        # Calculate rate of change
        value_diff = last_value - first_value
        return value_diff / time_diff
