from typing import Dict, Any
from datetime import datetime, timedelta

class AlertRule:
    """Base class for alert rules"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.threshold = config['threshold']
        self.severity = config['severity']
        self.last_check = None
        self.last_trigger = None
    
    def check_condition(self, value: Any) -> bool:
        """Check if alert condition is met"""
        raise NotImplementedError
    
    def should_notify(self) -> bool:
        """Check if notification should be sent"""
        if not self.last_trigger:
            return True
            
        cooldown = timedelta(
            seconds=self.config.get('notification_cooldown', 300)
        )
        return datetime.now() - self.last_trigger > cooldown

class ThresholdRule(AlertRule):
    """Alert rule based on threshold"""
    
    def check_condition(self, value: Any) -> bool:
        return value > self.threshold

class RateRule(AlertRule):
    """Alert rule based on rate of change"""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.last_value = None
        self.rate_threshold = config['rate_threshold']
    
    def check_condition(self, value: Any) -> bool:
        if self.last_value is None:
            self.last_value = value
            return False
            
        rate = (value - self.last_value) / self.last_value
        self.last_value = value
        return abs(rate) > self.rate_threshold
