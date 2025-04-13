import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class AlertRule:
    """Base class for alert rules"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.threshold = config.get('threshold')
        self.severity = config.get('severity', 'warning')
        self.duration = config.get('duration', 0)
        self.last_check = None
        self.last_trigger = None
        self.violation_start = None
        
    def check_condition(self, value: Any) -> bool:
        """Check if alert condition is met"""
        raise NotImplementedError
    
    def should_notify(self) -> bool:
        """Check if notification should be sent"""
        if not self.last_trigger:
            return True
            
        cooldown = timedelta(
            seconds=self.config.get('notification', {}).get('cooldown', 300)
        )
        return datetime.now() - self.last_trigger > cooldown
    
    def update_check_time(self):
        """Update the last check time"""
        self.last_check = datetime.now()
    
    def record_trigger(self):
        """Record when the alert was triggered"""
        self.last_trigger = datetime.now()
        if not self.violation_start:
            self.violation_start = datetime.now()
    
    def clear_violation(self):
        """Clear the violation state"""
        self.violation_start = None

class ThresholdRule(AlertRule):
    """Alert rule based on threshold value"""
    
    def check_condition(self, value: Any) -> bool:
        self.update_check_time()
        
        try:
            current_value = float(value)
            threshold_value = float(self.threshold)
            
            # Check threshold based on comparison type
            comparison = self.config.get('comparison', 'greater')
            violated = False
            
            if comparison == 'greater':
                violated = current_value > threshold_value
            elif comparison == 'less':
                violated = current_value < threshold_value
            elif comparison == 'equal':
                violated = abs(current_value - threshold_value) < 0.0001
            
            # If violation detected, check duration
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
            logging.error(f"Error checking threshold for {self.name}: {str(e)}")
            return False
