from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import AlertRule
import logging
import re

class PatternRule(AlertRule):
    """Alert rule based on pattern matching"""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.pattern = config.get('pattern')
        self.occurrence_threshold = config.get('occurrence_threshold', 1)
        self.pattern_window = config.get('pattern_window', 300)  # 5 minutes default
        self.occurrences = []
    
    def check_condition(self, value: Any) -> bool:
        self.update_check_time()
        
        try:
            # Convert value to string for pattern matching
            str_value = str(value)
            
            # Check if value matches pattern
            if re.search(self.pattern, str_value):
                self.record_occurrence()
                
                # Count occurrences within window
                count = self.count_recent_occurrences()
                
                if count >= self.occurrence_threshold:
                    self.record_trigger()
                    if self.duration > 0:
                        duration_exceeded = (
                            datetime.now() - self.violation_start
                        ).total_seconds() >= self.duration
                        return duration_exceeded
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking pattern for {self.name}: {str(e)}")
            return False
    
    def record_occurrence(self):
        """Record a pattern occurrence"""
        current_time = datetime.now()
        self.occurrences.append(current_time)
        
        # Clean up old occurrences
        self.clean_old_occurrences()
    
    def clean_old_occurrences(self):
        """Remove occurrences outside the window"""
        current_time = datetime.now()
        window_start = current_time.timestamp() - self.pattern_window
        
        self.occurrences = [
            t for t in self.occurrences
            if t.timestamp() >= window_start
        ]
    
    def count_recent_occurrences(self) -> int:
        """Count occurrences within the window"""
        self.clean_old_occurrences()
        return len(self.occurrences)
