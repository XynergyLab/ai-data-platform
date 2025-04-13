from datetime import datetime
from typing import Dict, Any, List
from .base import AlertRule
import logging

class CompositeRule(AlertRule):
    """Alert rule combining multiple conditions"""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.conditions = config.get('conditions', [])
        self.operator = config.get('operator', 'AND')
        self.subrules = self.create_subrules()
    
    def create_subrules(self) -> List[AlertRule]:
        """Create subrules from conditions"""
        rules = []
        for condition in self.conditions:
            rule_type = condition.get('type', 'threshold')
            if rule_type == 'threshold':
                from .base import ThresholdRule
                rules.append(ThresholdRule(f"{self.name}_sub", condition))
            elif rule_type == 'rate':
                from .rate import RateRule
                rules.append(RateRule(f"{self.name}_sub", condition))
            elif rule_type == 'pattern':
                from .pattern import PatternRule
                rules.append(PatternRule(f"{self.name}_sub", condition))
        return rules
    
    def check_condition(self, values: Dict[str, Any]) -> bool:
        self.update_check_time()
        
        try:
            results = []
            for rule, condition in zip(self.subrules, self.conditions):
                metric_name = condition.get('metric')
                if metric_name not in values:
                    logging.warning(f"Metric {metric_name} not found in values")
                    continue
                
                result = rule.check_condition(values[metric_name])
                results.append(result)
            
            if not results:
                return False
            
            # Evaluate based on operator
            if self.operator == 'AND':
                violated = all(results)
            elif self.operator == 'OR':
                violated = any(results)
            else:
                logging.error(f"Unknown operator {self.operator}")
                return False
            
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
                
        except Exception as e:
            logging.error(f"Error checking composite rule {self.name}: {str(e)}")
            return False
