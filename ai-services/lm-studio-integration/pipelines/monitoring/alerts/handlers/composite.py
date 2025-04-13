from datetime import datetime
from typing import Dict, Any, List
from .base import AlertHandler
from ..rules import AlertRule, CompositeRule

class CompositeAlertHandler(AlertHandler):
    """Handler for composite alerts"""
    
    def handle_alert(self, alert_rule: AlertRule, values: Dict[str, Any]) -> Dict:
        """Handle a composite alert condition"""
        try:
            if not isinstance(alert_rule, CompositeRule):
                raise ValueError("Expected CompositeRule instance")
            
            # Check if alert already exists
            existing_alert = next(
                (alert for alert in self.active_alerts.values()
                 if alert['name'] == alert_rule.name),
                None
            )
            
            if existing_alert:
                return self._update_existing_alert(
                    existing_alert,
                    alert_rule,
                    values
                )
            
            # Create new alert
            alert = self._create_alert_record(alert_rule, values)
            
            # Add composite-specific information
            alert.update({
                'conditions': alert_rule.conditions,
                'operator': alert_rule.operator,
                'triggered_conditions': self._get_triggered_conditions(
                    alert_rule,
                    values
                ),
                'notification_count': 0
            })
            
            self.active_alerts[alert['id']] = alert
            self._update_alert_history(alert)
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error handling composite alert: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _update_existing_alert(self, existing_alert: Dict,
                             alert_rule: CompositeRule,
                             values: Dict[str, Any]) -> Dict:
        """Update an existing composite alert"""
        existing_alert['last_updated'] = datetime.now()
        existing_alert['triggered_conditions'] = self._get_triggered_conditions(
            alert_rule,
            values
        )
        
        if alert_rule.should_notify():
            existing_alert['notification_count'] += 1
        
        self._update_alert_history(existing_alert)
        return existing_alert
    
    def _get_triggered_conditions(self, alert_rule: CompositeRule,
                                values: Dict[str, Any]) -> List[Dict]:
        """Get list of triggered conditions"""
        triggered = []
        for rule, condition in zip(alert_rule.subrules, alert_rule.conditions):
            metric_name = condition.get('metric')
            if metric_name in values and rule.check_condition(values[metric_name]):
                triggered.append({
                    'metric': metric_name,
                    'type': condition.get('type'),
                    'threshold': condition.get('threshold'),
                    'current_value': values[metric_name]
                })
        return triggered
    
    def process_resolution(self, alert_id: str) -> Dict:
        """Process composite alert resolution"""
        try:
            if alert_id not in self.active_alerts:
                return {
                    'status': 'error',
                    'error': 'Alert not found'
                }
            
            alert = self.active_alerts.pop(alert_id)
            
            # Update alert status
            alert['status'] = 'resolved'
            alert['resolution_time'] = datetime.now()
            alert['duration'] = (
                alert['resolution_time'] - alert['timestamp']
            ).total_seconds()
            alert['final_triggered_count'] = len(alert['triggered_conditions'])
            
            self._update_alert_history(alert)
            return alert
            
        except Exception as e:
            self.logger.error(f"Error processing alert resolution: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
