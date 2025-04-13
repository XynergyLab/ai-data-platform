from datetime import datetime
from typing import Dict, Any, List
from .base import AlertHandler
from ..rules import AlertRule, PatternRule

class PatternAlertHandler(AlertHandler):
    """Handler for pattern-based alerts"""
    
    def handle_alert(self, alert_rule: AlertRule, value: Any) -> Dict:
        """Handle a pattern-based alert condition"""
        try:
            if not isinstance(alert_rule, PatternRule):
                raise ValueError("Expected PatternRule instance")
            
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
                    value
                )
            
            # Create new alert
            alert = self._create_alert_record(alert_rule, value)
            
            # Add pattern-specific information
            alert.update({
                'pattern': alert_rule.pattern,
                'occurrences': 1,
                'occurrence_threshold': alert_rule.occurrence_threshold,
                'matched_values': [str(value)],
                'notification_count': 0
            })
            
            self.active_alerts[alert['id']] = alert
            self._update_alert_history(alert)
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error handling pattern alert: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _update_existing_alert(self, existing_alert: Dict,
                             alert_rule: PatternRule,
                             value: Any) -> Dict:
        """Update an existing pattern alert"""
        existing_alert['occurrences'] += 1
        existing_alert['last_updated'] = datetime.now()
        existing_alert['matched_values'].append(str(value))
        
        # Keep only recent matches
        max_matches = alert_rule.config.get('max_matches', 100)
        if len(existing_alert['matched_values']) > max_matches:
            existing_alert['matched_values'] = existing_alert['matched_values'][-max_matches:]
        
        if alert_rule.should_notify():
            existing_alert['notification_count'] += 1
        
        self._update_alert_history(existing_alert)
        return existing_alert
    
    def process_resolution(self, alert_id: str) -> Dict:
        """Process pattern alert resolution"""
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
            alert['final_occurrence_count'] = alert['occurrences']
            
            self._update_alert_history(alert)
            return alert
            
        except Exception as e:
            self.logger.error(f"Error processing alert resolution: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
