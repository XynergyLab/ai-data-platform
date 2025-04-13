from datetime import datetime
from typing import Dict, Any
from .base import AlertHandler
from ..rules import AlertRule

class MetricAlertHandler(AlertHandler):
    """Handler for metric-based alerts"""
    
    def handle_alert(self, alert_rule: AlertRule, metric_value: Any) -> Dict:
        """Handle a metric alert condition"""
        try:
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
                    metric_value
                )
            
            # Create new alert
            alert = self._create_alert_record(alert_rule, metric_value)
            
            # Add metric-specific information
            alert.update({
                'metric_name': alert_rule.config.get('metric'),
                'comparison': alert_rule.config.get('comparison', 'greater'),
                'duration': alert_rule.config.get('duration', 0),
                'notification_count': 0
            })
            
            self.active_alerts[alert['id']] = alert
            self._update_alert_history(alert)
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error handling metric alert: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _update_existing_alert(self, existing_alert: Dict,
                             alert_rule: AlertRule,
                             metric_value: Any) -> Dict:
        """Update an existing alert"""
        existing_alert['current_value'] = metric_value
        existing_alert['last_updated'] = datetime.now()
        
        if alert_rule.should_notify():
            existing_alert['notification_count'] += 1
            
        self._update_alert_history(existing_alert)
        return existing_alert
    
    def process_resolution(self, alert_id: str) -> Dict:
        """Process metric alert resolution"""
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
            
            self._update_alert_history(alert)
            return alert
            
        except Exception as e:
            self.logger.error(f"Error processing alert resolution: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
