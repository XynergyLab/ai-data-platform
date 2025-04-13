from typing import Dict, List
from datetime import datetime
from ..core.monitor_base import BaseMonitor

class AlertManager(BaseMonitor):
    """Manages monitoring alerts"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.initialize_alerts()
    
    def initialize_alerts(self):
        """Initialize alert configurations"""
        self.alerts = self.config['monitoring']['alerts']
        self.active_alerts = {}
    
    def check_alerts(self):
        """Check all alert conditions"""
        for alert_name, alert_config in self.alerts.items():
            self.check_alert_condition(alert_name, alert_config)
    
    def check_alert_condition(self, alert_name: str, alert_config: Dict):
        """Check specific alert condition"""
        current_value = self.get_metric_value(alert_config['metric'])
        
        if current_value > alert_config['threshold']:
            self.trigger_alert(alert_name, alert_config, current_value)
    
    def trigger_alert(self, alert_name: str, alert_config: Dict, current_value: float):
        """Trigger an alert"""
        alert = {
            'name': alert_name,
            'severity': alert_config['severity'],
            'threshold': alert_config['threshold'],
            'current_value': current_value,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.active_alerts[alert_name] = alert
        self.notify_alert(alert)
    
    def notify_alert(self, alert: Dict):
        """Send alert notification"""
        self.logger.warning(f"Alert triggered: {alert}")
