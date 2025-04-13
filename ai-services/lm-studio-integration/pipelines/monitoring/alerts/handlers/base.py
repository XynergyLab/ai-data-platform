import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from ..rules import AlertRule

class AlertHandler:
    """Base class for alert handlers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.active_alerts = {}
        self.alert_history = []
        
    def setup_logging(self):
        """Setup logging for the handler"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(f"logs/alerts/{self.__class__.__name__.lower()}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def handle_alert(self, alert_rule: AlertRule, metric_value: Any) -> Dict:
        """Handle an alert condition"""
        raise NotImplementedError
    
    def process_resolution(self, alert_id: str) -> Dict:
        """Process alert resolution"""
        raise NotImplementedError
    
    def get_alert_status(self, alert_id: str) -> Optional[Dict]:
        """Get current status of an alert"""
        return self.active_alerts.get(alert_id)
    
    def get_alert_history(self, start_time: datetime = None, 
                         end_time: datetime = None) -> List[Dict]:
        """Get alert history within time range"""
        if not start_time and not end_time:
            return self.alert_history
            
        filtered_history = []
        for alert in self.alert_history:
            alert_time = alert['timestamp']
            if start_time and alert_time < start_time:
                continue
            if end_time and alert_time > end_time:
                continue
            filtered_history.append(alert)
            
        return filtered_history
    
    def _create_alert_record(self, alert_rule: AlertRule, 
                           metric_value: Any) -> Dict:
        """Create a record for a new alert"""
        return {
            'id': f"{alert_rule.name}_{datetime.now().timestamp()}",
            'name': alert_rule.name,
            'severity': alert_rule.severity,
            'threshold': alert_rule.threshold,
            'current_value': metric_value,
            'timestamp': datetime.now(),
            'status': 'active'
        }
    
    def _update_alert_history(self, alert: Dict):
        """Update alert history"""
        self.alert_history.append(alert.copy())
        
        # Maintain history size limit
        max_history = self.config.get('max_history_size', 1000)
        if len(self.alert_history) > max_history:
            self.alert_history = self.alert_history[-max_history:]
