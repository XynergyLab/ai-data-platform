import logging
from typing import Dict, Any, List, Type
from datetime import datetime
from .base import AlertHandler
from .metric import MetricAlertHandler
from .pattern import PatternAlertHandler
from .composite import CompositeAlertHandler
from ..rules import AlertRule, PatternRule, CompositeRule

class AlertAggregator:
    """Aggregates and manages multiple alert handlers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.initialize_handlers()
    
    def setup_logging(self):
        """Setup logging for the aggregator"""
        self.logger = logging.getLogger("AlertAggregator")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("logs/alerts/aggregator.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def initialize_handlers(self):
        """Initialize alert handlers"""
        self.handlers = {
            'metric': MetricAlertHandler(self.config),
            'pattern': PatternAlertHandler(self.config),
            'composite': CompositeAlertHandler(self.config)
        }
    
    def process_alert(self, alert_rule: AlertRule, value: Any) -> Dict:
        """Process an alert through appropriate handler"""
        try:
            # Determine appropriate handler
            if isinstance(alert_rule, CompositeRule):
                handler = self.handlers['composite']
            elif isinstance(alert_rule, PatternRule):
                handler = self.handlers['pattern']
            else:
                handler = self.handlers['metric']
            
            # Process alert
            return handler.handle_alert(alert_rule, value)
            
        except Exception as e:
            self.logger.error(f"Error processing alert: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def resolve_alert(self, alert_id: str, handler_type: str = None) -> Dict:
        """Resolve an alert"""
        try:
            # If handler type not specified, try all handlers
            if not handler_type:
                for handler in self.handlers.values():
                    result = handler.process_resolution(alert_id)
                    if result.get('status') != 'error':
                        return result
                return {
                    'status': 'error',
                    'error': 'Alert not found in any handler'
                }
            
            # Use specified handler
            if handler_type not in self.handlers:
                return {
                    'status': 'error',
                    'error': f"Unknown handler type: {handler_type}"
                }
            
            return self.handlers[handler_type].process_resolution(alert_id)
            
        except Exception as e:
            self.logger.error(f"Error resolving alert: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_active_alerts(self) -> Dict[str, List[Dict]]:
        """Get all active alerts from all handlers"""
        active_alerts = {}
        for handler_type, handler in self.handlers.items():
            active_alerts[handler_type] = list(handler.active_alerts.values())
        return active_alerts
    
    def get_alert_history(self, start_time: datetime = None,
                         end_time: datetime = None) -> Dict[str, List[Dict]]:
        """Get alert history from all handlers"""
        history = {}
        for handler_type, handler in self.handlers.items():
            history[handler_type] = handler.get_alert_history(start_time, end_time)
        return history
