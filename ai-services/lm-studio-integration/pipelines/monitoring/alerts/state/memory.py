from datetime import datetime
from typing import Dict, Any, Optional, List
from .base import AlertStateManager
import logging

class InMemoryStateManager(AlertStateManager):
    """In-memory implementation of alert state manager"""
    
    def __init__(self):
        self.setup_logging()
        self.active_alerts = {}
        self.alert_history = []
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger("InMemoryStateManager")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("logs/alerts/memory_state.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def save_alert_state(self, alert_id: str, state: Dict[str, Any]) -> bool:
        """Save alert state in memory"""
        try:
            self.active_alerts[alert_id] = {
                'state': state,
                'updated_at': datetime.now()
            }
            return True
        except Exception as e:
            self.logger.error(f"Error saving alert state: {str(e)}")
            return False
    
    def load_alert_state(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Load alert state from memory"""
        try:
            alert_data = self.active_alerts.get(alert_id)
            return alert_data['state'] if alert_data else None
        except Exception as e:
            self.logger.error(f"Error loading alert state: {str(e)}")
            return None
    
    def delete_alert_state(self, alert_id: str) -> bool:
        """Delete alert state from memory"""
        try:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting alert state: {str(e)}")
            return False
    
    def list_active_alerts(self) -> List[str]:
        """List all active alert IDs"""
        return list(self.active_alerts.keys())
    
    def update_alert_history(self, alert: Dict[str, Any]) -> bool:
        """Update alert history"""
        try:
            self.alert_history.append({
                'alert': alert,
                'timestamp': datetime.now()
            })
            return True
        except Exception as e:
            self.logger.error(f"Error updating alert history: {str(e)}")
            return False
    
    def get_alert_history(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get alert history within time range"""
        try:
            filtered_history = []
            for entry in self.alert_history:
                if start_time and entry['timestamp'] < start_time:
                    continue
                if end_time and entry['timestamp'] > end_time:
                    continue
                filtered_history.append(entry['alert'])
            return filtered_history
        except Exception as e:
            self.logger.error(f"Error getting alert history: {str(e)}")
            return []
    
    def cleanup_old_history(self, max_age_days: int) -> int:
        """Clean up old history entries"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            original_length = len(self.alert_history)
            
            self.alert_history = [
                entry for entry in self.alert_history
                if entry['timestamp'] >= cutoff_date
            ]
            
            return original_length - len(self.alert_history)
        except Exception as e:
            self.logger.error(f"Error cleaning up history: {str(e)}")
            return 0
