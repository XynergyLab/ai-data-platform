import abc
from typing import Dict, Any, Optional, List
from datetime import datetime

class AlertStateManager(abc.ABC):
    """Abstract base class for alert state management"""
    
    @abc.abstractmethod
    def save_alert_state(self, alert_id: str, state: Dict[str, Any]) -> bool:
        """Save alert state"""
        pass
    
    @abc.abstractmethod
    def load_alert_state(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Load alert state"""
        pass
    
    @abc.abstractmethod
    def delete_alert_state(self, alert_id: str) -> bool:
        """Delete alert state"""
        pass
    
    @abc.abstractmethod
    def list_active_alerts(self) -> List[str]:
        """List all active alert IDs"""
        pass
    
    @abc.abstractmethod
    def update_alert_history(self, alert: Dict[str, Any]) -> bool:
        """Update alert history"""
        pass
    
    @abc.abstractmethod
    def get_alert_history(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get alert history within time range"""
        pass
    
    @abc.abstractmethod
    def cleanup_old_history(self, max_age_days: int) -> int:
        """Clean up old history entries"""
        pass

class StateManagerError(Exception):
    """Base exception for state manager errors"""
    pass

class StateNotFoundError(StateManagerError):
    """Exception raised when state is not found"""
    pass

class StateSaveError(StateManagerError):
    """Exception raised when state cannot be saved"""
    pass

class StateLoadError(StateManagerError):
    """Exception raised when state cannot be loaded"""
    pass
