from typing import Dict, Optional
from .base import AlertStateManager
from .sqlite import SQLiteStateManager
from .memory import InMemoryStateManager
import logging

class StateManagerFactory:
    """Factory for creating state managers"""
    
    def __init__(self):
        self.setup_logging()
        self._managers = {}
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger("StateManagerFactory")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("logs/alerts/state_factory.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def create_manager(self, manager_type: str, config: Optional[Dict] = None) -> AlertStateManager:
        """Create a state manager instance"""
        try:
            # Check if manager already exists
            if manager_type in self._managers:
                return self._managers[manager_type]
            
            # Create new manager
            if manager_type == 'sqlite':
                if not config or 'db_path' not in config:
                    raise ValueError("SQLite manager requires 'db_path' in config")
                manager = SQLiteStateManager(config['db_path'])
            elif manager_type == 'memory':
                manager = InMemoryStateManager()
            else:
                raise ValueError(f"Unknown state manager type: {manager_type}")
            
            # Cache and return manager
            self._managers[manager_type] = manager
            return manager
            
        except Exception as e:
            self.logger.error(f"Error creating state manager: {str(e)}")
            raise
    
    def get_manager(self, manager_type: str) -> Optional[AlertStateManager]:
        """Get existing state manager instance"""
        return self._managers.get(manager_type)
    
    def list_managers(self) -> List[str]:
        """List all active manager types"""
        return list(self._managers.keys())
