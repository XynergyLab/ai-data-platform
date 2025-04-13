import sqlite3
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base import AlertStateManager, StateNotFoundError, StateSaveError, StateLoadError
import logging

class SQLiteStateManager(AlertStateManager):
    """SQLite-based alert state manager"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_logging()
        self.initialize_database()
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger("SQLiteStateManager")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("logs/alerts/state_manager.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def initialize_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create active alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS active_alerts (
                        alert_id TEXT PRIMARY KEY,
                        alert_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create alert history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT NOT NULL,
                        alert_data TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT NOT NULL
                    )
                """)
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise StateManagerError(f"Failed to initialize database: {str(e)}")
    
    def save_alert_state(self, alert_id: str, state: Dict[str, Any]) -> bool:
        """Save alert state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert state to JSON
                state_json = json.dumps(state)
                
                # Update or insert state
                cursor.execute("""
                    INSERT OR REPLACE INTO active_alerts 
                    (alert_id, alert_data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (alert_id, state_json))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error saving alert state: {str(e)}")
            raise StateSaveError(f"Failed to save alert state: {str(e)}")
    
    def load_alert_state(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Load alert state from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT alert_data FROM active_alerts
                    WHERE alert_id = ?
                """, (alert_id,))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Error loading alert state: {str(e)}")
            raise StateLoadError(f"Failed to load alert state: {str(e)}")
    
    def delete_alert_state(self, alert_id: str) -> bool:
        """Delete alert state from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM active_alerts
                    WHERE alert_id = ?
                """, (alert_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting alert state: {str(e)}")
            return False
    
    def list_active_alerts(self) -> List[str]:
        """List all active alert IDs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT alert_id FROM active_alerts")
                return [row[0] for row in cursor.fetchall()]
                
        except sqlite3.Error as e:
            self.logger.error(f"Error listing active alerts: {str(e)}")
            return []
    
    def update_alert_history(self, alert: Dict[str, Any]) -> bool:
        """Update alert history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                alert_json = json.dumps(alert)
                status = alert.get('status', 'unknown')
                
                cursor.execute("""
                    INSERT INTO alert_history 
                    (alert_id, alert_data, status)
                    VALUES (?, ?, ?)
                """, (alert['id'], alert_json, status))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error updating alert history: {str(e)}")
            return False
    
    def get_alert_history(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get alert history within time range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT alert_data FROM alert_history"
                params = []
                
                if start_time or end_time:
                    conditions = []
                    if start_time:
                        conditions.append("timestamp >= ?")
                        params.append(start_time.isoformat())
                    if end_time:
                        conditions.append("timestamp <= ?")
                        params.append(end_time.isoformat())
                    
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                return [json.loads(row[0]) for row in cursor.fetchall()]
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting alert history: {str(e)}")
            return []
    
    def cleanup_old_history(self, max_age_days: int) -> int:
        """Clean up old history entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = (
                    datetime.now() - timedelta(days=max_age_days)
                ).isoformat()
                
                cursor.execute("""
                    DELETE FROM alert_history
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                return deleted_count
                
        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up history: {str(e)}")
            return 0
