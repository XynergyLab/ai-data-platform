import sqlite3
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

class MetricsStore:
    """Handles storage and retrieval of metrics data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.initialize_storage()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("MetricsStore")
        self.logger.setLevel(logging.INFO)
        
        log_dir = Path("logs/monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "metrics_store.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def initialize_storage(self):
        """Initialize the metrics database"""
        try:
            db_path = Path(self.config['metrics_storage']['db_path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.db_path = str(db_path)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        labels TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        collector TEXT NOT NULL
                    )
                """)
                
                # Create index on timestamp and name
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON metrics(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name 
                    ON metrics(name)
                """)
                
                # Create metrics summary table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics_summary (
                        name TEXT PRIMARY KEY,
                        min_value REAL,
                        max_value REAL,
                        avg_value REAL,
                        last_value REAL,
                        last_update TIMESTAMP,
                        collector TEXT NOT NULL
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing metrics storage: {str(e)}")
            raise
    
    def store_metrics(self, metrics: Dict[str, Any], collector: str):
        """Store collected metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for name, data in metrics.items():
                    # Store raw metric
                    cursor.execute("""
                        INSERT INTO metrics (name, value, labels, timestamp, collector)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        name,
                        data['value'],
                        json.dumps(data.get('labels', {})),
                        data['timestamp'],
                        collector
                    ))
                    
                    # Update summary
                    self._update_metric_summary(cursor, name, data['value'], 
                                             data['timestamp'], collector)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing metrics: {str(e)}")
    
    def _update_metric_summary(self, cursor: sqlite3.Cursor, name: str, 
                             value: float, timestamp: str, collector: str):
        """Update metrics summary"""
        try:
            # Get existing summary
            cursor.execute("""
                SELECT min_value, max_value, avg_value, last_value
                FROM metrics_summary
                WHERE name = ?
            """, (name,))
            
            result = cursor.fetchone()
            
            if result:
                min_val, max_val, avg_val, last_val = result
                
                # Update summary
                cursor.execute("""
                    UPDATE metrics_summary
                    SET min_value = ?,
                        max_value = ?,
                        avg_value = ?,
                        last_value = ?,
                        last_update = ?
                    WHERE name = ?
                """, (
                    min(min_val, value),
                    max(max_val, value),
                    (avg_val + value) / 2,  # Simple moving average
                    value,
                    timestamp,
                    name
                ))
            else:
                # Insert new summary
                cursor.execute("""
                    INSERT INTO metrics_summary 
                    (name, min_value, max_value, avg_value, last_value, 
                     last_update, collector)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, value, value, value, value, timestamp, collector
                ))
                
        except Exception as e:
            self.logger.error(f"Error updating metric summary: {str(e)}")
            raise
    
    def get_metrics(self, name: Optional[str] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   collector: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve metrics based on criteria"""
        try:
            query = "SELECT name, value, labels, timestamp, collector FROM metrics"
            params = []
            conditions = []
            
            if name:
                conditions.append("name = ?")
                params.append(name)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())
            
            if collector:
                conditions.append("collector = ?")
                params.append(collector)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append({
                        'name': row[0],
                        'value': row[1],
                        'labels': json.loads(row[2]),
                        'timestamp': row[3],
                        'collector': row[4]
                    })
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error retrieving metrics: {str(e)}")
            return []
    
    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific metric"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT min_value, max_value, avg_value, last_value,
                           last_update, collector
                    FROM metrics_summary
                    WHERE name = ?
                """, (name,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'name': name,
                        'min_value': result[0],
                        'max_value': result[1],
                        'avg_value': result[2],
                        'last_value': result[3],
                        'last_update': result[4],
                        'collector': result[5]
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving metric summary: {str(e)}")
            return None
    
    def cleanup_old_metrics(self, max_age_days: int) -> int:
        """Clean up metrics older than specified days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM metrics
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {str(e)}")
            return 0
