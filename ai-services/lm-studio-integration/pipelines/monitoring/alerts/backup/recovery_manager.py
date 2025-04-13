import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
import sqlite3

class RecoveryManager:
    """Manages recovery of alert state data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("RecoveryManager")
        self.logger.setLevel(logging.INFO)
        
        log_dir = Path("logs/alerts")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "recovery_manager.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def recover_from_backup(self, backup_path: Path) -> bool:
        """Recover database from backup"""
        try:
            db_path = Path(self.config['state_management']['sqlite']['db_path'])
            
            # Verify backup integrity
            if not self.verify_backup(backup_path):
                self.logger.error(f"Backup verification failed: {backup_path}")
                return False
            
            # Create recovery directory
            recovery_dir = db_path.parent / "recovery"
            recovery_dir.mkdir(exist_ok=True)
            
            # Create timestamp for recovery files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup current database if it exists
            if db_path.exists():
                current_backup = recovery_dir / f"pre_recovery_{timestamp}.db"
                shutil.copy2(db_path, current_backup)
                self.logger.info(f"Created pre-recovery backup: {current_backup}")
            
            # Restore from backup
            shutil.copy2(backup_path, db_path)
            
            # Verify restored database
            if self.verify_restored_database(db_path):
                self.logger.info(f"Successfully restored from backup: {backup_path}")
                return True
            else:
                self.logger.error(f"Restored database verification failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error recovering from backup: {str(e)}")
            return False
    
    def verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity"""
        try:
            with sqlite3.connect(str(backup_path)) as conn:
                cursor = conn.cursor()
                
                # Check required tables exist
                required_tables = ['active_alerts', 'alert_history']
                for table in required_tables:
                    cursor.execute(f"""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name=?
                    """, (table,))
                    
                    if not cursor.fetchone():
                        self.logger.error(f"Required table {table} not found in backup")
                        return False
                
                # Basic data check
                cursor.execute("SELECT COUNT(*) FROM alert_history")
                if cursor.fetchone()[0] < 0:
                    self.logger.error("Invalid record count in alert_history")
                    return False
                
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error verifying backup: {str(e)}")
            return False
    
    def verify_restored_database(self, db_path: Path) -> bool:
        """Verify restored database"""
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                
                # Check database structure
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ['active_alerts', 'alert_history']
                for table in required_tables:
                    if table not in tables:
                        self.logger.error(f"Required table {table} missing after restore")
                        return False
                
                # Check data accessibility
                for table in required_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        cursor.fetchone()
                    except sqlite3.Error:
                        self.logger.error(f"Cannot access table {table} after restore")
                        return False
                
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error verifying restored database: {str(e)}")
            return False
    
    def create_recovery_point(self) -> Optional[str]:
        """Create a recovery point"""
        try:
            db_path = Path(self.config['state_management']['sqlite']['db_path'])
            if not db_path.exists():
                self.logger.error("Database file not found")
                return None
            
            # Create recovery point directory
            recovery_dir = db_path.parent / "recovery_points"
            recovery_dir.mkdir(exist_ok=True)
            
            # Create recovery point
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recovery_point = recovery_dir / f"recovery_point_{timestamp}.db"
            
            # Copy database to recovery point
            shutil.copy2(db_path, recovery_point)
            
            # Create metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'source_path': str(db_path),
                'recovery_point': str(recovery_point),
                'size': os.path.getsize(recovery_point)
            }
            
            metadata_path = recovery_point.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Created recovery point: {recovery_point}")
            return str(recovery_point)
            
        except Exception as e:
            self.logger.error(f"Error creating recovery point: {str(e)}")
            return None
    
    def list_recovery_points(self) -> Dict[str, Dict]:
        """List available recovery points"""
        try:
            db_path = Path(self.config['state_management']['sqlite']['db_path'])
            recovery_dir = db_path.parent / "recovery_points"
            
            if not recovery_dir.exists():
                return {}
            
            recovery_points = {}
            for point in recovery_dir.glob("*.db"):
                metadata_file = point.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        'timestamp': datetime.fromtimestamp(
                            point.stat().st_mtime
                        ).isoformat(),
                        'recovery_point': str(point),
                        'size': point.stat().st_size
                    }
                
                recovery_points[str(point)] = metadata
            
            return recovery_points
            
        except Exception as e:
            self.logger.error(f"Error listing recovery points: {str(e)}")
            return {}
