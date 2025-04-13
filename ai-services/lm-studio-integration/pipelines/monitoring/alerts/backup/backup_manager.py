import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Optional, List
import sqlite3
from pathlib import Path

class BackupManager:
    """Manages backups of alert state data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.initialize_backup_directory()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("BackupManager")
        self.logger.setLevel(logging.INFO)
        
        log_dir = Path("logs/alerts")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "backup_manager.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def initialize_backup_directory(self):
        """Initialize backup directory structure"""
        try:
            self.backup_dir = Path(self.config['state_management']['sqlite']['backup_path'])
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different backup types
            (self.backup_dir / "daily").mkdir(exist_ok=True)
            (self.backup_dir / "weekly").mkdir(exist_ok=True)
            (self.backup_dir / "monthly").mkdir(exist_ok=True)
            
        except Exception as e:
            self.logger.error(f"Error initializing backup directory: {str(e)}")
            raise
    
    def create_backup(self, backup_type: str = "daily") -> bool:
        """Create a new backup"""
        try:
            db_path = self.config['state_management']['sqlite']['db_path']
            if not os.path.exists(db_path):
                self.logger.warning(f"Database file not found at {db_path}")
                return False
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"alert_state_{backup_type}_{timestamp}.db"
            backup_path = self.backup_dir / backup_type / backup_filename
            
            # Create backup
            shutil.copy2(db_path, backup_path)
            
            # Create metadata file
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'type': backup_type,
                'source_path': db_path,
                'backup_path': str(backup_path),
                'size': os.path.getsize(backup_path)
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Created {backup_type} backup: {backup_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False
    
    def rotate_backups(self, backup_type: str, max_backups: int) -> int:
        """Rotate old backups"""
        try:
            backup_path = self.backup_dir / backup_type
            backups = sorted(
                [f for f in backup_path.glob("*.db")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Remove old backups
            removed = 0
            for backup in backups[max_backups:]:
                # Remove backup and its metadata
                backup.unlink()
                metadata_file = backup.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()
                removed += 1
            
            self.logger.info(f"Removed {removed} old {backup_type} backups")
            return removed
            
        except Exception as e:
            self.logger.error(f"Error rotating backups: {str(e)}")
            return 0
    
    def verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity"""
        try:
            # Check if backup exists
            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Try to open and query the backup database
            with sqlite3.connect(str(backup_path)) as conn:
                cursor = conn.cursor()
                
                # Check tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                if not tables:
                    self.logger.error(f"No tables found in backup: {backup_path}")
                    return False
                
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error verifying backup {backup_path}: {str(e)}")
            return False
    
    def list_backups(self, backup_type: Optional[str] = None) -> Dict[str, List[Dict]]:
        """List available backups"""
        try:
            backups = {}
            types = [backup_type] if backup_type else ['daily', 'weekly', 'monthly']
            
            for btype in types:
                backup_path = self.backup_dir / btype
                if not backup_path.exists():
                    continue
                
                backups[btype] = []
                for backup in backup_path.glob("*.db"):
                    metadata_file = backup.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            backups[btype].append(metadata)
                    else:
                        # Create basic metadata if file doesn't exist
                        metadata = {
                            'timestamp': datetime.fromtimestamp(
                                backup.stat().st_mtime
                            ).isoformat(),
                            'type': btype,
                            'backup_path': str(backup),
                            'size': backup.stat().st_size
                        }
                        backups[btype].append(metadata)
            
            return backups
            
        except Exception as e:
            self.logger.error(f"Error listing backups: {str(e)}")
            return {}
