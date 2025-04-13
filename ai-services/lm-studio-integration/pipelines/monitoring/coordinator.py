import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .collectors import QueueMetricsCollector, SystemMetricsCollector, PipelineMetricsCollector
from .alerts import AlertAggregator
from .alerts.state import StateManagerFactory
from .alerts.backup import BackupManager, RecoveryManager
from .scheduler import IntervalScheduler, CronScheduler

class MonitoringCoordinator:
    """Coordinates all monitoring components"""
    
    def __init__(self, config_path: str):
        self.load_config(config_path)
        self.setup_logging()
        self.initialize_components()
    
    def load_config(self, config_path: str):
        """Load monitoring configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("MonitoringCoordinator")
        self.logger.setLevel(logging.INFO)
        
        log_dir = Path("logs/monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "coordinator.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def initialize_components(self):
        """Initialize all monitoring components"""
        try:
            # Initialize collectors
            self.collectors = {
                'queue': QueueMetricsCollector(self.config),
                'system': SystemMetricsCollector(self.config),
                'pipeline': PipelineMetricsCollector(self.config)
            }
            
            # Initialize alert management
            self.alert_aggregator = AlertAggregator(self.config)
            
            # Initialize state management
            state_factory = StateManagerFactory()
            self.state_manager = state_factory.create_manager(
                self.config['state_management']['default_manager'],
                self.config['state_management']
            )
            
            # Initialize backup management
            self.backup_manager = BackupManager(self.config)
            self.recovery_manager = RecoveryManager(self.config)
            
            # Initialize schedulers
            self.interval_scheduler = IntervalScheduler()
            self.cron_scheduler = CronScheduler()
            
            self.setup_scheduled_tasks()
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def setup_scheduled_tasks(self):
        """Setup scheduled monitoring tasks"""
        try:
            # Setup collector tasks
            for name, collector in self.collectors.items():
                interval = self.config['collectors'][name]['interval']
                self.interval_scheduler.add_task(
                    f"collect_{name}",
                    collector.collect_metrics,
                    interval
                )
            
            # Setup backup tasks
            backup_config = self.config['state_management']['sqlite']
            self.cron_scheduler.add_task(
                "daily_backup",
                self.backup_manager.create_backup,
                "0 0 * * *",  # Every day at midnight
                args=("daily",)
            )
            self.cron_scheduler.add_task(
                "weekly_backup",
                self.backup_manager.create_backup,
                "0 0 * * 0",  # Every Sunday at midnight
                args=("weekly",)
            )
            self.cron_scheduler.add_task(
                "monthly_backup",
                self.backup_manager.create_backup,
                "0 0 1 * *",  # First day of every month
                args=("monthly",)
            )
            
            # Setup cleanup tasks
            self.cron_scheduler.add_task(
                "rotate_backups",
                self.rotate_old_backups,
                "0 1 * * *"  # Every day at 1 AM
            )
            
            # Setup state cleanup task
            max_age = backup_config.get('max_history_days', 30)
            self.cron_scheduler.add_task(
                "cleanup_history",
                self.state_manager.cleanup_old_history,
                "0 2 * * *",  # Every day at 2 AM
                args=(max_age,)
            )
            
        except Exception as e:
            self.logger.error(f"Error setting up scheduled tasks: {str(e)}")
            raise
    
    def start_monitoring(self):
        """Start all monitoring components"""
        try:
            self.logger.info("Starting monitoring system")
            
            # Start schedulers
            self.interval_scheduler.start()
            self.cron_scheduler.start()
            
            self.logger.info("Monitoring system started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
            self.stop_monitoring()
            raise
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        try:
            self.logger.info("Stopping monitoring system")
            
            # Stop schedulers
            self.interval_scheduler.stop()
            self.cron_scheduler.stop()
            
            self.logger.info("Monitoring system stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}")
            raise
    
    def rotate_old_backups(self):
        """Rotate old backups based on configuration"""
        try:
            backup_config = self.config['state_management']['sqlite']
            
            # Rotate daily backups
            self.backup_manager.rotate_backups(
                'daily',
                backup_config.get('max_daily_backups', 7)
            )
            
            # Rotate weekly backups
            self.backup_manager.rotate_backups(
                'weekly',
                backup_config.get('max_weekly_backups', 4)
            )
            
            # Rotate monthly backups
            self.backup_manager.rotate_backups(
                'monthly',
                backup_config.get('max_monthly_backups', 12)
            )
            
        except Exception as e:
            self.logger.error(f"Error rotating backups: {str(e)}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current status of monitoring system"""
        try:
            status = {
                'collectors': {},
                'alerts': {
                    'active': self.alert_aggregator.get_active_alerts(),
                    'total_triggered': len(self.state_manager.get_alert_history())
                },
                'backups': self.backup_manager.list_backups(),
                'components': {
                    'interval_scheduler': self.interval_scheduler.running,
                    'cron_scheduler': self.cron_scheduler.running
                }
            }
            
            # Get collector statuses
            for name, collector in self.collectors.items():
                status['collectors'][name] = {
                    'last_collection': collector.last_collection,
                    'metrics_count': len(collector.metrics)
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting monitoring status: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_metrics_summary(self, time_range: Optional[str] = "1h") -> Dict[str, Any]:
        """Get summary of collected metrics"""
        try:
            # Parse time range
            range_map = {
                "1h": timedelta(hours=1),
                "6h": timedelta(hours=6),
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7)
            }
            
            if time_range not in range_map:
                raise ValueError(f"Invalid time range: {time_range}")
            
            time_delta = range_map[time_range]
            start_time = datetime.now() - time_delta
            
            summary = {}
            for name, collector in self.collectors.items():
                summary[name] = collector.get_metrics_summary(start_time)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting metrics summary: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
