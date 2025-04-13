from typing import Dict, List
import threading
from .monitor_base import BaseMonitor
from ..collectors import (
    QueueCollector,
    SystemCollector,
    PipelineCollector
)
from ..alerts import AlertManager

class MonitoringManager(BaseMonitor):
    """Manages all monitoring components"""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.initialize_collectors()
        self.initialize_alert_manager()
        self.start_monitoring()
    
    def initialize_collectors(self):
        """Initialize metric collectors"""
        self.collectors = {
            'queue': QueueCollector(self.config),
            'system': SystemCollector(self.config),
            'pipeline': PipelineCollector(self.config)
        }
    
    def initialize_alert_manager(self):
        """Initialize alert management"""
        self.alert_manager = AlertManager(self.config)
    
    def start_monitoring(self):
        """Start all monitoring threads"""
        self.monitoring_threads = []
        
        for collector in self.collectors.values():
            thread = threading.Thread(
                target=collector.run,
                daemon=True
            )
            self.monitoring_threads.append(thread)
            thread.start()
        
        # Start alert manager
        alert_thread = threading.Thread(
            target=self.alert_manager.run,
            daemon=True
        )
        self.monitoring_threads.append(alert_thread)
        alert_thread.start()
