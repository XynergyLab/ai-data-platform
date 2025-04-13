import psutil
from typing import Dict
from datetime import datetime
from ..core.monitor_base import BaseMonitor

class SystemCollector(BaseMonitor):
    """Collects system metrics"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.initialize_metrics()
    
    def initialize_metrics(self):
        """Initialize system metrics"""
        self.metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': {}
        }
    
    def collect_metrics(self):
        """Collect current system metrics"""
        # CPU Usage
        self.metrics['cpu_usage'] = psutil.cpu_percent()
        
        # Memory Usage
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'] = memory.percent
        
        # Disk Usage
        for partition in psutil.disk_partitions():
            usage = psutil.disk_usage(partition.mountpoint)
            self.metrics['disk_usage'][partition.mountpoint] = usage.percent
