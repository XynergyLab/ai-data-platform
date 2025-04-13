import psutil
import os
import json
from datetime import datetime
from typing import Dict
from prometheus_client import Gauge

class SystemMetricsCollector:
    """Collects and processes system metrics"""
    
    def __init__(self, config_path: str):
        self.load_config(config_path)
        self.initialize_metrics()
        
    def load_config(self, config_path: str):
        """Load collector configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.config = config['collectors']['system']
            
    def initialize_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            'cpu_usage': Gauge(
                'pipeline_cpu_usage',
                'CPU usage percentage',
                ['cpu_type']
            ),
            'memory_usage': Gauge(
                'pipeline_memory_usage',
                'Memory usage in bytes',
                ['memory_type']
            ),
            'disk_usage': Gauge(
                'pipeline_disk_usage',
                'Disk usage percentage',
                ['mount_point']
            ),
            'network_io': Gauge(
                'pipeline_network_io',
                'Network I/O bytes',
                ['interface', 'direction']
            )
        }
        
    def collect_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_metrics = self.get_cpu_metrics()
            for cpu_type, usage in cpu_metrics.items():
                self.metrics['cpu_usage'].labels(
                    cpu_type=cpu_type
                ).set(usage)
            
            # Memory metrics
            memory_metrics = self.get_memory_metrics()
            for memory_type, usage in memory_metrics.items():
                self.metrics['memory_usage'].labels(
                    memory_type=memory_type
                ).set(usage)
            
            # Disk metrics
            disk_metrics = self.get_disk_metrics()
            for mount_point, usage in disk_metrics.items():
                self.metrics['disk_usage'].labels(
                    mount_point=mount_point
                ).set(usage)
            
            # Network metrics
            network_metrics = self.get_network_metrics()
            for interface, data in network_metrics.items():
                for direction, bytes_count in data.items():
                    self.metrics['network_io'].labels(
                        interface=interface,
                        direction=direction
                    ).set(bytes_count)
                    
        except Exception as e:
            print(f"Error collecting system metrics: {str(e)}")
    
    def get_cpu_metrics(self) -> Dict:
        """Get CPU usage metrics"""
        try:
            return {
                'system': psutil.cpu_percent(interval=1),
                'user': psutil.cpu_times_percent().user,
                'system_time': psutil.cpu_times_percent().system,
                'idle': psutil.cpu_times_percent().idle
            }
        except:
            return {'system': 0, 'user': 0, 'system_time': 0, 'idle': 0}
    
    def get_memory_metrics(self) -> Dict:
        """Get memory usage metrics"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'free': memory.free,
                'cached': memory.cached if hasattr(memory, 'cached') else 0
            }
        except:
            return {'total': 0, 'available': 0, 'used': 0, 'free': 0, 'cached': 0}
    
    def get_disk_metrics(self) -> Dict:
        """Get disk usage metrics"""
        try:
            disk_metrics = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_metrics[partition.mountpoint] = usage.percent
                except:
                    continue
            return disk_metrics
        except:
            return {}
    
    def get_network_metrics(self) -> Dict:
        """Get network I/O metrics"""
        try:
            network_metrics = {}
            net_io = psutil.net_io_counters(pernic=True)
            for interface, counters in net_io.items():
                network_metrics[interface] = {
                    'bytes_sent': counters.bytes_sent,
                    'bytes_recv': counters.bytes_recv
                }
            return network_metrics
        except:
            return {}
