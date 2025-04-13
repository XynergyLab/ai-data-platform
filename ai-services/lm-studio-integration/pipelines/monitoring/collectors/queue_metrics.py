import os
import json
import time
from datetime import datetime
from typing import Dict, List
import psutil
from prometheus_client import Counter, Gauge, Histogram

class QueueMetricsCollector:
    """Collects and processes queue metrics"""
    
    def __init__(self, config_path: str):
        self.load_config(config_path)
        self.initialize_metrics()
        
    def load_config(self, config_path: str):
        """Load collector configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.config = config['collectors']['queue']
            
    def initialize_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            'queue_size': Gauge(
                'pipeline_queue_size',
                'Current size of processing queues',
                ['queue_name', 'priority']
            ),
            'processing_rate': Gauge(
                'pipeline_processing_rate',
                'Items processed per minute',
                ['queue_name', 'priority']
            ),
            'wait_time': Histogram(
                'pipeline_queue_wait_time',
                'Time items spend in queue',
                ['queue_name', 'priority'],
                buckets=[30, 60, 300, 600, 1800, 3600]
            )
        }
        
    def collect_metrics(self):
        """Collect metrics from all queues"""
        for queue_type, queues in self.config['queues'].items():
            for queue_name, queue_path in queues.items():
                metrics = self.get_queue_metrics(queue_path)
                
                # Update Prometheus metrics
                self.metrics['queue_size'].labels(
                    queue_name=queue_name,
                    priority=self.get_queue_priority(queue_name)
                ).set(metrics['size'])
                
                self.metrics['processing_rate'].labels(
                    queue_name=queue_name,
                    priority=self.get_queue_priority(queue_name)
                ).set(metrics['processing_rate'])
                
                # Record wait times for items in queue
                for wait_time in metrics['wait_times']:
                    self.metrics['wait_time'].labels(
                        queue_name=queue_name,
                        priority=self.get_queue_priority(queue_name)
                    ).observe(wait_time)
                
    def get_queue_metrics(self, queue_path: str) -> Dict:
        """Get metrics for a specific queue"""
        try:
            metrics = {
                'size': 0,
                'processing_rate': 0,
                'wait_times': []
            }
            
            # Get queue size
            metrics['size'] = len([
                f for f in os.listdir(queue_path)
                if not f.endswith('.metadata')
            ])
            
            # Calculate processing rate and wait times
            now = datetime.now()
            for file in os.listdir(queue_path):
                if file.endswith('.metadata'):
                    continue
                    
                file_path = os.path.join(queue_path, file)
                metadata_path = f"{file_path}.metadata"
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Calculate wait time
                    created_time = datetime.fromisoformat(metadata['created_at'])
                    wait_time = (now - created_time).total_seconds()
                    metrics['wait_times'].append(wait_time)
            
            # Calculate processing rate (items per minute)
            if metrics['wait_times']:
                avg_wait = sum(metrics['wait_times']) / len(metrics['wait_times'])
                if avg_wait > 0:
                    metrics['processing_rate'] = 60 / avg_wait
            
            return metrics
            
        except Exception as e:
            print(f"Error collecting metrics for queue {queue_path}: {str(e)}")
            return {
                'size': 0,
                'processing_rate': 0,
                'wait_times': []
            }
    
    def get_queue_priority(self, queue_name: str) -> str:
        """Determine queue priority"""
        if 'high' in queue_name:
            return 'high'
        elif 'medium' in queue_name:
            return 'medium'
        else:
            return 'low'
