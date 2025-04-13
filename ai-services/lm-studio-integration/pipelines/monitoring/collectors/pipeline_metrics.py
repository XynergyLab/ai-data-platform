import os
import json
from datetime import datetime
from typing import Dict, List
from prometheus_client import Counter, Gauge, Histogram

class PipelineMetricsCollector:
    """Collects and processes pipeline metrics"""
    
    def __init__(self, config_path: str):
        self.load_config(config_path)
        self.initialize_metrics()
        
    def load_config(self, config_path: str):
        """Load collector configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.config = config['collectors']['pipeline']
            
    def initialize_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            'processing_time': Histogram(
                'pipeline_processing_time',
                'Time taken to process items',
                ['processor_type', 'file_type'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            'success_rate': Gauge(
                'pipeline_success_rate',
                'Processing success rate',
                ['processor_type']
            ),
            'error_rate': Gauge(
                'pipeline_error_rate',
                'Processing error rate',
                ['processor_type', 'error_type']
            ),
            'throughput': Counter(
                'pipeline_throughput',
                'Items processed per second',
                ['processor_type']
            )
        }
        
    def collect_metrics(self):
        """Collect pipeline processing metrics"""
        try:
            # Get processor metrics
            for processor_type in ['text', 'image', 'video', 'embedding']:
                metrics = self.get_processor_metrics(processor_type)
                
                # Update success/error rates
                self.metrics['success_rate'].labels(
                    processor_type=processor_type
                ).set(metrics['success_rate'])
                
                for error_type, rate in metrics['error_rates'].items():
                    self.metrics['error_rate'].labels(
                        processor_type=processor_type,
                        error_type=error_type
                    ).set(rate)
                
                # Update throughput
                self.metrics['throughput'].labels(
                    processor_type=processor_type
                ).inc(metrics['items_processed'])
                
                # Record processing times
                for file_type, times in metrics['processing_times'].items():
                    for time_value in times:
                        self.metrics['processing_time'].labels(
                            processor_type=processor_type,
                            file_type=file_type
                        ).observe(time_value)
                        
        except Exception as e:
            print(f"Error collecting pipeline metrics: {str(e)}")
    
    def get_processor_metrics(self, processor_type: str) -> Dict:
        """Get metrics for a specific processor"""
        try:
            # This would typically come from the processor's metrics storage
            # For now, we'll return sample data
            return {
                'success_rate': 0.95,
                'error_rates': {
                    'validation_error': 0.03,
                    'processing_error': 0.02
                },
                'items_processed': 100,
                'processing_times': {
                    'text/plain': [0.5, 0.6, 0.4],
                    'image/jpeg': [1.2, 1.5, 1.1]
                }
            }
        except:
            return {
                'success_rate': 0,
                'error_rates': {},
                'items_processed': 0,
                'processing_times': {}
            }
