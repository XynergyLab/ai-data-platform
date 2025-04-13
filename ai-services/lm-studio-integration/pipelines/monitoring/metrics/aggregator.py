from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import numpy as np
from collections import defaultdict

class MetricsAggregator:
    """Aggregates and analyzes metrics from different collectors"""
    
    def __init__(self, metrics_store, config: Dict):
        self.store = metrics_store
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("MetricsAggregator")
        self.logger.setLevel(logging.INFO)
        
        log_dir = Path("logs/monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "metrics_aggregator.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get aggregated system health metrics"""
        try:
            # Get recent system metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
            
            metrics = self.store.get_metrics(
                collector='system',
                start_time=start_time,
                end_time=end_time
            )
            
            health_metrics = {
                'cpu': self._analyze_cpu_metrics(metrics),
                'memory': self._analyze_memory_metrics(metrics),
                'disk': self._analyze_disk_metrics(metrics),
                'timestamp': end_time.isoformat()
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health metrics: {str(e)}")
            return {}
    
    def _analyze_cpu_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze CPU-related metrics"""
        cpu_metrics = [m for m in metrics if m['name'].startswith('cpu_')]
        
        if not cpu_metrics:
            return {}
        
        cpu_usage = [
            m['value'] for m in cpu_metrics 
            if m['name'] == 'cpu_usage'
        ]
        
        return {
            'current_usage': cpu_usage[-1] if cpu_usage else 0,
            'avg_usage': np.mean(cpu_usage) if cpu_usage else 0,
            'max_usage': max(cpu_usage) if cpu_usage else 0,
            'trend': self._calculate_trend(cpu_usage)
        }
    
    def _analyze_memory_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory-related metrics"""
        memory_metrics = [m for m in metrics if m['name'].startswith('memory_')]
        
        if not memory_metrics:
            return {}
        
        memory_usage = [
            m['value'] for m in memory_metrics 
            if m['name'] == 'memory_percent'
        ]
        
        return {
            'current_usage': memory_usage[-1] if memory_usage else 0,
            'avg_usage': np.mean(memory_usage) if memory_usage else 0,
            'max_usage': max(memory_usage) if memory_usage else 0,
            'trend': self._calculate_trend(memory_usage)
        }
    
    def _analyze_disk_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze disk-related metrics"""
        disk_metrics = [m for m in metrics if m['name'].startswith('disk_')]
        
        if not disk_metrics:
            return {}
        
        results = defaultdict(dict)
        
        for metric in disk_metrics:
            if 'mount_point' in metric['labels']:
                mount_point = metric['labels']['mount_point']
                name = metric['name']
                
                if name.startswith('disk_percent'):
                    if mount_point not in results:
                        results[mount_point] = {
                            'usage_values': []
                        }
                    results[mount_point]['usage_values'].append(metric['value'])
        
        disk_status = {}
        for mount_point, data in results.items():
            usage_values = data['usage_values']
            if usage_values:
                disk_status[mount_point] = {
                    'current_usage': usage_values[-1],
                    'avg_usage': np.mean(usage_values),
                    'max_usage': max(usage_values),
                    'trend': self._calculate_trend(usage_values)
                }
        
        return disk_status
    
    def get_pipeline_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated pipeline performance metrics"""
        try:
            # Get recent pipeline metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            metrics = self.store.get_metrics(
                collector='pipeline',
                start_time=start_time,
                end_time=end_time
            )
            
            performance_metrics = {
                'processing': self._analyze_processing_metrics(metrics),
                'errors': self._analyze_error_metrics(metrics),
                'resource_usage': self._analyze_resource_metrics(metrics),
                'timestamp': end_time.isoformat()
            }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline performance metrics: {str(e)}")
            return {}
    
    def _analyze_processing_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze processing-related metrics"""
        processing_metrics = [
            m for m in metrics 
            if m['name'].startswith(('processing_', 'avg_processing_'))
        ]
        
        if not processing_metrics:
            return {}
        
        success_rates = [
            m['value'] for m in processing_metrics 
            if m['name'].startswith('processing_success_rate')
        ]
        
        processing_times = [
            m['value'] for m in processing_metrics 
            if m['name'].startswith('avg_processing_time')
        ]
        
        return {
            'success_rate': {
                'current': success_rates[-1] if success_rates else 0,
                'avg': np.mean(success_rates) if success_rates else 0,
                'trend': self._calculate_trend(success_rates)
            },
            'processing_time': {
                'current': processing_times[-1] if processing_times else 0,
                'avg': np.mean(processing_times) if processing_times else 0,
                'trend': self._calculate_trend(processing_times)
            }
        }
    
    def _analyze_error_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error-related metrics"""
        error_metrics = [
            m for m in metrics 
            if m['name'].startswith(('error_', 'pipeline_error_'))
        ]
        
        if not error_metrics:
            return {}
        
        error_rates = [
            m['value'] for m in error_metrics 
            if m['name'].startswith('pipeline_error_rate')
        ]
        
        error_counts = defaultdict(list)
        for metric in error_metrics:
            if 'error_type' in metric['labels']:
                error_type = metric['labels']['error_type']
                error_counts[error_type].append(metric['value'])
        
        return {
            'error_rate': {
                'current': error_rates[-1] if error_rates else 0,
                'avg': np.mean(error_rates) if error_rates else 0,
                'trend': self._calculate_trend(error_rates)
            },
            'error_types': {
                error_type: {
                    'current': values[-1] if values else 0,
                    'total': sum(values),
                    'trend': self._calculate_trend(values)
                }
                for error_type, values in error_counts.items()
            }
        }
    
    def _analyze_resource_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource utilization metrics"""
        resource_metrics = [
            m for m in metrics 
            if m['name'].startswith('pipeline_') and 'resource' in m['labels']
        ]
        
        if not resource_metrics:
            return {}
        
        resources = defaultdict(list)
        for metric in resource_metrics:
            resource = metric['labels']['resource']
            resources[resource].append(metric['value'])
        
        return {
            resource: {
                'current': values[-1] if values else 0,
                'avg': np.mean(values) if values else 0,
                'max': max(values) if values else 0,
                'trend': self._calculate_trend(values)
            }
            for resource, values in resources.items()
        }
    
    def get_queue_metrics(self) -> Dict[str, Any]:
        """Get aggregated queue metrics"""
        try:
            # Get recent queue metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=15)
            
            metrics = self.store.get_metrics(
                collector='queue',
                start_time=start_time,
                end_time=end_time
            )
            
            queue_metrics = self._analyze_queue_metrics(metrics)
            queue_metrics['timestamp'] = end_time.isoformat()
            
            return queue_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting queue metrics: {str(e)}")
            return {}
    
    def _analyze_queue_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze queue-related metrics"""
        if not metrics:
            return {}
        
        queues = defaultdict(lambda: defaultdict(list))
        
        for metric in metrics:
            if 'queue_name' in metric['labels'] and 'priority' in metric['labels']:
                queue = metric['labels']['queue_name']
                priority = metric['labels']['priority']
                name = metric['name']
                
                if name.startswith('queue_size'):
                    queues[queue][priority].append(metric['value'])
        
        results = {}
        for queue, priorities in queues.items():
            results[queue] = {
                priority: {
                    'current_size': values[-1] if values else 0,
                    'avg_size': np.mean(values) if values else 0,
                    'max_size': max(values) if values else 0,
                    'trend': self._calculate_trend(values)
                }
                for priority, values in priorities.items()
            }
        
        return results
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values"""
        if not values or len(values) < 2:
            return 'stable'
        
        # Calculate simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend based on slope
        if abs(slope) < 0.1:  # Threshold for considering trend significant
            return 'stable'
        return 'increasing' if slope > 0 else 'decreasing'
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate daily metrics report"""
        try:
            if not date:
                date = datetime.now()
            
            start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=1)
            
            report = {
                'date': date.date().isoformat(),
                'system_metrics': self._generate_system_report(start_time, end_time),
                'pipeline_metrics': self._generate_pipeline_report(start_time, end_time),
                'queue_metrics': self._generate_queue_report(start_time, end_time),
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {str(e)}")
            return {}
    
    def _generate_system_report(self, start_time: datetime, 
                              end_time: datetime) -> Dict[str, Any]:
        """Generate system metrics report"""
        metrics = self.store.get_metrics(
            collector='system',
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            'cpu': self._analyze_cpu_metrics(metrics),
            'memory': self._analyze_memory_metrics(metrics),
            'disk': self._analyze_disk_metrics(metrics)
        }
    
    def _generate_pipeline_report(self, start_time: datetime, 
                                end_time: datetime) -> Dict[str, Any]:
        """Generate pipeline metrics report"""
        metrics = self.store.get_metrics(
            collector='pipeline',
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            'processing': self._analyze_processing_metrics(metrics),
            'errors': self._analyze_error_metrics(metrics),
            'resource_usage': self._analyze_resource_metrics(metrics)
        }
    
    def _generate_queue_report(self, start_time: datetime, 
                             end_time: datetime) -> Dict[str, Any]:
        """Generate queue metrics report"""
        metrics = self.store.get_metrics(
            collector='queue',
            start_time=start_time,
            end_time=end_time
        )
        
        return self._analyze_queue_metrics(metrics)
