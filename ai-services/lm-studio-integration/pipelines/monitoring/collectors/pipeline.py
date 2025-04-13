import os
from typing import Dict, Any, List
from datetime import datetime
import logging
from pathlib import Path
import json
from .base import MetricsCollector

class PipelineMetricsCollector(MetricsCollector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.metrics_path = Path(config['collectors']['pipeline'].get(
            'metrics_path', 'data/pipeline_metrics'
        ))
        self.metrics_path.mkdir(parents=True, exist_ok=True)
    
    def collect_metrics(self) -> Dict[str, Any]:
        try:
            metrics = {}
            
            # Collect processing metrics
            processing_metrics = self._collect_processing_metrics()
            metrics.update(processing_metrics)
            
            # Collect error metrics
            error_metrics = self._collect_error_metrics()
            metrics.update(error_metrics)
            
            # Collect performance metrics
            performance_metrics = self._collect_performance_metrics()
            metrics.update(performance_metrics)
            
            self.update_metrics(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f'Error collecting pipeline metrics: {str(e)}')
            return {}    def _collect_error_metrics(self) -> Dict[str, Any]:
        """Collect error-related metrics"""
        try:
            metrics = {}
            timestamp = datetime.now().isoformat()
            
            # Read error stats file
            error_file = self.metrics_path / 'error_stats.json'
            if error_file.exists():
                with open(error_file, 'r') as f:
                    error_stats = json.load(f)
                
                # Overall error rate
                total_ops = error_stats.get('total_operations', 0)
                total_errors = error_stats.get('total_errors', 0)
                error_rate = (total_errors / total_ops * 100) if total_ops > 0 else 0
                
                metrics['pipeline_error_rate'] = {
                    'value': error_rate,
                    'labels': {'type': 'overall'},
                    'timestamp': timestamp
                }
                
                # Error counts by type
                for error_type, count in error_stats.get('error_types', {}).items():
                    metrics[f'error_count_{error_type}'] = {
                        'value': count,
                        'labels': {'error_type': error_type},
                        'timestamp': timestamp
                    }
                
                # Error rates by processor
                for processor, proc_stats in error_stats.get('processor_errors', {}).items():
                    proc_ops = proc_stats.get('operations', 0)
                    proc_errors = proc_stats.get('errors', 0)
                    proc_error_rate = (proc_errors / proc_ops * 100) if proc_ops > 0 else 0
                    
                    metrics[f'processor_error_rate_{processor}'] = {
                        'value': proc_error_rate,
                        'labels': {
                            'processor': processor,
                            'type': 'error_rate'
                        },
                        'timestamp': timestamp
                    }
                    
                    # Error types distribution for processor
                    for error_type, type_count in proc_stats.get('error_types', {}).items():
                        metrics[f'processor_error_count_{processor}_{error_type}'] = {
                            'value': type_count,
                            'labels': {
                                'processor': processor,
                                'error_type': error_type
                            },
                            'timestamp': timestamp
                        }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting error metrics: {str(e)}")
            return {}
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics"""
        try:
            metrics = {}
            timestamp = datetime.now().isoformat()
            
            # Read performance stats file
            perf_file = self.metrics_path / 'performance_stats.json'
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    perf_stats = json.load(f)
                
                # Overall processing time statistics
                metrics.update({
                    'avg_processing_time': {
                        'value': perf_stats.get('average_processing_time', 0),
                        'labels': {'type': 'overall'},
                        'timestamp': timestamp
                    },
                    'max_processing_time': {
                        'value': perf_stats.get('max_processing_time', 0),
                        'labels': {'type': 'overall'},
                        'timestamp': timestamp
                    },
                    'min_processing_time': {
                        'value': perf_stats.get('min_processing_time', 0),
                        'labels': {'type': 'overall'},
                        'timestamp': timestamp
                    }
                })
                
                # Per-processor performance metrics
                for processor, proc_stats in perf_stats.get('processors', {}).items():
                    metrics.update({
                        f'processor_avg_time_{processor}': {
                            'value': proc_stats.get('average_time', 0),
                            'labels': {
                                'processor': processor,
                                'metric': 'average_time'
                            },
                            'timestamp': timestamp
                        },
                        f'processor_max_time_{processor}': {
                            'value': proc_stats.get('max_time', 0),
                            'labels': {
                                'processor': processor,
                                'metric': 'max_time'
                            },
                            'timestamp': timestamp
                        },
                        f'processor_min_time_{processor}': {
                            'value': proc_stats.get('min_time', 0),
                            'labels': {
                                'processor': processor,
                                'metric': 'min_time'
                            },
                            'timestamp': timestamp
                        }
                    })
                    
                    # Performance by file type
                    for file_type, type_stats in proc_stats.get('file_types', {}).items():
                        metrics.update({
                            f'processor_avg_time_{processor}_{file_type}': {
                                'value': type_stats.get('average_time', 0),
                                'labels': {
                                    'processor': processor,
                                    'file_type': file_type,
                                    'metric': 'average_time'
                                },
                                'timestamp': timestamp
                            }
                        })
                
                # Resource utilization metrics
                resources = perf_stats.get('resource_utilization', {})
                metrics.update({
                    'pipeline_cpu_usage': {
                        'value': resources.get('cpu_usage', 0),
                        'labels': {'resource': 'cpu'},
                        'timestamp': timestamp
                    },
                    'pipeline_memory_usage': {
                        'value': resources.get('memory_usage', 0),
                        'labels': {'resource': 'memory'},
                        'timestamp': timestamp
                    },
                    'pipeline_disk_io': {
                        'value': resources.get('disk_io', 0),
                        'labels': {'resource': 'disk'},
                        'timestamp': timestamp
                    }
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {str(e)}")
            return {}
    
    def get_metrics_summary(self, start_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of pipeline metrics"""
        summary = super().get_metrics_summary(start_time)
        
        if summary:
            # Add pipeline-specific summaries
            try:
                # Calculate overall success rate
                success_rates = [
                    v['value'] for k, v in summary.items()
                    if k.startswith('processing_success_rate')
                ]
                if success_rates:
                    summary['overall_success_rate'] = sum(success_rates) / len(success_rates)
                
                # Calculate overall error rate
                error_rates = [
                    v['value'] for k, v in summary.items()
                    if k.startswith('pipeline_error_rate')
                ]
                if error_rates:
                    summary['overall_error_rate'] = sum(error_rates) / len(error_rates)
                
                # Calculate average processing time
                proc_times = [
                    v['value'] for k, v in summary.items()
                    if k.startswith('avg_processing_time')
                ]
                if proc_times:
                    summary['overall_avg_processing_time'] = sum(proc_times) / len(proc_times)
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics summary: {str(e)}")
        
        return summary
