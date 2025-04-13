import os
import json
import jsonschema
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

class ConfigurationValidator:
    """Validates monitoring system configuration"""
    
    def __init__(self):
        self.setup_logging()
        self.load_schemas()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("ConfigValidator")
        self.logger.setLevel(logging.INFO)
        
        log_dir = Path("logs/monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "config_validator.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def load_schemas(self):
        """Load JSON schemas for configuration validation"""
        self.schemas = {
            'collectors': {
                'type': 'object',
                'properties': {
                    'queue': {
                        'type': 'object',
                        'required': ['enabled', 'interval', 'metrics', 'queues'],
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'interval': {'type': 'integer', 'minimum': 1},
                            'metrics': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'required': ['name', 'type', 'description'],
                                    'properties': {
                                        'name': {'type': 'string'},
                                        'type': {'enum': ['gauge', 'counter', 'histogram']},
                                        'description': {'type': 'string'},
                                        'labels': {
                                            'type': 'array',
                                            'items': {'type': 'string'}
                                        }
                                    }
                                }
                            },
                            'queues': {'type': 'object'}
                        }
                    },
                    'system': {
                        'type': 'object',
                        'required': ['enabled', 'interval', 'metrics'],
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'interval': {'type': 'integer', 'minimum': 1},
                            'metrics': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'required': ['name', 'type', 'description'],
                                    'properties': {
                                        'name': {'type': 'string'},
                                        'type': {'enum': ['gauge', 'counter', 'histogram']},
                                        'description': {'type': 'string'},
                                        'labels': {
                                            'type': 'array',
                                            'items': {'type': 'string'}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    'pipeline': {
                        'type': 'object',
                        'required': ['enabled', 'interval', 'metrics'],
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'interval': {'type': 'integer', 'minimum': 1},
                            'metrics': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'required': ['name', 'type', 'description'],
                                    'properties': {
                                        'name': {'type': 'string'},
                                        'type': {'enum': ['gauge', 'counter', 'histogram']},
                                        'description': {'type': 'string'},
                                        'labels': {
                                            'type': 'array',
                                            'items': {'type': 'string'}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            'alerts': {
                'type': 'object',
                'properties': {
                    'queue_alerts': {'type': 'object'},
                    'system_alerts': {'type': 'object'},
                    'pipeline_alerts': {'type': 'object'},
                    'notification_channels': {
                        'type': 'object',
                        'properties': {
                            'slack': {
                                'type': 'object',
                                'required': ['type', 'webhook_url', 'channel'],
                                'properties': {
                                    'type': {'enum': ['slack']},
                                    'webhook_url': {'type': 'string'},
                                    'channel': {'type': 'string'}
                                }
                            },
                            'email': {
                                'type': 'object',
                                'required': ['type', 'from', 'to'],
                                'properties': {
                                    'type': {'enum': ['email']},
                                    'from': {'type': 'string'},
                                    'to': {
                                        'type': 'array',
                                        'items': {'type': 'string'}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            'state_management': {
                'type': 'object',
                'required': ['default_manager', 'sqlite'],
                'properties': {
                    'default_manager': {'enum': ['sqlite', 'memory']},
                    'sqlite': {
                        'type': 'object',
                        'required': ['db_path', 'backup_path'],
                        'properties': {
                            'db_path': {'type': 'string'},
                            'backup_path': {'type': 'string'},
                            'backup_interval': {'type': 'integer', 'minimum': 1},
                            'max_history_days': {'type': 'integer', 'minimum': 1}
                        }
                    }
                }
            }
        }
    
    def validate_config(self, config: Dict) -> List[str]:
        """Validate configuration against schemas"""
        errors = []
        
        try:
            # Validate each section
            for section, schema in self.schemas.items():
                if section not in config:
                    errors.append(f"Missing required section: {section}")
                    continue
                
                try:
                    jsonschema.validate(config[section], schema)
                except jsonschema.exceptions.ValidationError as e:
                    errors.append(f"Validation error in {section}: {str(e)}")
            
            # Additional custom validations
            errors.extend(self._validate_paths(config))
            errors.extend(self._validate_intervals(config))
            errors.extend(self._validate_metric_names(config))
            
            return errors
            
        except Exception as e:
            self.logger.error(f"Error during configuration validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return errors
    
    def _validate_paths(self, config: Dict) -> List[str]:
        """Validate file paths in configuration"""
        errors = []
        
        # Check SQLite paths
        db_path = Path(config['state_management']['sqlite']['db_path'])
        backup_path = Path(config['state_management']['sqlite']['backup_path'])
        
        # Validate db_path
        if not db_path.parent.exists():
            errors.append(f"Database directory does not exist: {db_path.parent}")
        
        # Validate backup_path
        if not backup_path.exists():
            try:
                backup_path.mkdir(parents=True)
            except Exception as e:
                errors.append(f"Cannot create backup directory: {str(e)}")
        
        return errors
    
    def _validate_intervals(self, config: Dict) -> List[str]:
        """Validate collection and backup intervals"""
        errors = []
        
        # Check collector intervals
        for collector, settings in config['collectors'].items():
            interval = settings.get('interval')
            if interval and interval < 10:
                errors.append(
                    f"Collector interval too low for {collector}: {interval} seconds"
                )
        
        # Check backup interval
        backup_interval = config['state_management']['sqlite'].get('backup_interval')
        if backup_interval and backup_interval < 3600:
            errors.append(
                f"Backup interval too low: {backup_interval} seconds (minimum 1 hour)"
            )
        
        return errors
    
    def _validate_metric_names(self, config: Dict) -> List[str]:
        """Validate metric names for uniqueness"""
        errors = []
        metric_names = set()
        
        for collector in config['collectors'].values():
            for metric in collector.get('metrics', []):
                name = metric.get('name')
                if name in metric_names:
                    errors.append(f"Duplicate metric name: {name}")
                metric_names.add(name)
        
        return errors
    
    def generate_default_config(self) -> Dict[str, Any]:
        """Generate default configuration"""
        return {
            'collectors': {
                'queue': {
                    'enabled': True,
                    'interval': 30,
                    'metrics': [
                        {
                            'name': 'queue_size',
                            'type': 'gauge',
                            'description': 'Current size of processing queues',
                            'labels': ['queue_name', 'priority']
                        },
                        {
                            'name': 'processing_rate',
                            'type': 'gauge',
                            'description': 'Items processed per minute',
                            'labels': ['queue_name', 'priority']
                        }
                    ],
                    'queues': {
                        'ingestion': {
                            'high_priority': '/pipelines/ingestion/high',
                            'medium_priority': '/pipelines/ingestion/medium',
                            'low_priority': '/pipelines/ingestion/low'
                        }
                    }
                },
                'system': {
                    'enabled': True,
                    'interval': 60,
                    'metrics': [
                        {
                            'name': 'cpu_usage',
                            'type': 'gauge',
                            'description': 'CPU usage percentage',
                            'labels': ['cpu_type']
                        },
                        {
                            'name': 'memory_usage',
                            'type': 'gauge',
                            'description': 'Memory usage in bytes',
                            'labels': ['memory_type']
                        }
                    ]
                }
            },
            'alerts': {
                'queue_alerts': {
                    'high_queue_size': {
                        'type': 'threshold',
                        'metric': 'queue_size',
                        'threshold': 1000,
                        'severity': 'warning'
                    }
                },
                'notification_channels': {
                    'slack': {
                        'type': 'slack',
                        'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
                        'channel': '#monitoring'
                    }
                }
            },
            'state_management': {
                'default_manager': 'sqlite',
                'sqlite': {
                    'db_path': 'data/monitoring.db',
                    'backup_path': 'data/backups',
                    'backup_interval': 86400,
                    'max_history_days': 30
                }
            }
        }
    
    def update_config(self, config_path: str, updates: Dict) -> bool:
        """Update existing configuration"""
        try:
            # Load existing config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply updates
            self._deep_update(config, updates)
            
            # Validate updated config
            errors = self.validate_config(config)
            if errors:
                self.logger.error("Configuration validation failed:")
                for error in errors:
                    self.logger.error(f"  - {error}")
                return False
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return False
    
    def _deep_update(self, d: Dict, u: Dict):
        """Recursively update dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

def create_default_config(config_path: str) -> bool:
    """Create default configuration file"""
    try:
        validator = ConfigurationValidator()
        config = validator.generate_default_config()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        logging.error(f"Error creating default configuration: {str(e)}")
        return False
