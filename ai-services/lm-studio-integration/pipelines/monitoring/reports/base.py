from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import csv
from jinja2 import Environment, FileSystemLoader

class ReportGenerator:
    """Generates monitoring system reports"""
    
    def __init__(self, metrics_aggregator, config: Dict):
        self.aggregator = metrics_aggregator
        self.config = config
        self.setup_logging()
        self.setup_templates()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("ReportGenerator")
        self.logger.setLevel(logging.INFO)
        
        log_dir = Path("logs/monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "report_generator.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def setup_templates(self):
        """Setup Jinja2 templates"""
        template_dir = Path("templates/reports")
        template_dir.mkdir(parents=True, exist_ok=True)
        
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
    
    def _generate_html_report(self, data: Dict[str, Any], 
                            template_name: str) -> str:
        """Generate HTML report from template"""
        try:
            template = self.env.get_template(template_name)
            return template.render(data=data)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            return ""
    
    def _generate_csv_report(self, data: Dict[str, Any]) -> str:
        """Generate CSV report"""
        try:
            # Flatten metrics data for CSV format
            rows = self._flatten_metrics(data)
            
            output = []
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            
            return '\n'.join(output)
            
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {str(e)}")
            return ""
    
    def _flatten_metrics(self, data: Dict[str, Any], 
                        prefix: str = "") -> List[Dict[str, Any]]:
        """Flatten nested metrics structure"""
        rows = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                rows.extend(self._flatten_metrics(
                    value, f"{prefix}{key}_" if prefix else f"{key}_"
                ))
            else:
                rows.append({f"{prefix}{key}": value})
        
        return rows
