from typing import Dict, List

class GrafanaDashboard:
    """Base class for Grafana dashboards"""
    
    def __init__(self, title: str, refresh: str = "10s"):
        self.title = title
        self.refresh = refresh
        self.panels = []
    
    def add_panel(self, panel: Dict):
        """Add panel to dashboard"""
        self.panels.append(panel)
    
    def to_json(self) -> Dict:
        """Convert dashboard to JSON format"""
        return {
            "dashboard": {
                "title": self.title,
                "refresh": self.refresh,
                "panels": self.panels
            }
        }

class QueueDashboard(GrafanaDashboard):
    """Dashboard for queue metrics"""
    
    def __init__(self):
        super().__init__("Queue Metrics")
        self.initialize_panels()
    
    def initialize_panels(self):
        """Initialize dashboard panels"""
        self.add_panel({
            "title": "Queue Sizes",
            "type": "graph",
            "metrics": ["queue_size"]
        })
        self.add_panel({
            "title": "Processing Rates",
            "type": "graph",
            "metrics": ["processing_rate"]
        })

class SystemDashboard(GrafanaDashboard):
    """Dashboard for system metrics"""
    
    def __init__(self):
        super().__init__("System Metrics")
        self.initialize_panels()
    
    def initialize_panels(self):
        """Initialize dashboard panels"""
        self.add_panel({
            "title": "CPU Usage",
            "type": "gauge",
            "metrics": ["cpu_usage"]
        })
        self.add_panel({
            "title": "Memory Usage",
            "type": "gauge",
            "metrics": ["memory_usage"]
        })
