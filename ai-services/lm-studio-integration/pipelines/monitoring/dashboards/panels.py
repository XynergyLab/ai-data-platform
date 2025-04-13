from typing import Dict, List

class Panel:
    """Base class for dashboard panels"""
    
    def __init__(self, title: str, type: str):
        self.title = title
        self.type = type
        self.targets = []
    
    def add_target(self, target: Dict):
        """Add data target to panel"""
        self.targets.append(target)
    
    def to_json(self) -> Dict:
        """Convert panel to JSON format"""
        return {
            "title": self.title,
            "type": self.type,
            "targets": self.targets
        }

class GraphPanel(Panel):
    """Graph panel type"""
    
    def __init__(self, title: str):
        super().__init__(title, "graph")
        self.yaxes = []
        self.alert = None
    
    def set_axes(self, min_value: float = None, max_value: float = None):
        """Set panel axes"""
        self.yaxes = [{
            "min": min_value,
            "max": max_value
        }]

class TablePanel(Panel):
    """Table panel type"""
    
    def __init__(self, title: str):
        super().__init__(title, "table")
        self.columns = []
    
    def add_column(self, column: Dict):
        """Add column to table"""
        self.columns.append(column)
