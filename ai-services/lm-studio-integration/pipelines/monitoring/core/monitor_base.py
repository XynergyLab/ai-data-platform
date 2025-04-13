import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import threading
import time

class BaseMonitor:
    """Base class for all monitoring components"""
    
    def __init__(self, config_path: str):
        self.load_config(config_path)
        self.setup_logging()
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(self.__class__.__name__)
        log_dir = os.path.join('logs', 'monitoring')
        os.makedirs(log_dir, exist_ok=True)
        
        handler = logging.FileHandler(
            os.path.join(log_dir, f"{self.__class__.__name__.lower()}.log")
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
