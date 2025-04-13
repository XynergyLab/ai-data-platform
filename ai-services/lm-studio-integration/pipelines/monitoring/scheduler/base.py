import abc
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

class SchedulerTask:
    """Represents a scheduled task"""
    
    def __init__(self, name: str, func: Callable, interval: int,
                 args: Optional[tuple] = None, kwargs: Optional[Dict] = None):
        self.name = name
        self.func = func
        self.interval = interval
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.last_run = None
        self.next_run = None
        self.is_running = False
        
    def should_run(self) -> bool:
        """Check if task should run"""
        if not self.next_run:
            return True
        return datetime.now() >= self.next_run
    
    def update_schedule(self):
        """Update task schedule"""
        self.last_run = datetime.now()
        self.next_run = self.last_run + timedelta(seconds=self.interval)

class BaseScheduler(abc.ABC):
    """Base class for schedulers"""
    
    @abc.abstractmethod
    def add_task(self, name: str, func: Callable, interval: int,
                 args: Optional[tuple] = None, kwargs: Optional[Dict] = None):
        """Add a task to the scheduler"""
        pass
    
    @abc.abstractmethod
    def remove_task(self, name: str):
        """Remove a task from the scheduler"""
        pass
    
    @abc.abstractmethod
    def start(self):
        """Start the scheduler"""
        pass
    
    @abc.abstractmethod
    def stop(self):
        """Stop the scheduler"""
        pass
