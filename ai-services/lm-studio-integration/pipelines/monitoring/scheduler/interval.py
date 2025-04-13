import threading
import time
from datetime import datetime
from typing import Callable, Dict, Any, Optional
from .base import BaseScheduler, SchedulerTask
import logging

class IntervalScheduler(BaseScheduler):
    """Scheduler that runs tasks at fixed intervals"""
    
    def __init__(self):
        self.tasks = {}
        self.running = False
        self.setup_logging()
        self._thread = None
        self._lock = threading.Lock()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("IntervalScheduler")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("logs/scheduler.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def add_task(self, name: str, func: Callable, interval: int,
                 args: Optional[tuple] = None, kwargs: Optional[Dict] = None):
        """Add a task to the scheduler"""
        with self._lock:
            if name in self.tasks:
                self.logger.warning(f"Task {name} already exists, updating")
            
            task = SchedulerTask(name, func, interval, args, kwargs)
            self.tasks[name] = task
            self.logger.info(f"Added task: {name}")
    
    def remove_task(self, name: str):
        """Remove a task from the scheduler"""
        with self._lock:
            if name in self.tasks:
                del self.tasks[name]
                self.logger.info(f"Removed task: {name}")
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        self.logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        self.logger.info("Scheduler stopped")
    
    def _run(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self._check_tasks()
                time.sleep(1)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
    
    def _check_tasks(self):
        """Check and run due tasks"""
        with self._lock:
            for task in self.tasks.values():
                if task.should_run() and not task.is_running:
                    self._run_task(task)
    
    def _run_task(self, task: SchedulerTask):
        """Run a specific task"""
        try:
            task.is_running = True
            self.logger.info(f"Running task: {task.name}")
            
            task.func(*task.args, **task.kwargs)
            
            task.update_schedule()
            self.logger.info(f"Task completed: {task.name}")
            
        except Exception as e:
            self.logger.error(f"Error running task {task.name}: {str(e)}")
        finally:
            task.is_running = False
