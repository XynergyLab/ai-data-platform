import threading
import time
from datetime import datetime
from typing import Callable, Dict, Any, Optional
from croniter import croniter
from .base import BaseScheduler, SchedulerTask
import logging

class CronScheduler(BaseScheduler):
    """Scheduler that runs tasks using cron expressions"""
    
    def __init__(self):
        self.tasks = {}
        self.running = False
        self.setup_logging()
        self._thread = None
        self._lock = threading.Lock()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("CronScheduler")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("logs/cron_scheduler.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def add_task(self, name: str, func: Callable, cron_expr: str,
                 args: Optional[tuple] = None, kwargs: Optional[Dict] = None):
        """Add a task with cron expression"""
        with self._lock:
            if name in self.tasks:
                self.logger.warning(f"Task {name} already exists, updating")
            
            if not croniter.is_valid(cron_expr):
                raise ValueError(f"Invalid cron expression: {cron_expr}")
            
            task = CronTask(name, func, cron_expr, args, kwargs)
            self.tasks[name] = task
            self.logger.info(f"Added task: {name} with cron: {cron_expr}")
    
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
            now = datetime.now()
            for task in self.tasks.values():
                if task.should_run(now) and not task.is_running:
                    self._run_task(task)
    
    def _run_task(self, task: 'CronTask'):
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

class CronTask(SchedulerTask):
    """Task scheduled with cron expression"""
    
    def __init__(self, name: str, func: Callable, cron_expr: str,
                 args: Optional[tuple] = None, kwargs: Optional[Dict] = None):
        super().__init__(name, func, 0, args, kwargs)  # interval not used
        self.cron_expr = cron_expr
        self.cron = croniter(cron_expr, datetime.now())
        self.next_run = self.cron.get_next(datetime)
    
    def should_run(self, current_time: datetime) -> bool:
        """Check if task should run"""
        return current_time >= self.next_run
    
    def update_schedule(self):
        """Update next run time"""
        self.last_run = datetime.now()
        self.next_run = self.cron.get_next(datetime)
