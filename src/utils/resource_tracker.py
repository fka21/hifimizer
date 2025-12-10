# utils/resource_tracker.py
import time
import psutil
import logging
from typing import Dict

class ResourceTracker:
    """
    Utility class to track CPU and memory usage of a process.
    """
    def __init__(self, log_interval: int = 60, timeout_hours=48, trial_id = None):
        """
        Initialize the resource tracker.
        
        Args:
            log_interval: Interval in seconds between log entries
        """
        self.log_interval = log_interval
        self.cpu_counts = []
        self.mem_usage_gb = []
        self.start_time = None
        self.timeout = timeout_hours * 3600
        self.end_time = None
        self.trial_id = trial_id
        self.logger = logging.getLogger(f'ResourceTracker_{trial_id}')
        
    def track_process(self, process) -> None:
        """
        Track resources for a given process until completion.
        
        Args:
            process: subprocess.Popen object to track
        """
        self.start_time = time.time()
        proc = psutil.Process(process.pid)
        proc.cpu_percent(interval=None)  # Warm-up CPU stats
        last_log_time = time.time()

        while process.poll() is None:  # While process is running
            # Timeout check
            if time.time() - self.start_time > self.timeout:
                process.terminate()
                raise TimeoutError("Trial exceeded 48-hour limit")

            # Resource monitoring
            try:
                all_procs = [proc] + proc.children(recursive=True)
                total_cpu = sum(p.cpu_num() for p in all_procs)
                total_mem = sum(p.memory_info().rss/(1024**3) for p in all_procs)
                
                self.cpu_counts.append(total_cpu)
                self.mem_usage_gb.append(total_mem)

                # Logging
                if time.time() - last_log_time >= self.log_interval:
                    logging.info(f"CPU: {total_cpu}, Memory: {total_mem} GB")
                    last_log_time = time.time()
                    
            except psutil.NoSuchProcess:
                break

        self.end_time = time.time()
        
        if time.time() - last_log_time >= self.log_interval:
            self.logger.info(f"CPU: {total_cpu}, Memory: {total_mem} GB")

    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about resource usage.
        
        Returns:
            Dictionary with cpu_avg, mem_avg_gb, and runtime
        """
        return {
            'cpu_avg': sum(self.cpu_counts) / len(self.cpu_counts) if self.cpu_counts else 0,
            'mem_avg_gb': sum(self.mem_usage_gb) / len(self.mem_usage_gb) if self.mem_usage_gb else 0,
            'runtime': self.end_time - self.start_time if self.start_time and self.end_time else 0
        }
    
    def set_trial_attributes(self, trial) -> None:
        """
        Set resource usage as attributes on an Optuna trial.
        
        Args:
            trial: Optuna trial object
        """
        stats = self.get_statistics()
        for key, value in stats.items():
            trial.set_user_attr(key, value)
