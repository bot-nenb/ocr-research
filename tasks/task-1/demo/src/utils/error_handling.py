"""
Comprehensive Error Handling and Recovery System

This module provides robust error handling, logging, and recovery mechanisms
for OCR batch processing operations.
"""

import functools
import json
import logging
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import psutil


class OCRProcessingError(Exception):
    """Base exception for OCR processing errors."""
    def __init__(self, message: str, doc_id: str = None, error_code: str = None):
        super().__init__(message)
        self.doc_id = doc_id
        self.error_code = error_code
        self.timestamp = datetime.now()


class DatasetError(OCRProcessingError):
    """Errors related to dataset loading and preparation."""
    pass


class ModelInitializationError(OCRProcessingError):
    """Errors related to OCR model initialization."""
    pass


class ProcessingTimeoutError(OCRProcessingError):
    """Errors related to processing timeouts."""
    pass


class ResourceExhaustionError(OCRProcessingError):
    """Errors related to system resource exhaustion."""
    pass


class ErrorLogger:
    """Centralized error logging and reporting system with singleton pattern."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, log_dir: str = "logs"):
        if cls._instance is None:
            cls._instance = super(ErrorLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize error logger (singleton pattern).
        
        Args:
            log_dir: Directory to store error logs
        """
        # Only initialize once
        if self._initialized:
            return
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.error_summary_file = self.log_dir / "error_summary.json"
        self.error_history: List[Dict] = []
        
        # Load existing error history
        if self.error_summary_file.exists():
            try:
                with open(self.error_summary_file, 'r') as f:
                    self.error_history = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load error history: {e}")
        
        self.logger = logging.getLogger(__name__)
        self._initialized = True
    
    def log_error(self, error: Exception, 
                  context: Dict = None,
                  doc_id: str = None,
                  recovery_attempted: bool = False,
                  recovery_successful: bool = False):
        """
        Log error with comprehensive context information.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            doc_id: Document ID if applicable
            recovery_attempted: Whether recovery was attempted
            recovery_successful: Whether recovery was successful
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "doc_id": doc_id,
            "context": context or {},
            "recovery_attempted": recovery_attempted,
            "recovery_successful": recovery_successful,
            "traceback": traceback.format_exc(),
            "system_info": self._get_system_info()
        }
        
        # Add specific error information
        if isinstance(error, OCRProcessingError):
            error_entry["error_code"] = getattr(error, 'error_code', None)
            error_entry["doc_id"] = getattr(error, 'doc_id', doc_id)
        
        self.error_history.append(error_entry)
        
        # Log to standard logger
        self.logger.error(
            f"Error processing {doc_id or 'unknown'}: {error}",
            extra={
                "error_type": type(error).__name__,
                "doc_id": doc_id,
                "recovery_attempted": recovery_attempted
            }
        )
        
        # Save error summary
        self._save_error_summary()
    
    def _get_system_info(self) -> Dict:
        """Get current system information for error context."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            }
        except Exception:
            return {}
    
    def _save_error_summary(self):
        """Save error summary to file."""
        try:
            with open(self.error_summary_file, 'w') as f:
                json.dump(self.error_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Could not save error summary: {e}")
    
    def get_error_summary(self) -> Dict:
        """Get summary of all recorded errors."""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_types = {}
        recovery_stats = {"attempted": 0, "successful": 0}
        recent_errors = []
        
        for error in self.error_history:
            # Count error types
            error_type = error.get("error_type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count recovery attempts
            if error.get("recovery_attempted"):
                recovery_stats["attempted"] += 1
                if error.get("recovery_successful"):
                    recovery_stats["successful"] += 1
            
            # Track recent errors (last 10)
            if len(recent_errors) < 10:
                recent_errors.append({
                    "timestamp": error.get("timestamp"),
                    "error_type": error.get("error_type"),
                    "doc_id": error.get("doc_id"),
                    "message": error.get("error_message", "")[:100]
                })
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recovery_stats": recovery_stats,
            "recovery_rate": (recovery_stats["successful"] / recovery_stats["attempted"] * 100) if recovery_stats["attempted"] > 0 else 0,
            "recent_errors": recent_errors
        }
    
    def clear_history(self):
        """Clear error history."""
        self.error_history = []
        self._save_error_summary()


class RetryMechanism:
    """Implements retry logic with exponential backoff."""
    
    @staticmethod
    def retry_with_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,),
        logger: logging.Logger = None
    ):
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Backoff multiplier
            exceptions: Tuple of exceptions to catch and retry
            logger: Logger for retry messages
            
        Returns:
            Result of successful function call
            
        Raises:
            Last exception if all retries fail
        """
        import time
        
        if logger is None:
            logger = logging.getLogger(__name__)
        
        last_exception = None
        delay = base_delay
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(f"All {max_retries} retry attempts failed for {func.__name__}")
                    break
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
        
        raise last_exception


def retry_on_failure(max_retries: int = 3, 
                    base_delay: float = 1.0,
                    exceptions: tuple = (Exception,)):
    """Decorator for automatic retry on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return RetryMechanism.retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                exceptions=exceptions,
                logger=logging.getLogger(func.__module__)
            )
        return wrapper
    return decorator


class ErrorResult:
    """Wrapper for error handling results."""
    def __init__(self, value: Any = None, error: Exception = None):
        self.value = value
        self.error = error
        self.has_error = error is not None

@contextmanager
def error_handler(error_logger: ErrorLogger = None,
                 doc_id: str = None,
                 context: Dict = None,
                 reraise: bool = True,
                 default_return: Any = None):
    """
    Context manager for comprehensive error handling.
    
    Args:
        error_logger: ErrorLogger instance
        doc_id: Document ID for context
        context: Additional context information
        reraise: Whether to reraise the exception
        default_return: Default return value if exception is caught
        
    Yields:
        ErrorResult: Wrapper object to store results
    """
    if error_logger is None:
        error_logger = ErrorLogger()
    
    result = ErrorResult()
    
    try:
        yield result
    except Exception as e:
        error_logger.log_error(e, context, doc_id)
        
        if reraise:
            raise
        else:
            result.value = default_return
            result.error = e


class ResourceMonitor:
    """Monitors system resources and prevents resource exhaustion."""
    
    def __init__(self, 
                 memory_threshold: float = 0.9,
                 cpu_threshold: float = 0.95,
                 disk_threshold: float = 0.9):
        """
        Initialize resource monitor.
        
        Args:
            memory_threshold: Memory usage threshold (0.0-1.0)
            cpu_threshold: CPU usage threshold (0.0-1.0) 
            disk_threshold: Disk usage threshold (0.0-1.0)
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.disk_threshold = disk_threshold
        self.logger = logging.getLogger(__name__)
    
    def check_resources(self) -> Dict[str, bool]:
        """
        Check if system resources are within acceptable limits.
        
        Returns:
            Dictionary with resource status
        """
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            memory_ok = memory.percent / 100 < self.memory_threshold
            cpu_ok = cpu / 100 < self.cpu_threshold
            disk_ok = disk.percent / 100 < self.disk_threshold
            
            status = {
                "memory_ok": memory_ok,
                "cpu_ok": cpu_ok,
                "disk_ok": disk_ok,
                "overall_ok": all([memory_ok, cpu_ok, disk_ok]),
                "memory_percent": memory.percent,
                "cpu_percent": cpu,
                "disk_percent": disk.percent
            }
            
            if not status["overall_ok"]:
                self.logger.warning(
                    f"Resource constraints detected: "
                    f"Memory: {memory.percent:.1f}%, "
                    f"CPU: {cpu:.1f}%, "
                    f"Disk: {disk.percent:.1f}%"
                )
            
            return status
            
        except Exception as e:
            self.logger.error(f"Could not check system resources: {e}")
            return {"overall_ok": True}  # Assume OK if check fails
    
    def wait_for_resources(self, max_wait_time: float = 300.0, check_interval: float = 5.0):
        """
        Wait for system resources to become available.
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            
        Raises:
            ResourceExhaustionError: If resources don't become available
        """
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.check_resources()
            
            if status["overall_ok"]:
                return
            
            self.logger.info(f"Waiting for resources... ({time.time() - start_time:.1f}s elapsed)")
            time.sleep(check_interval)
        
        raise ResourceExhaustionError(
            f"System resources not available after {max_wait_time} seconds",
            error_code="RESOURCE_TIMEOUT"
        )


class GracefulShutdown:
    """Handles graceful shutdown of processing operations."""
    
    def __init__(self):
        """Initialize graceful shutdown handler."""
        self.shutdown_requested = False
        self.cleanup_functions = []
        self.logger = logging.getLogger(__name__)
        
        # Register signal handlers
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.cleanup()
    
    def register_cleanup(self, cleanup_func: Callable):
        """Register a cleanup function to run on shutdown."""
        self.cleanup_functions.append(cleanup_func)
    
    def cleanup(self):
        """Run all registered cleanup functions."""
        self.logger.info("Running cleanup functions...")
        
        for cleanup_func in self.cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
        
        self.logger.info("Cleanup completed")
    
    def check_shutdown(self):
        """Check if shutdown has been requested."""
        return self.shutdown_requested


def safe_file_operation(operation: Callable, 
                       file_path: Union[str, Path],
                       error_logger: ErrorLogger = None,
                       max_retries: int = 3) -> Any:
    """
    Safely perform file operations with error handling and retries.
    
    Args:
        operation: File operation function
        file_path: Path to file
        error_logger: Error logger instance
        max_retries: Maximum retry attempts
        
    Returns:
        Result of file operation
    """
    if error_logger is None:
        error_logger = ErrorLogger()
    
    def file_op():
        return operation(file_path)
    
    try:
        return RetryMechanism.retry_with_backoff(
            file_op,
            max_retries=max_retries,
            exceptions=(IOError, OSError, PermissionError)
        )
    except Exception as e:
        error_logger.log_error(
            e, 
            context={"file_path": str(file_path), "operation": operation.__name__}
        )
        raise


class HealthChecker:
    """Performs health checks on the OCR processing system."""
    
    def __init__(self):
        """Initialize health checker."""
        self.logger = logging.getLogger(__name__)
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available."""
        checks = {}
        
        # Check Python packages
        required_packages = [
            'easyocr', 'opencv-python', 'pillow', 'numpy', 
            'pandas', 'matplotlib', 'seaborn', 'tqdm', 
            'psutil', 'Levenshtein', 'click', 'jinja2'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                checks[f"package_{package}"] = True
            except ImportError:
                checks[f"package_{package}"] = False
                self.logger.warning(f"Package {package} not available")
        
        # Check system tools
        import shutil
        system_tools = ['python', 'pip']
        
        for tool in system_tools:
            checks[f"tool_{tool}"] = shutil.which(tool) is not None
        
        return checks
    
    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and capabilities."""
        gpu_info = {
            "available": False,
            "device_count": 0,
            "memory_total": 0,
            "memory_free": 0
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["device_count"] = torch.cuda.device_count()
                
                if gpu_info["device_count"] > 0:
                    gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
                    gpu_info["memory_free"] = torch.cuda.memory_reserved(0)
                    
        except Exception as e:
            self.logger.debug(f"Could not check GPU availability: {e}")
        
        return gpu_info
    
    def run_full_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "checks": {}
        }
        
        # Check dependencies
        dep_checks = self.check_dependencies()
        health_status["checks"]["dependencies"] = dep_checks
        
        if not all(dep_checks.values()):
            health_status["overall_healthy"] = False
        
        # Check system resources
        resource_monitor = ResourceMonitor()
        resource_status = resource_monitor.check_resources()
        health_status["checks"]["resources"] = resource_status
        
        if not resource_status["overall_ok"]:
            health_status["overall_healthy"] = False
        
        # Check GPU
        gpu_status = self.check_gpu_availability()
        health_status["checks"]["gpu"] = gpu_status
        
        # Check disk space for temp files
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            health_status["checks"]["disk_space"] = {
                "free_gb": free_gb,
                "sufficient": free_gb > 1.0  # Need at least 1GB free
            }
            
            if free_gb < 1.0:
                health_status["overall_healthy"] = False
                
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            health_status["checks"]["disk_space"] = {"error": str(e)}
        
        return health_status