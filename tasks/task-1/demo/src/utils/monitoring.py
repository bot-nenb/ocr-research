"""
System Monitoring Utilities for Performance Tracking

This module provides comprehensive system monitoring capabilities including
CPU, memory, and GPU resource tracking.
"""

import csv
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import torch


@dataclass
class SystemMetrics:
    """Container for system metrics at a point in time."""
    
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    process_memory_gb: float
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_temperature: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0


class SystemMonitor:
    """Monitors system resources during processing."""
    
    def __init__(self, log_dir: str = "logs", sampling_interval: float = 1.0):
        """
        Initialize system monitor.
        
        Args:
            log_dir: Directory to save monitoring logs
            sampling_interval: Time between samples in seconds
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history: List[SystemMetrics] = []
        
        # Initialize baseline metrics
        self.process = psutil.Process(os.getpid())
        self.baseline_disk_io = psutil.disk_io_counters()
        self.baseline_network_io = psutil.net_io_counters()
        
        # Check GPU availability
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                self.nvml_available = True
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                logging.info("NVIDIA Management Library initialized for GPU monitoring")
            except Exception as e:
                self.nvml_available = False
                logging.warning(f"NVIDIA ML not available for detailed GPU monitoring: {e}")
        else:
            self.nvml_available = False
    
    def __del__(self):
        """Ensure cleanup happens even if stop() isn't called."""
        if hasattr(self, 'is_monitoring') and self.is_monitoring:
            self.stop()
    
    def start(self):
        """Start monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.metrics_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("System monitoring started")
    
    def stop(self):
        """Stop monitoring and save results."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Clean up GPU resources
        self._cleanup_gpu_resources()
        
        # Save monitoring results
        self._save_metrics()
        logging.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
            except Exception as e:
                logging.error(f"Error collecting metrics: {e}")
            
            time.sleep(self.sampling_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Process memory
        try:
            process_memory = self.process.memory_info()
            process_memory_gb = process_memory.rss / (1024**3)
        except:
            process_memory_gb = 0.0
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = (disk_io.read_bytes - self.baseline_disk_io.read_bytes) / (1024**2)
            disk_io_write_mb = (disk_io.write_bytes - self.baseline_disk_io.write_bytes) / (1024**2)
        except:
            disk_io_read_mb = 0.0
            disk_io_write_mb = 0.0
        
        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            network_sent_mb = (net_io.bytes_sent - self.baseline_network_io.bytes_sent) / (1024**2)
            network_recv_mb = (net_io.bytes_recv - self.baseline_network_io.bytes_recv) / (1024**2)
        except:
            network_sent_mb = 0.0
            network_recv_mb = 0.0
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            process_memory_gb=process_memory_gb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
        
        # GPU metrics if available
        if self.has_gpu:
            metrics.gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
            metrics.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if self.nvml_available:
                try:
                    import nvidia_ml_py3 as nvml
                    
                    # GPU utilization
                    utilization = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    metrics.gpu_utilization = utilization.gpu
                    
                    # GPU temperature
                    temperature = nvml.nvmlDeviceGetTemperature(
                        self.gpu_handle, nvml.NVML_TEMPERATURE_GPU
                    )
                    metrics.gpu_temperature = temperature
                    
                except Exception as e:
                    logging.debug(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def _cleanup_gpu_resources(self):
        """Clean up GPU resources and free memory."""
        if self.has_gpu:
            try:
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Force garbage collection on GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
                
                # Clean up NVML handle if available
                if self.nvml_available:
                    try:
                        import nvidia_ml_py3 as nvml
                        nvml.nvmlShutdown()
                        self.nvml_available = False
                        logging.debug("NVIDIA ML library shutdown complete")
                    except Exception as e:
                        logging.debug(f"Error shutting down NVIDIA ML: {e}")
                        
                logging.debug("GPU resources cleaned up")
                
            except Exception as e:
                logging.warning(f"Error during GPU cleanup: {e}")
    
    def _save_metrics(self):
        """Save metrics to files."""
        if not self.metrics_history:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = self.log_dir / f"monitoring_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(m) for m in self.metrics_history], f, indent=2)
        
        # Save as CSV for easy analysis
        csv_path = self.log_dir / f"monitoring_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.metrics_history:
                fieldnames = list(asdict(self.metrics_history[0]).keys())
                # Convert list fields to strings for CSV
                fieldnames.remove('cpu_per_core')
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for metric in self.metrics_history:
                    row = asdict(metric)
                    row.pop('cpu_per_core')  # Remove list field for CSV
                    writer.writerow(row)
        
        logging.info(f"Monitoring data saved to {json_path} and {csv_path}")
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self._collect_metrics()
    
    def get_summary(self) -> Dict:
        """Get summary statistics from monitoring history."""
        if not self.metrics_history:
            return {}
        
        summary = {
            "duration_seconds": self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp,
            "samples_collected": len(self.metrics_history),
            "cpu": {
                "avg_percent": sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history),
                "max_percent": max(m.cpu_percent for m in self.metrics_history),
                "min_percent": min(m.cpu_percent for m in self.metrics_history)
            },
            "memory": {
                "avg_percent": sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history),
                "max_percent": max(m.memory_percent for m in self.metrics_history),
                "peak_used_gb": max(m.memory_used_gb for m in self.metrics_history),
                "avg_process_gb": sum(m.process_memory_gb for m in self.metrics_history) / len(self.metrics_history),
                "peak_process_gb": max(m.process_memory_gb for m in self.metrics_history)
            }
        }
        
        if self.has_gpu:
            gpu_metrics = [m for m in self.metrics_history if m.gpu_memory_used_gb > 0]
            if gpu_metrics:
                summary["gpu"] = {
                    "avg_utilization": sum(m.gpu_utilization for m in gpu_metrics) / len(gpu_metrics),
                    "max_utilization": max(m.gpu_utilization for m in gpu_metrics),
                    "avg_memory_gb": sum(m.gpu_memory_used_gb for m in gpu_metrics) / len(gpu_metrics),
                    "peak_memory_gb": max(m.gpu_memory_used_gb for m in gpu_metrics),
                    "avg_temperature": sum(m.gpu_temperature for m in gpu_metrics) / len(gpu_metrics) if gpu_metrics[0].gpu_temperature > 0 else 0
                }
        
        return summary
    
    def print_live_stats(self):
        """Print current system stats to console."""
        metrics = self.get_current_metrics()
        
        print("\n" + "="*60)
        print("SYSTEM RESOURCE USAGE")
        print("="*60)
        print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
        print(f"Memory: {metrics.memory_used_gb:.2f}/{metrics.memory_used_gb + metrics.memory_available_gb:.2f} GB ({metrics.memory_percent:.1f}%)")
        print(f"Process Memory: {metrics.process_memory_gb:.2f} GB")
        
        if self.has_gpu:
            print(f"GPU Memory: {metrics.gpu_memory_used_gb:.2f}/{metrics.gpu_memory_total_gb:.2f} GB")
            if metrics.gpu_utilization > 0:
                print(f"GPU Utilization: {metrics.gpu_utilization:.1f}%")
                print(f"GPU Temperature: {metrics.gpu_temperature:.1f}°C")
        
        print("="*60)


class PerformanceTracker:
    """Tracks performance metrics for batch processing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.document_times = []
        self.document_sizes = []
        self.error_count = 0
        self.success_count = 0
    
    def start_tracking(self):
        """Start performance tracking."""
        self.start_time = time.time()
        self.document_times = []
        self.document_sizes = []
        self.error_count = 0
        self.success_count = 0
    
    def record_document(self, processing_time: float, doc_size: Optional[int] = None, success: bool = True):
        """Record processing metrics for a document."""
        self.document_times.append(processing_time)
        if doc_size:
            self.document_sizes.append(doc_size)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def stop_tracking(self):
        """Stop performance tracking."""
        self.end_time = time.time()
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        if not self.document_times:
            return {}
        
        total_time = (self.end_time or time.time()) - self.start_time if self.start_time else 0
        
        metrics = {
            "total_time_seconds": total_time,
            "total_documents": len(self.document_times),
            "successful_documents": self.success_count,
            "failed_documents": self.error_count,
            "success_rate": self.success_count / len(self.document_times) if self.document_times else 0,
            "avg_time_per_doc": sum(self.document_times) / len(self.document_times),
            "min_time_per_doc": min(self.document_times),
            "max_time_per_doc": max(self.document_times),
            "throughput_docs_per_second": len(self.document_times) / total_time if total_time > 0 else 0,
            "throughput_docs_per_minute": (len(self.document_times) / total_time) * 60 if total_time > 0 else 0
        }
        
        if self.document_sizes:
            total_size = sum(self.document_sizes)
            metrics.update({
                "total_size_mb": total_size / (1024 * 1024),
                "avg_size_kb": (total_size / len(self.document_sizes)) / 1024,
                "throughput_mb_per_second": (total_size / (1024 * 1024)) / total_time if total_time > 0 else 0
            })
        
        return metrics