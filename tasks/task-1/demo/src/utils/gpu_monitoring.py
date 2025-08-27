"""
GPU Performance Monitoring Module

Provides cross-platform GPU monitoring for CUDA and MPS (Apple Silicon).
Tracks GPU utilization, memory usage, and performance metrics.
"""

import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import psutil

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Container for GPU performance metrics."""
    timestamp: float
    gpu_utilization: float  # Percentage
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature: Optional[float] = None  # Celsius
    power_draw: Optional[float] = None  # Watts
    device_name: str = ""
    device_type: str = ""  # cuda, mps, etc.


class GPUMonitor:
    """Cross-platform GPU monitoring for CUDA and MPS devices."""
    
    def __init__(self, device_type: str = "auto"):
        """
        Initialize GPU monitor.
        
        Args:
            device_type: Type of GPU device ('cuda', 'mps', 'auto')
        """
        self.device_type = self._detect_device(device_type)
        self.metrics_history: List[GPUMetrics] = []
        self.monitoring_active = False
        self.start_time = None
        
        logger.info(f"GPU Monitor initialized for device type: {self.device_type}")
    
    def _detect_device(self, device_type: str) -> str:
        """Detect available GPU device type."""
        if device_type != "auto":
            return device_type
            
        # Try to detect CUDA
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        
        # Try to detect MPS (Apple Silicon)
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
            
        # Check platform for additional hints
        if platform.system() == "Darwin" and platform.processor() == "arm":
            return "mps"
            
        return "cpu"
    
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics based on device type."""
        if self.device_type == "cuda":
            return self._get_cuda_metrics()
        elif self.device_type == "mps":
            return self._get_mps_metrics()
        else:
            return None
    
    def _get_cuda_metrics(self) -> Optional[GPUMetrics]:
        """Get CUDA GPU metrics using nvidia-smi."""
        try:
            # Use nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = None
                    
                    # Get power draw
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
                    except:
                        power = None
                    
                    # Get device name
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    return GPUMetrics(
                        timestamp=time.time(),
                        gpu_utilization=util.gpu,
                        memory_used_mb=mem_info.used / 1024 / 1024,
                        memory_total_mb=mem_info.total / 1024 / 1024,
                        memory_percent=(mem_info.used / mem_info.total) * 100,
                        temperature=temp,
                        power_draw=power,
                        device_name=name,
                        device_type="cuda"
                    )
                    
            except ImportError:
                pass
            
            # Fallback to nvidia-smi command
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) >= 6:
                    return GPUMetrics(
                        timestamp=time.time(),
                        gpu_utilization=float(values[0]),
                        memory_used_mb=float(values[1]),
                        memory_total_mb=float(values[2]),
                        memory_percent=(float(values[1]) / float(values[2])) * 100,
                        temperature=float(values[3]) if values[3] != 'N/A' else None,
                        power_draw=float(values[4]) if values[4] != 'N/A' else None,
                        device_name=values[5],
                        device_type="cuda"
                    )
                    
        except Exception as e:
            logger.debug(f"Failed to get CUDA metrics: {e}")
            
        return None
    
    def _get_mps_metrics(self) -> Optional[GPUMetrics]:
        """Get MPS (Apple Silicon) metrics."""
        try:
            metrics = GPUMetrics(
                timestamp=time.time(),
                gpu_utilization=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                memory_percent=0.0,
                device_name="Apple Silicon",
                device_type="mps"
            )
            
            # Try to get Metal Performance Shaders info using ioreg and powermetrics
            if platform.system() == "Darwin":
                # Get GPU utilization using powermetrics (requires sudo)
                # Note: In production, you'd want to use a daemon or different approach
                try:
                    # Try Activity Monitor stats via ps
                    result = subprocess.run(
                        ['ps', '-A', '-o', '%cpu,%mem,command'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        # Parse for Python/Metal processes
                        total_gpu_estimate = 0.0
                        for line in result.stdout.split('\n'):
                            if 'python' in line.lower() or 'metal' in line.lower():
                                parts = line.strip().split(None, 2)
                                if len(parts) >= 2:
                                    try:
                                        cpu_percent = float(parts[0])
                                        # Rough estimate: high CPU often correlates with GPU usage on M1/M2
                                        if cpu_percent > 50:
                                            total_gpu_estimate += cpu_percent * 0.7
                                    except:
                                        pass
                        
                        metrics.gpu_utilization = min(total_gpu_estimate, 100.0)
                
                except Exception as e:
                    logger.debug(f"Could not get GPU utilization: {e}")
                
                # Get memory pressure
                try:
                    result = subprocess.run(
                        ['vm_stat'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        # Parse vm_stat output
                        vm_stats = {}
                        for line in result.stdout.split('\n'):
                            if ':' in line:
                                key, value = line.split(':', 1)
                                try:
                                    # Extract number from "value."
                                    vm_stats[key.strip()] = int(value.strip().rstrip('.'))
                                except:
                                    pass
                        
                        # Calculate memory usage (rough approximation for unified memory)
                        page_size = 4096  # Default page size
                        if 'Pages wired down' in vm_stats and 'Pages active' in vm_stats:
                            wired = vm_stats['Pages wired down'] * page_size / 1024 / 1024
                            active = vm_stats['Pages active'] * page_size / 1024 / 1024
                            
                            # Get total memory
                            total_mem = psutil.virtual_memory().total / 1024 / 1024
                            
                            # Estimate GPU memory usage (portion of wired memory)
                            metrics.memory_used_mb = wired * 0.3  # Rough estimate
                            metrics.memory_total_mb = total_mem * 0.5  # Unified memory can use up to half
                            metrics.memory_percent = (metrics.memory_used_mb / metrics.memory_total_mb) * 100
                            
                except Exception as e:
                    logger.debug(f"Could not get memory stats: {e}")
                
                # Get temperature using osx-cpu-temp if available
                try:
                    result = subprocess.run(
                        ['osx-cpu-temp'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        # Parse temperature
                        import re
                        match = re.search(r'(\d+\.?\d*)', result.stdout)
                        if match:
                            metrics.temperature = float(match.group(1))
                            
                except:
                    # Try alternative method using powermetrics
                    pass
                
                # Get power consumption estimate
                try:
                    # Use ioreg to get power info
                    result = subprocess.run(
                        ['ioreg', '-r', '-c', 'AppleSmartBattery'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        # Parse for InstantAmperage and Voltage
                        import re
                        amp_match = re.search(r'"InstantAmperage"\s*=\s*(-?\d+)', result.stdout)
                        volt_match = re.search(r'"Voltage"\s*=\s*(\d+)', result.stdout)
                        
                        if amp_match and volt_match:
                            # Calculate power in watts
                            amperage = abs(int(amp_match.group(1))) / 1000.0  # Convert to Amps
                            voltage = int(volt_match.group(1)) / 1000.0  # Convert to Volts
                            metrics.power_draw = amperage * voltage
                            
                except Exception as e:
                    logger.debug(f"Could not get power stats: {e}")
            
            # Use psutil as fallback for basic metrics
            if metrics.memory_total_mb == 0:
                mem = psutil.virtual_memory()
                metrics.memory_used_mb = mem.used / 1024 / 1024
                metrics.memory_total_mb = mem.total / 1024 / 1024
                metrics.memory_percent = mem.percent
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get MPS metrics: {e}")
            return None
    
    def start_monitoring(self):
        """Start collecting GPU metrics."""
        self.monitoring_active = True
        self.start_time = time.time()
        self.metrics_history = []
        logger.info(f"Started GPU monitoring for {self.device_type}")
    
    def stop_monitoring(self):
        """Stop collecting GPU metrics."""
        self.monitoring_active = False
        logger.info(f"Stopped GPU monitoring. Collected {len(self.metrics_history)} samples")
    
    def collect_sample(self):
        """Collect a single GPU metrics sample."""
        if not self.monitoring_active:
            return
            
        metrics = self.get_current_metrics()
        if metrics:
            self.metrics_history.append(metrics)
    
    def get_summary(self) -> Dict:
        """Get summary statistics of collected metrics."""
        if not self.metrics_history:
            return {
                "device_type": self.device_type,
                "samples_collected": 0,
                "monitoring_duration": 0,
                "error": "No metrics collected"
            }
        
        # Calculate statistics
        gpu_utils = [m.gpu_utilization for m in self.metrics_history]
        mem_used = [m.memory_used_mb for m in self.metrics_history]
        mem_percent = [m.memory_percent for m in self.metrics_history]
        temps = [m.temperature for m in self.metrics_history if m.temperature]
        powers = [m.power_draw for m in self.metrics_history if m.power_draw]
        
        duration = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            "device_type": self.device_type,
            "device_name": self.metrics_history[0].device_name if self.metrics_history else "Unknown",
            "samples_collected": len(self.metrics_history),
            "monitoring_duration": duration,
            "gpu_utilization": {
                "min": min(gpu_utils) if gpu_utils else 0,
                "max": max(gpu_utils) if gpu_utils else 0,
                "avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
            },
            "memory_usage_mb": {
                "min": min(mem_used) if mem_used else 0,
                "max": max(mem_used) if mem_used else 0,
                "avg": sum(mem_used) / len(mem_used) if mem_used else 0,
                "total": self.metrics_history[0].memory_total_mb if self.metrics_history else 0,
            },
            "memory_percent": {
                "min": min(mem_percent) if mem_percent else 0,
                "max": max(mem_percent) if mem_percent else 0,
                "avg": sum(mem_percent) / len(mem_percent) if mem_percent else 0,
            }
        }
        
        # Add optional metrics if available
        if temps:
            summary["temperature_celsius"] = {
                "min": min(temps),
                "max": max(temps),
                "avg": sum(temps) / len(temps),
            }
            
        if powers:
            summary["power_watts"] = {
                "min": min(powers),
                "max": max(powers),
                "avg": sum(powers) / len(powers),
                "total_kwh": (sum(powers) / len(powers)) * duration / 3600 / 1000,  # kWh
            }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export collected metrics to JSON file."""
        data = {
            "summary": self.get_summary(),
            "metrics": [
                {
                    "timestamp": m.timestamp,
                    "gpu_utilization": m.gpu_utilization,
                    "memory_used_mb": m.memory_used_mb,
                    "memory_percent": m.memory_percent,
                    "temperature": m.temperature,
                    "power_draw": m.power_draw,
                }
                for m in self.metrics_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported GPU metrics to {filepath}")