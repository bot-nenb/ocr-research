"""
GPU Performance Report Generator

Generates performance-focused HTML reports for GPU/MPS runs.
Tracks only processing time and resource usage, no accuracy statistics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class GPUPerformanceReportGenerator:
    """Generate performance-only HTML reports for GPU/MPS runs."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize report generator."""
        self.output_dir = Path(output_dir) if output_dir else Path("results/gpu_performance")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_performance_report(
        self,
        processing_results: List[Dict],
        gpu_metrics: Dict,
        system_metrics: Dict,
        device_type: str,
        model_name: str,
        config: Dict
    ) -> Path:
        """
        Generate performance-focused HTML report.
        
        Args:
            processing_results: List of processing results with timing info
            gpu_metrics: GPU monitoring summary from GPUMonitor
            system_metrics: System monitoring summary
            device_type: Type of device (cuda, mps)
            model_name: OCR model name
            config: Configuration used
            
        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device_label = "gpu" if device_type == "cuda" else device_type
        
        # Calculate performance statistics
        processing_times = [r.get('processing_time', 0) for r in processing_results if r.get('success')]
        transform_times = [r.get('transform_time', 0) for r in processing_results if r.get('success')]
        total_times = [p + t for p, t in zip(processing_times, transform_times)]
        
        perf_stats = {
            'total_documents': len(processing_results),
            'successful_documents': len([r for r in processing_results if r.get('success')]),
            'failed_documents': len([r for r in processing_results if not r.get('success')]),
            'total_processing_time': sum(total_times),
            'avg_processing_time': np.mean(total_times) if total_times else 0,
            'std_processing_time': np.std(total_times) if total_times else 0,
            'min_processing_time': min(total_times) if total_times else 0,
            'max_processing_time': max(total_times) if total_times else 0,
            'throughput_docs_per_sec': len(total_times) / sum(total_times) if sum(total_times) > 0 else 0,
            'throughput_docs_per_min': (len(total_times) / sum(total_times)) * 60 if sum(total_times) > 0 else 0,
        }
        
        # Generate HTML content
        html_content = self._generate_html(
            perf_stats,
            gpu_metrics,
            system_metrics,
            device_type,
            model_name,
            config,
            timestamp,
            processing_times,
            transform_times
        )
        
        # Save report
        report_filename = f"{model_name.lower()}_{device_label}_performance_{timestamp}.html"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"GPU performance report saved to {report_path}")
        return report_path
    
    def _generate_html(
        self,
        perf_stats: Dict,
        gpu_metrics: Dict,
        system_metrics: Dict,
        device_type: str,
        model_name: str,
        config: Dict,
        timestamp: str,
        processing_times: List[float],
        transform_times: List[float]
    ) -> str:
        """Generate HTML content for performance report."""
        
        # Determine device display name
        if device_type == "cuda":
            device_display = f"NVIDIA GPU ({gpu_metrics.get('device_name', 'Unknown')})"
            device_icon = "⚡"
        elif device_type == "mps":
            device_display = "Apple Silicon (MPS)"
            device_icon = "🍎"
        else:
            device_display = device_type.upper()
            device_icon = "🖥️"
        
        # Format timestamp
        report_date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate processing time distribution chart data
        chart_data_json = json.dumps({
            'processing_times': processing_times[:50],  # First 50 for chart
            'transform_times': transform_times[:50],
        })
        
        # Generate GPU utilization chart data
        gpu_chart_data = json.dumps({
            'utilization': {
                'min': gpu_metrics.get('gpu_utilization', {}).get('min', 0),
                'avg': gpu_metrics.get('gpu_utilization', {}).get('avg', 0),
                'max': gpu_metrics.get('gpu_utilization', {}).get('max', 0),
            },
            'memory': {
                'min': gpu_metrics.get('memory_percent', {}).get('min', 0),
                'avg': gpu_metrics.get('memory_percent', {}).get('avg', 0),
                'max': gpu_metrics.get('memory_percent', {}).get('max', 0),
            }
        })
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} {device_display} Performance Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 20px;
            padding: 3rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }}
        
        .device-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-size: 1.1rem;
            margin: 0.5rem;
            font-weight: 600;
        }}
        
        .timestamp {{
            color: #6c757d;
            margin-top: 1rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .metric-card h2 {{
            font-size: 1.3rem;
            color: #4a5568;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .metric-value {{
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .metric-label {{
            color: #718096;
            font-size: 1rem;
        }}
        
        .metric-subvalue {{
            font-size: 1.2rem;
            color: #2d3748;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .chart-title {{
            font-size: 1.5rem;
            color: #2d3748;
            margin-bottom: 1.5rem;
        }}
        
        .bar-chart {{
            display: flex;
            align-items: flex-end;
            height: 200px;
            gap: 2px;
            margin-bottom: 1rem;
        }}
        
        .bar {{
            flex: 1;
            min-width: 10px;
            background: linear-gradient(to top, #667eea, #764ba2);
            border-radius: 3px 3px 0 0;
            position: relative;
            cursor: pointer;
            transition: opacity 0.3s;
        }}
        
        .bar:hover {{
            opacity: 0.8;
        }}
        
        .bar-tooltip {{
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 0.3rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
        }}
        
        .bar:hover .bar-tooltip {{
            opacity: 1;
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1rem;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        
        .stats-table {{
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .stats-table h2 {{
            font-size: 1.5rem;
            color: #2d3748;
            margin-bottom: 1.5rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #f7fafc;
            color: #4a5568;
            font-weight: 600;
            text-align: left;
            padding: 1rem;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        td {{
            padding: 1rem;
            border-bottom: 1px solid #e2e8f0;
            color: #2d3748;
        }}
        
        tr:hover {{
            background: #f7fafc;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e2e8f0;
            border-radius: 15px;
            overflow: hidden;
            margin: 1rem 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .gpu-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .gpu-metric {{
            text-align: center;
            padding: 1rem;
            background: #f7fafc;
            border-radius: 10px;
        }}
        
        .gpu-metric-label {{
            font-size: 0.9rem;
            color: #718096;
            margin-bottom: 0.5rem;
        }}
        
        .gpu-metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #2d3748;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 3rem;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{device_icon} {model_name} Performance Report</h1>
            <div class="device-badge">{device_display}</div>
            <div class="timestamp">Report Generated: {report_date}</div>
        </div>
        
        <!-- Key Performance Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h2>⚡ Throughput</h2>
                <div class="metric-value">{perf_stats['throughput_docs_per_min']:.1f}</div>
                <div class="metric-label">Documents per minute</div>
                <div class="metric-subvalue">{perf_stats['throughput_docs_per_sec']:.2f} docs/sec</div>
            </div>
            
            <div class="metric-card">
                <h2>⏱️ Processing Time</h2>
                <div class="metric-value">{perf_stats['avg_processing_time']:.2f}s</div>
                <div class="metric-label">Average per document</div>
                <div class="metric-subvalue">Total: {perf_stats['total_processing_time']:.1f}s</div>
            </div>
            
            <div class="metric-card">
                <h2>📊 Success Rate</h2>
                <div class="metric-value">{(perf_stats['successful_documents'] / perf_stats['total_documents'] * 100):.1f}%</div>
                <div class="metric-label">{perf_stats['successful_documents']} of {perf_stats['total_documents']} documents</div>
                <div class="metric-subvalue">{perf_stats['failed_documents']} failed</div>
            </div>
        </div>
        
        <!-- GPU Utilization Metrics -->
        <div class="metric-card">
            <h2>🎮 GPU Resource Utilization</h2>
            <div class="gpu-metrics">
                <div class="gpu-metric">
                    <div class="gpu-metric-label">GPU Usage</div>
                    <div class="gpu-metric-value">{gpu_metrics.get('gpu_utilization', {}).get('avg', 0):.1f}%</div>
                </div>
                <div class="gpu-metric">
                    <div class="gpu-metric-label">Memory Usage</div>
                    <div class="gpu-metric-value">{gpu_metrics.get('memory_percent', {}).get('avg', 0):.1f}%</div>
                </div>
                <div class="gpu-metric">
                    <div class="gpu-metric-label">Memory Used</div>
                    <div class="gpu-metric-value">{gpu_metrics.get('memory_usage_mb', {}).get('avg', 0):.0f} MB</div>
                </div>
            </div>
            
            <div class="progress-bar">
                <div class="progress-fill" style="width: {gpu_metrics.get('gpu_utilization', {}).get('avg', 0)}%">
                    GPU: {gpu_metrics.get('gpu_utilization', {}).get('avg', 0):.1f}%
                </div>
            </div>
            
            <div class="progress-bar">
                <div class="progress-fill" style="width: {gpu_metrics.get('memory_percent', {}).get('avg', 0)}%">
                    Memory: {gpu_metrics.get('memory_percent', {}).get('avg', 0):.1f}%
                </div>
            </div>
        </div>
        
        <!-- Processing Time Distribution -->
        <div class="chart-container">
            <h2 class="chart-title">Processing Time Distribution</h2>
            <div class="bar-chart" id="timeChart"></div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: linear-gradient(to top, #667eea, #764ba2);"></div>
                    <span>Processing Time</span>
                </div>
            </div>
        </div>
        
        <!-- Detailed Statistics Table -->
        <div class="stats-table">
            <h2>Detailed Performance Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total Documents Processed</td>
                        <td>{perf_stats['total_documents']}</td>
                        <td>{perf_stats['successful_documents']} successful, {perf_stats['failed_documents']} failed</td>
                    </tr>
                    <tr>
                        <td>Average Processing Time</td>
                        <td>{perf_stats['avg_processing_time']:.3f} seconds</td>
                        <td>σ = {perf_stats['std_processing_time']:.3f}s</td>
                    </tr>
                    <tr>
                        <td>Min/Max Processing Time</td>
                        <td>{perf_stats['min_processing_time']:.3f}s / {perf_stats['max_processing_time']:.3f}s</td>
                        <td>Range: {(perf_stats['max_processing_time'] - perf_stats['min_processing_time']):.3f}s</td>
                    </tr>
                    <tr>
                        <td>Throughput</td>
                        <td>{perf_stats['throughput_docs_per_sec']:.3f} docs/sec</td>
                        <td>{perf_stats['throughput_docs_per_min']:.1f} docs/min</td>
                    </tr>
                    <tr>
                        <td>GPU Utilization</td>
                        <td>{gpu_metrics.get('gpu_utilization', {}).get('avg', 0):.1f}%</td>
                        <td>Min: {gpu_metrics.get('gpu_utilization', {}).get('min', 0):.1f}%, Max: {gpu_metrics.get('gpu_utilization', {}).get('max', 0):.1f}%</td>
                    </tr>
                    <tr>
                        <td>GPU Memory Usage</td>
                        <td>{gpu_metrics.get('memory_usage_mb', {}).get('avg', 0):.0f} MB</td>
                        <td>{gpu_metrics.get('memory_percent', {}).get('avg', 0):.1f}% of {gpu_metrics.get('memory_usage_mb', {}).get('total', 0):.0f} MB</td>
                    </tr>"""
        
        # Add temperature if available
        if 'temperature_celsius' in gpu_metrics:
            html += f"""
                    <tr>
                        <td>GPU Temperature</td>
                        <td>{gpu_metrics['temperature_celsius'].get('avg', 0):.1f}°C</td>
                        <td>Min: {gpu_metrics['temperature_celsius'].get('min', 0):.1f}°C, Max: {gpu_metrics['temperature_celsius'].get('max', 0):.1f}°C</td>
                    </tr>"""
        
        # Add power consumption if available
        if 'power_watts' in gpu_metrics:
            html += f"""
                    <tr>
                        <td>Power Consumption</td>
                        <td>{gpu_metrics['power_watts'].get('avg', 0):.1f} W</td>
                        <td>Total: {gpu_metrics['power_watts'].get('total_kwh', 0):.4f} kWh</td>
                    </tr>"""
        
        html += f"""
                    <tr>
                        <td>System CPU Usage</td>
                        <td>{system_metrics.get('cpu', {}).get('avg_percent', 0):.1f}%</td>
                        <td>Cores: {system_metrics.get('cpu', {}).get('cores', 0)}</td>
                    </tr>
                    <tr>
                        <td>System Memory Usage</td>
                        <td>{system_metrics.get('memory', {}).get('avg_percent', 0):.1f}%</td>
                        <td>{system_metrics.get('memory', {}).get('avg_used_gb', 0):.1f} GB of {system_metrics.get('memory', {}).get('total_gb', 0):.1f} GB</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Configuration Details -->
        <div class="stats-table">
            <h2>Configuration</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Model</td>
                        <td>{model_name}</td>
                    </tr>
                    <tr>
                        <td>Device Type</td>
                        <td>{device_type.upper()}</td>
                    </tr>
                    <tr>
                        <td>Batch Size</td>
                        <td>{config.get('batch_size', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Number of Workers</td>
                        <td>{config.get('num_workers', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Image Enhancement</td>
                        <td>{'Enabled' if config.get('quality_enhancement') else 'Disabled'}</td>
                    </tr>
                    <tr>
                        <td>Monitoring Duration</td>
                        <td>{gpu_metrics.get('monitoring_duration', 0):.1f} seconds</td>
                    </tr>
                    <tr>
                        <td>Samples Collected</td>
                        <td>{gpu_metrics.get('samples_collected', 0)}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by OCR Performance Analysis System</p>
            <p>{model_name} on {device_display}</p>
        </div>
    </div>
    
    <script>
        // Processing time chart data
        const chartData = {chart_data_json};
        
        // Create bar chart
        const chart = document.getElementById('timeChart');
        const maxTime = Math.max(...chartData.processing_times);
        
        chartData.processing_times.forEach((time, index) => {{
            const bar = document.createElement('div');
            bar.className = 'bar';
            bar.style.height = `${{(time / maxTime) * 100}}%`;
            
            const tooltip = document.createElement('div');
            tooltip.className = 'bar-tooltip';
            tooltip.textContent = `Doc ${{index + 1}}: ${{time.toFixed(2)}}s`;
            bar.appendChild(tooltip);
            
            chart.appendChild(bar);
        }});
    </script>
</body>
</html>"""
        
        return html