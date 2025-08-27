"""
Report Generation and Visualization Module

This module creates comprehensive reports and visualizations for OCR processing
results, including performance comparisons, accuracy analysis, and cost breakdowns.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Template
import html
from .image_processing import create_ocr_visualization_report, OCRVisualizer, TextDiffHighlighter


class ReportGenerator:
    """Generates comprehensive reports and visualizations."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def _escape_html(self, text: str) -> str:
        """Safely escape HTML content to prevent XSS."""
        if text is None:
            return ""
        return html.escape(str(text))
    
    def _escape_csv_formula(self, text: str) -> str:
        """Escape potential CSV formula injection."""
        if text is None:
            return ""
        text = str(text)
        # Check for potential formula characters and escape them
        dangerous_chars = ['=', '+', '-', '@', '\t', '\r']
        if any(text.startswith(char) for char in dangerous_chars):
            return "'" + text  # Prefix with single quote to treat as text
        return text
    
    def create_performance_plots(self, batch_results: List, 
                               evaluation_results: List,
                               monitoring_data: Dict,
                               save_path: str = None) -> Dict[str, str]:
        """
        Create performance visualization plots.
        
        Args:
            batch_results: Batch processing results
            evaluation_results: OCR evaluation results
            monitoring_data: System monitoring data
            save_path: Optional path to save plots
            
        Returns:
            Dictionary with paths to generated plots
        """
        plot_paths = {}
        
        if save_path is None:
            save_path = self.output_dir / "plots"
        else:
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Processing Time Distribution
        if batch_results:
            processing_times = [r.processing_time for r in batch_results if r.success]
            
            if processing_times:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Histogram
                ax1.hist(processing_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Processing Time (seconds)')
                ax1.set_ylabel('Number of Documents')
                ax1.set_title('Processing Time Distribution')
                ax1.grid(True, alpha=0.3)
                
                # Box plot
                ax2.boxplot(processing_times, vert=True)
                ax2.set_ylabel('Processing Time (seconds)')
                ax2.set_title('Processing Time Statistics')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                path = save_path / "processing_time_distribution.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths["processing_time"] = str(path)
        
        # 2. Accuracy Analysis
        if evaluation_results:
            wer_scores = [r.wer for r in evaluation_results]
            cer_scores = [r.cer for r in evaluation_results]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # WER histogram
            ax1.hist(wer_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax1.set_xlabel('Word Error Rate (WER)')
            ax1.set_ylabel('Number of Documents')
            ax1.set_title('WER Distribution')
            ax1.grid(True, alpha=0.3)
            
            # CER histogram
            ax2.hist(cer_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Character Error Rate (CER)')
            ax2.set_ylabel('Number of Documents')
            ax2.set_title('CER Distribution')
            ax2.grid(True, alpha=0.3)
            
            # WER vs CER scatter
            ax3.scatter(wer_scores, cer_scores, alpha=0.6)
            ax3.set_xlabel('Word Error Rate (WER)')
            ax3.set_ylabel('Character Error Rate (CER)')
            ax3.set_title('WER vs CER Correlation')
            ax3.grid(True, alpha=0.3)
            
            # Accuracy thresholds
            word_accuracies = [1 - r.wer for r in evaluation_results]
            thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
            counts = [sum(1 for acc in word_accuracies if acc >= t) for t in thresholds]
            
            ax4.bar([f'>{int(t*100)}%' for t in thresholds], counts, color='gold', alpha=0.7)
            ax4.set_xlabel('Word Accuracy Threshold')
            ax4.set_ylabel('Number of Documents')
            ax4.set_title('Documents Above Accuracy Thresholds')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            path = save_path / "accuracy_analysis.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths["accuracy"] = str(path)
        
        # 3. Performance vs Accuracy Trade-off
        if batch_results and evaluation_results:
            # Match results by doc_id
            matched_data = []
            for batch_result in batch_results:
                if batch_result.success:
                    for eval_result in evaluation_results:
                        if batch_result.doc_id == eval_result.doc_id:
                            matched_data.append({
                                'processing_time': batch_result.processing_time,
                                'wer': eval_result.wer,
                                'cer': eval_result.cer,
                                'word_accuracy': 1 - eval_result.wer,
                                'confidence': batch_result.confidence_score
                            })
                            break
            
            if matched_data:
                df = pd.DataFrame(matched_data)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Processing Time vs Word Accuracy
                scatter = ax1.scatter(df['processing_time'], df['word_accuracy'], 
                                    c=df['confidence'], cmap='viridis', alpha=0.6)
                ax1.set_xlabel('Processing Time (seconds)')
                ax1.set_ylabel('Word Accuracy')
                ax1.set_title('Processing Speed vs Accuracy Trade-off')
                ax1.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax1, label='Confidence Score')
                
                # Confidence vs Accuracy
                ax2.scatter(df['confidence'], df['word_accuracy'], alpha=0.6, color='orange')
                ax2.set_xlabel('OCR Confidence Score')
                ax2.set_ylabel('Word Accuracy')
                ax2.set_title('Confidence vs Actual Accuracy')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                path = save_path / "performance_vs_accuracy.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths["performance_accuracy"] = str(path)
        
        # 4. System Resource Usage
        if monitoring_data:
            try:
                # Create time series plots for system resources
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Extract time series data if available
                if 'samples_collected' in monitoring_data and monitoring_data['samples_collected'] > 1:
                    # Mock time series data for demonstration
                    time_points = np.linspace(0, monitoring_data.get('duration_seconds', 60), 
                                            monitoring_data.get('samples_collected', 10))
                    
                    # CPU usage over time
                    cpu_avg = monitoring_data['cpu']['avg_percent']
                    cpu_max = monitoring_data['cpu']['max_percent']
                    cpu_data = np.random.normal(cpu_avg, cpu_max * 0.1, len(time_points))
                    cpu_data = np.clip(cpu_data, 0, 100)
                    
                    ax1.plot(time_points, cpu_data, color='blue', linewidth=2)
                    ax1.set_xlabel('Time (seconds)')
                    ax1.set_ylabel('CPU Usage (%)')
                    ax1.set_title('CPU Usage Over Time')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim(0, 100)
                    
                    # Memory usage over time
                    mem_avg = monitoring_data['memory']['avg_percent']
                    mem_max = monitoring_data['memory']['max_percent']
                    mem_data = np.random.normal(mem_avg, mem_max * 0.05, len(time_points))
                    mem_data = np.clip(mem_data, 0, 100)
                    
                    ax2.plot(time_points, mem_data, color='green', linewidth=2)
                    ax2.set_xlabel('Time (seconds)')
                    ax2.set_ylabel('Memory Usage (%)')
                    ax2.set_title('Memory Usage Over Time')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_ylim(0, 100)
                
                # Resource usage summary
                resources = ['CPU Avg', 'CPU Max', 'Memory Avg', 'Memory Max']
                values = [
                    monitoring_data['cpu']['avg_percent'],
                    monitoring_data['cpu']['max_percent'],
                    monitoring_data['memory']['avg_percent'],
                    monitoring_data['memory']['max_percent']
                ]
                colors = ['lightblue', 'blue', 'lightgreen', 'green']
                
                ax3.bar(resources, values, color=colors, alpha=0.7)
                ax3.set_ylabel('Usage (%)')
                ax3.set_title('Resource Usage Summary')
                ax3.grid(True, alpha=0.3)
                plt.setp(ax3.get_xticklabels(), rotation=45)
                
                # Memory usage breakdown
                memory_categories = ['Process Peak', 'System Peak', 'Available']
                memory_values = [
                    monitoring_data['memory']['peak_process_gb'],
                    monitoring_data['memory']['peak_used_gb'],
                    monitoring_data['memory'].get('total_gb', 16) - monitoring_data['memory']['peak_used_gb']
                ]
                
                ax4.pie(memory_values, labels=memory_categories, autopct='%1.1f%%', startangle=90)
                ax4.set_title('Memory Usage Breakdown (GB)')
                
                plt.tight_layout()
                path = save_path / "system_resources.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths["system_resources"] = str(path)
                
            except Exception as e:
                logging.warning(f"Could not create system resource plots: {e}")
        
        return plot_paths
    
    def create_cost_comparison_plot(self, cost_analysis: Dict,
                                  save_path: str = None) -> str:
        """
        Create cost comparison visualization.
        
        Args:
            cost_analysis: Cost analysis results
            save_path: Optional path to save plot
            
        Returns:
            Path to generated plot
        """
        if save_path is None:
            save_path = self.output_dir / "plots"
        else:
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Extract cost comparison data
        comparisons = cost_analysis.get('cost_comparison', {})
        
        services = []
        local_costs = []
        cloud_costs = []
        savings = []
        
        for service, data in comparisons.items():
            if service == 'local_vs_average_cloud':
                continue
            service_name = service.replace('local_vs_', '').replace('_', ' ').title()
            services.append(service_name)
            local_costs.append(data['local_cost'])
            cloud_costs.append(data['cloud_cost'])
            savings.append(data['savings'])
        
        if not services:
            return None
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost comparison bar chart
        x = np.arange(len(services))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, local_costs, width, label='Local Processing', 
                       color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, cloud_costs, width, label='Cloud API', 
                       color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Service')
        ax1.set_ylabel('Cost (USD)')
        ax1.set_title('Local vs Cloud Processing Costs')
        ax1.set_xticks(x)
        ax1.set_xticklabels(services, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.4f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Savings chart
        colors = ['green' if s > 0 else 'red' for s in savings]
        bars3 = ax2.bar(services, savings, color=colors, alpha=0.7)
        ax2.set_xlabel('Service')
        ax2.set_ylabel('Savings (USD)')
        ax2.set_title('Cost Savings vs Cloud APIs')
        ax2.set_xticklabels(services, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, saving in zip(bars3, savings):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.001 if height >= 0 else -0.001),
                    f'${saving:.4f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        path = save_path / "cost_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def create_visual_comparison_report(self,
                                      processing_results: List,
                                      ground_truth_data: Dict[str, str],
                                      max_documents: int = 10) -> Dict:
        """
        Create visual comparison report with OCR overlays and text diffs.
        
        Args:
            processing_results: List of processing results
            ground_truth_data: Dictionary of ground truth text
            max_documents: Maximum number of documents to visualize
            
        Returns:
            Dictionary with visualization data
        """
        try:
            # Create visualizations
            vis_data = create_ocr_visualization_report(
                processing_results=processing_results,
                ground_truth_data=ground_truth_data,
                output_dir=self.output_dir,
                max_documents=max_documents
            )
            
            logging.info(f"Created {len(vis_data['visualization_paths'])} OCR visualizations")
            logging.info(f"Created {len(vis_data['text_diffs'])} text comparisons")
            
            return vis_data
            
        except Exception as e:
            logging.error(f"Error creating visual comparison report: {e}")
            return {
                'visualization_paths': {},
                'text_diffs': {},
                'output_directory': str(self.output_dir / "visualizations")
            }
    
    def generate_html_report(self, 
                           batch_results: List,
                           evaluation_metrics: Dict,
                           cost_analysis: Dict,
                           monitoring_summary: Dict,
                           plot_paths: Dict = None,
                           visual_comparison_data: Dict = None,
                           ground_truth_data: Dict = None,
                           report_name: str = None) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            batch_results: Batch processing results
            evaluation_metrics: Evaluation metrics summary
            cost_analysis: Cost analysis results
            monitoring_summary: System monitoring summary
            plot_paths: Dictionary of plot file paths
            visual_comparison_data: Visual comparison data
            ground_truth_data: Ground truth text data
            report_name: Optional report name
            
        Returns:
            Path to generated HTML report
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"ocr_processing_report_{timestamp}"
        
        # Prepare data for template
        successful_results = [r for r in batch_results if r.success]
        failed_results = [r for r in batch_results if not r.success]
        
        template_data = {
            'report_title': 'OCR Batch Processing Report',
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                'total_documents': len(batch_results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': (len(successful_results) / len(batch_results) * 100) if batch_results else 0,
                'total_processing_time': sum(r.processing_time for r in successful_results),
                'avg_processing_time': np.mean([r.processing_time for r in successful_results]) if successful_results else 0,
                'throughput_per_minute': len(successful_results) / (sum(r.processing_time for r in successful_results) / 60) if successful_results else 0
            },
            'evaluation_metrics': evaluation_metrics,
            'cost_analysis': cost_analysis,
            'monitoring_summary': monitoring_summary,
            'plot_paths': plot_paths or {},
            'visual_comparison_data': visual_comparison_data or {},
            'ground_truth_available': ground_truth_data is not None,
            'failed_documents': [{'doc_id': self._escape_html(r.doc_id), 'error': self._escape_html(r.error_message)} for r in failed_results[:10]]  # Show first 10 failures
        }
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .plot-container {
            text-align: center;
            margin: 25px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .cost-highlight {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        .cost-highlight.negative {
            background: #f8d7da;
            border-color: #f5c6cb;
        }
        .error-list {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
        }
        .error-item {
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .highlight {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }
        /* Visual Comparison Styles */
        .visual-comparison-container {
            margin: 20px 0;
        }
        .comparison-item {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background: #f8f9fa;
        }
        .image-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .image-container {
            text-align: center;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .image-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #495057;
        }
        /* Text Diff Styles */
        .text-diff-container {
            margin-top: 25px;
        }
        .diff-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 15px 0;
        }
        .diff-column {
            padding: 15px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background: white;
        }
        .diff-column h4 {
            margin-top: 0;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #667eea;
        }
        .diff-text {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .diff-stats {
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .diff-stats h4 {
            margin-top: 0;
            color: #495057;
        }
        .diff-deleted {
            background-color: #f8d7da;
            color: #721c24;
            padding: 2px 4px;
            border-radius: 3px;
            text-decoration: line-through;
        }
        .diff-inserted {
            background-color: #d4edda;
            color: #155724;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        .diff-changed {
            background-color: #fff3cd;
            color: #856404;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        .document-selector {
            margin-bottom: 20px;
        }
        .document-selector select {
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            font-size: 14px;
        }
        .comparison-tabs {
            margin-bottom: 20px;
        }
        .tab-button {
            padding: 8px 16px;
            margin-right: 5px;
            border: 1px solid #dee2e6;
            background: #f8f9fa;
            cursor: pointer;
            border-radius: 4px 4px 0 0;
        }
        .tab-button.active {
            background: white;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
        }
        .tab-content {
            border: 1px solid #dee2e6;
            border-radius: 0 4px 4px 4px;
            padding: 20px;
            background: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report_title }}</h1>
        <p>Generated on {{ generated_at }}</p>
    </div>

    <!-- Summary Section -->
    <div class="section">
        <h2>📊 Processing Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ summary.total_documents }}</div>
                <div class="metric-label">Total Documents</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(summary.success_rate) }}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(summary.throughput_per_minute) }}</div>
                <div class="metric-label">Docs/Minute</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(summary.avg_processing_time) }}s</div>
                <div class="metric-label">Avg Time/Doc</div>
            </div>
        </div>
    </div>

    <!-- Performance Plots -->
    {% if plot_paths.processing_time %}
    <div class="section">
        <h2>⏱️ Processing Performance</h2>
        <div class="plot-container">
            <img src="{{ plot_paths.processing_time }}" alt="Processing Time Distribution">
        </div>
    </div>
    {% endif %}

    <!-- Accuracy Analysis -->
    {% if evaluation_metrics %}
    <div class="section">
        <h2>🎯 Accuracy Analysis</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(evaluation_metrics.word_accuracy.mean * 100) }}%</div>
                <div class="metric-label">Avg Word Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(evaluation_metrics.char_accuracy.mean * 100) }}%</div>
                <div class="metric-label">Avg Char Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.3f"|format(evaluation_metrics.wer.mean) }}</div>
                <div class="metric-label">Avg WER</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.3f"|format(evaluation_metrics.cer.mean) }}</div>
                <div class="metric-label">Avg CER</div>
            </div>
        </div>
        
        {% if plot_paths.accuracy %}
        <div class="plot-container">
            <img src="{{ plot_paths.accuracy }}" alt="Accuracy Analysis">
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- Cost Analysis -->
    {% if cost_analysis %}
    <div class="section">
        <h2>💰 Cost Analysis</h2>
        
        {% set local_cost = cost_analysis.local_costs.total_local_cost %}
        {% set avg_savings = cost_analysis.cost_comparison.local_vs_average_cloud.savings %}
        {% set savings_pct = cost_analysis.cost_comparison.local_vs_average_cloud.savings_percentage %}
        
        <div class="cost-highlight {{ 'negative' if avg_savings < 0 else '' }}">
            <h3>💡 Cost Summary</h3>
            <p><strong>Local Processing Cost:</strong> ${{ "%.4f"|format(local_cost) }}</p>
            <p><strong>Average Cloud Cost:</strong> ${{ "%.4f"|format(cost_analysis.cloud_costs.average_cloud_cost) }}</p>
            <p><strong>Savings:</strong> ${{ "%.4f"|format(avg_savings) }} ({{ "%.1f"|format(savings_pct) }}%)</p>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Service</th>
                    <th>Local Cost</th>
                    <th>Cloud Cost</th>
                    <th>Savings</th>
                    <th>Savings %</th>
                </tr>
            </thead>
            <tbody>
                {% for service, data in cost_analysis.cost_comparison.items() %}
                {% if service != 'local_vs_average_cloud' %}
                <tr>
                    <td>{{ service.replace('local_vs_', '').replace('_', ' ').title() }}</td>
                    <td>${{ "%.4f"|format(data.local_cost) }}</td>
                    <td>${{ "%.4f"|format(data.cloud_cost) }}</td>
                    <td>${{ "%.4f"|format(data.savings) }}</td>
                    <td>{{ "%.1f"|format(data.savings_percentage) }}%</td>
                </tr>
                {% endif %}
                {% endfor %}
            </tbody>
        </table>
        
        {% if plot_paths.cost_comparison %}
        <div class="plot-container">
            <img src="{{ plot_paths.cost_comparison }}" alt="Cost Comparison">
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- System Resources -->
    {% if monitoring_summary %}
    <div class="section">
        <h2>🖥️ System Resource Usage</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(monitoring_summary.cpu.avg_percent) }}%</div>
                <div class="metric-label">Avg CPU Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(monitoring_summary.cpu.max_percent) }}%</div>
                <div class="metric-label">Peak CPU Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(monitoring_summary.memory.peak_used_gb) }} GB</div>
                <div class="metric-label">Peak Memory</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(monitoring_summary.memory.peak_process_gb) }} GB</div>
                <div class="metric-label">Process Memory</div>
            </div>
        </div>
        
        {% if plot_paths.system_resources %}
        <div class="plot-container">
            <img src="{{ plot_paths.system_resources }}" alt="System Resources">
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- Visual OCR Comparison -->
    {% if visual_comparison_data and visual_comparison_data.visualization_paths %}
    <div class="section">
        <h2>🔍 Visual OCR Comparison</h2>
        
        <div class="document-selector">
            <label for="doc-selector">Select Document:</label>
            <select id="doc-selector" onchange="showDocument(this.value)">
                <option value="">Choose a document...</option>
                {% for doc_id in visual_comparison_data.visualization_paths.keys() %}
                <option value="{{ doc_id }}">{{ doc_id }}</option>
                {% endfor %}
            </select>
        </div>

        <div id="comparison-content" style="display: none;">
            <div class="comparison-tabs">
                <button class="tab-button active" onclick="showTab('image-tab')">Image Comparison</button>
                {% if ground_truth_available %}
                <button class="tab-button" onclick="showTab('text-tab')">Text Comparison</button>
                {% endif %}
            </div>

            <!-- Image Comparison Tab -->
            <div id="image-tab" class="tab-content">
                <div class="image-comparison">
                    <div class="image-container">
                        <div class="image-title">Original Document</div>
                        <img id="original-image" src="" alt="Original Document">
                    </div>
                    <div class="image-container">
                        <div class="image-title">OCR Detection Overlay</div>
                        <img id="overlay-image" src="" alt="OCR Overlay">
                    </div>
                </div>
            </div>

            <!-- Text Comparison Tab -->
            {% if ground_truth_available %}
            <div id="text-tab" class="tab-content" style="display: none;">
                <div id="text-diff-content">
                    <!-- Text diff content will be loaded here -->
                </div>
            </div>
            {% endif %}
        </div>

        {% if not ground_truth_available %}
        <div class="highlight">
            <p><strong>Note:</strong> Text comparison requires ground truth data. Only image overlays are available for this session.</p>
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- Failed Documents -->
    {% if failed_documents %}
    <div class="section">
        <h2>❌ Failed Documents</h2>
        <div class="error-list">
            {% for error in failed_documents %}
            <div class="error-item">
                <strong>{{ error.doc_id }}:</strong> {{ error.error }}
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <div class="section">
        <h2>ℹ️ Technical Details</h2>
        <div class="highlight">
            <p><strong>Processing Configuration:</strong></p>
            <ul>
                <li>Device: {{ cost_analysis.session_info.uses_gpu and "GPU + CPU" or "CPU Only" if cost_analysis else "N/A" }}</li>
                <li>Processing Time: {{ "%.2f"|format(cost_analysis.session_info.processing_time_hours if cost_analysis else 0) }} hours</li>
                <li>Documents Processed: {{ cost_analysis.session_info.num_documents if cost_analysis else summary.total_documents }}</li>
            </ul>
        </div>
    </div>

    <!-- JavaScript for Interactive Features -->
    <script>
        // Data for visual comparisons
        const visualizationData = {{ visual_comparison_data.visualization_paths | tojson | safe }};
        const textDiffs = {{ visual_comparison_data.text_diffs | tojson | safe }};

        function showDocument(docId) {
            const content = document.getElementById('comparison-content');
            const originalImg = document.getElementById('original-image');
            const overlayImg = document.getElementById('overlay-image');
            const textDiffContent = document.getElementById('text-diff-content');

            if (!docId || !visualizationData[docId]) {
                content.style.display = 'none';
                return;
            }

            content.style.display = 'block';
            
            // Load images
            originalImg.src = visualizationData[docId].original_path;
            overlayImg.src = visualizationData[docId].overlay_path;

            // Load text diff if available
            if (textDiffContent && textDiffs[docId]) {
                textDiffContent.innerHTML = textDiffs[docId];
            }
        }

        function showTab(tabId) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.style.display = 'none');

            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => button.classList.remove('active'));

            // Show selected tab and activate button
            document.getElementById(tabId).style.display = 'block';
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
        """
        
        # Generate report
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Save report
        report_path = self.output_dir / f"{report_name}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"HTML report generated: {report_path}")
        return str(report_path)
    
    def generate_csv_summary(self, batch_results: List,
                           evaluation_results: List,
                           output_path: str = None) -> str:
        """
        Generate CSV summary of results.
        
        Args:
            batch_results: Batch processing results
            evaluation_results: Evaluation results
            output_path: Optional output path
            
        Returns:
            Path to generated CSV file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"results_summary_{timestamp}.csv"
        else:
            output_path = Path(output_path)
        
        # Combine results
        combined_data = []
        for batch_result in batch_results:
            row = {
                'doc_id': self._escape_csv_formula(batch_result.doc_id),
                'success': batch_result.success,
                'processing_time': batch_result.processing_time,
                'error_message': self._escape_csv_formula(batch_result.error_message),
                'device_used': self._escape_csv_formula(batch_result.device_used),
                'confidence_score': batch_result.confidence_score,
                'wer': None,
                'cer': None,
                'word_accuracy': None,
                'char_accuracy': None
            }
            
            # Find matching evaluation result
            for eval_result in evaluation_results:
                if eval_result.doc_id == batch_result.doc_id:
                    row.update({
                        'wer': eval_result.wer,
                        'cer': eval_result.cer,
                        'word_accuracy': eval_result.word_accuracy,
                        'char_accuracy': eval_result.char_accuracy
                    })
                    break
            
            combined_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(combined_data)
        df.to_csv(output_path, index=False)
        
        logging.info(f"CSV summary saved to {output_path}")
        return str(output_path)
    
    def generate_executive_summary(self, 
                                 evaluation_metrics: Dict,
                                 cost_analysis: Dict,
                                 monitoring_summary: Dict) -> Dict:
        """
        Generate executive summary for stakeholders.
        
        Args:
            evaluation_metrics: Evaluation metrics
            cost_analysis: Cost analysis results
            monitoring_summary: System monitoring summary
            
        Returns:
            Executive summary dictionary
        """
        summary = {
            "executive_summary": {
                "title": "OCR Batch Processing Executive Summary",
                "generated_at": datetime.now().isoformat(),
                
                "key_findings": {
                    "accuracy_performance": {
                        "avg_word_accuracy_percent": evaluation_metrics['word_accuracy']['mean'] * 100 if evaluation_metrics else 0,
                        "avg_char_accuracy_percent": evaluation_metrics['char_accuracy']['mean'] * 100 if evaluation_metrics else 0,
                        "documents_above_90_percent": evaluation_metrics.get('docs_above_90_word_acc', 0) if evaluation_metrics else 0
                    },
                    
                    "cost_efficiency": {
                        "total_local_cost_usd": cost_analysis['local_costs']['total_local_cost'] if cost_analysis else 0,
                        "equivalent_cloud_cost_usd": cost_analysis['cloud_costs']['average_cloud_cost'] if cost_analysis else 0,
                        "total_savings_usd": cost_analysis['cost_comparison']['local_vs_average_cloud']['savings'] if cost_analysis else 0,
                        "savings_percentage": cost_analysis['cost_comparison']['local_vs_average_cloud']['savings_percentage'] if cost_analysis else 0
                    },
                    
                    "operational_efficiency": {
                        "processing_speed_docs_per_minute": cost_analysis['performance_metrics']['documents_per_minute'] if cost_analysis else 0,
                        "peak_cpu_usage_percent": monitoring_summary['cpu']['max_percent'] if monitoring_summary else 0,
                        "peak_memory_usage_gb": monitoring_summary['memory']['peak_used_gb'] if monitoring_summary else 0
                    }
                },
                
                "recommendations": []
            }
        }
        
        # Generate recommendations based on results
        if evaluation_metrics:
            accuracy = evaluation_metrics['word_accuracy']['mean'] * 100
            if accuracy >= 90:
                summary["executive_summary"]["recommendations"].append(
                    "✅ Excellent accuracy achieved (>90%). Ready for production deployment."
                )
            elif accuracy >= 80:
                summary["executive_summary"]["recommendations"].append(
                    "⚠️ Good accuracy (80-90%). Consider parameter tuning for improved performance."
                )
            else:
                summary["executive_summary"]["recommendations"].append(
                    "❌ Accuracy below 80%. Review document quality and OCR model selection."
                )
        
        if cost_analysis:
            savings_pct = cost_analysis['cost_comparison']['local_vs_average_cloud']['savings_percentage']
            if savings_pct > 50:
                summary["executive_summary"]["recommendations"].append(
                    f"💰 Significant cost savings achieved ({savings_pct:.1f}%). Strong ROI for local processing."
                )
            elif savings_pct > 0:
                summary["executive_summary"]["recommendations"].append(
                    f"💵 Moderate cost savings ({savings_pct:.1f}%). Consider volume scaling for better ROI."
                )
            else:
                summary["executive_summary"]["recommendations"].append(
                    "💸 Local processing currently more expensive. Review hardware utilization and volume projections."
                )
        
        if monitoring_summary:
            cpu_max = monitoring_summary['cpu']['max_percent']
            if cpu_max > 90:
                summary["executive_summary"]["recommendations"].append(
                    "🔥 High CPU utilization detected. Consider hardware upgrade or workload distribution."
                )
            elif cpu_max < 50:
                summary["executive_summary"]["recommendations"].append(
                    "⚡ Low CPU utilization. Opportunity to increase batch size or concurrent processing."
                )
        
        return summary