#!/usr/bin/env python3
"""
OCR Batch Processing Demo

High-volume batch OCR processing demonstration with comprehensive analysis.
Supports EasyOCR and PaddleOCR with CPU/GPU flexibility, performance monitoring,
accuracy evaluation, and cost analysis.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import PipelineCoordinator, PipelineConfig, ImageTransformConfig, OCRConfig
from models.easyocr_model import EasyOCRModel
from models.paddleocr_model import PaddleOCRModel
from utils.config_manager import ConfigManager, ProcessingConfig
from utils.cost_analysis import CostAnalyzer
from utils.dataset_loader import FUNSDLoader
from utils.evaluation import EvaluationMetrics
from utils.monitoring import PerformanceTracker, SystemMonitor
from utils.reporting import ReportGenerator
from utils.gpu_monitoring import GPUMonitor
from utils.gpu_report_generator import GPUPerformanceReportGenerator


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup comprehensive logging configuration."""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(logs_dir / log_file))
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(logs_dir / f"batch_processing_{timestamp}.log"))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('easyocr').setLevel(logging.WARNING)


def setup_logging_from_config(logging_config):
    """Setup comprehensive logging configuration from config."""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = []
    
    if logging_config.console_output:
        handlers.append(logging.StreamHandler())
    
    if logging_config.file_output:
        if logging_config.log_file:
            handlers.append(logging.FileHandler(logs_dir / logging_config.log_file))
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            handlers.append(logging.FileHandler(logs_dir / f"batch_processing_{timestamp}.log"))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, logging_config.level.upper()),
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Reduce noise from external libraries (but keep our model logs)
    logging.getLogger('matplotlib').setLevel(getattr(logging, logging_config.external_lib_level))
    logging.getLogger('PIL').setLevel(getattr(logging, logging_config.external_lib_level))
    logging.getLogger('easyocr').setLevel(getattr(logging, logging_config.external_lib_level))
    logging.getLogger('paddleocr').setLevel(getattr(logging, logging_config.external_lib_level))
    logging.getLogger('urllib3').setLevel(getattr(logging, logging_config.external_lib_level))
    
    # Keep our model logs at INFO level for detailed OCR logging
    logging.getLogger('models.easyocr_model').setLevel(logging.INFO)
    logging.getLogger('models.paddleocr_model').setLevel(logging.INFO)
    logging.getLogger('pipeline.ocr_reader').setLevel(logging.INFO)
    logging.getLogger('pipeline.pipeline_coordinator').setLevel(logging.INFO)


@click.command()
@click.option('--device', default=None,
              help='Device to use: auto, cpu, gpu, cuda, mps')
@click.option('--model', default=None,
              help='OCR model: easyocr, paddleocr')
@click.option('--num-documents', default=None, type=int,
              help='Number of documents to process')
@click.option('--num-workers', default=None, type=int,
              help='Number of worker processes/threads')
@click.option('--batch-size', default=None, type=int,
              help='Batch size for processing')
@click.option('--config', default=None,
              help='Path to configuration file')
@click.option('--output-dir', default='results',
              help='Output directory for results')
@click.option('--dataset-dir', default='data/funsd_subset',
              help='Dataset directory')
@click.option('--log-level', default='INFO',
              help='Logging level: DEBUG, INFO, WARNING, ERROR')
@click.option('--dry-run', is_flag=True,
              help='Perform dry run without actual processing')
@click.option('--resume-from', default=None,
              help='Resume from previous results file')
@click.option('--generate-reports', is_flag=True, default=True,
              help='Generate comprehensive reports')
@click.option('--skip-download', is_flag=True,
              help='Skip dataset download if already present')
@click.option('--quality-enhancement', is_flag=True, default=None,
              help='Enable image quality enhancement (noise reduction, CLAHE)')
@click.option('--normalize-images', is_flag=True, default=None,  
              help='Enable image normalization (histogram equalization)')
@click.option('--ocr-batch-size', default=None, type=int,
              help='OCR batch size (images per OCR call)')
def main(device: str, model: str, num_documents: int, num_workers: Optional[int],
         batch_size: int, config: Optional[str], output_dir: str,
         dataset_dir: str, log_level: str, dry_run: bool,
         resume_from: Optional[str], generate_reports: bool, skip_download: bool,
         quality_enhancement: Optional[bool], normalize_images: Optional[bool], 
         ocr_batch_size: Optional[int]):
    """
    OCR Batch Processing Demo
    
    Process documents using OCR models with comprehensive analysis including
    performance monitoring, accuracy evaluation, and cost analysis.
    """
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        if config:
            app_config = config_manager.load_config(config)
        else:
            app_config = config_manager.config
        
        # Override with command line arguments
        config_manager.override_from_cli(
            device=device if device else app_config.processing.device,
            num_workers=num_workers if num_workers is not None else app_config.processing.num_workers,
            batch_size=batch_size if batch_size is not None else app_config.processing.batch_size,
            num_documents=num_documents if num_documents is not None else app_config.dataset.num_documents,
            dataset_dir=dataset_dir if dataset_dir is not None else app_config.dataset.data_dir,
            output_dir=output_dir if output_dir is not None else app_config.reporting.output_dir,
            generate_reports=generate_reports if generate_reports is not None else app_config.reporting.generate_reports,
            model=model if model else app_config.ocr_model.name,
            log_level=log_level if log_level else app_config.logging.level,
            skip_download=skip_download if skip_download else app_config.dataset.skip_download,
            # Pipeline-specific parameters
            quality_enhancement=quality_enhancement,
            normalize_images=normalize_images,
            ocr_batch_size=ocr_batch_size
        )
        
        # Create model-specific output directory
        base_output_dir = app_config.reporting.output_dir
        model_name = app_config.ocr_model.name.lower()
        app_config.reporting.output_dir = f"{base_output_dir}/{model_name}"
        
        # Validate configuration
        validation_issues = config_manager.validate_config()
        if validation_issues:
            click.echo("⚠️  Configuration validation issues found:")
            for issue in validation_issues:
                click.echo(f"  - {issue}")
            if not click.confirm("Continue with potentially invalid configuration?"):
                return False
        
        # Setup logging with config
        setup_logging_from_config(app_config.logging)
        logging.info("=" * 80)
        logging.info("OCR BATCH PROCESSING DEMO STARTED")
        logging.info("=" * 80)
        
        # Print configuration summary
        config_manager.print_config_summary()
        
        # Create output directory
        output_path = Path(app_config.reporting.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Dataset Preparation
        click.echo("\n🗂️  Step 1: Dataset Preparation")
        click.echo("-" * 50)
        
        dataset_loader = FUNSDLoader(data_dir=app_config.dataset.data_dir)
        
        if not app_config.dataset.skip_download:
            if not dataset_loader.download_dataset():
                click.echo("❌ Failed to download dataset")
                return False
        
        # Check if subset exists
        if not (dataset_loader.subset_path / 'images').exists():
            click.echo(f"Creating subset of {app_config.dataset.num_documents} documents...")
            dataset_loader.select_subset(num_documents=app_config.dataset.num_documents)
        
        # Convert to standardized format
        standardized_path = dataset_loader.processed_path / 'standardized_annotations.json'
        if not standardized_path.exists():
            click.echo("Converting annotations to standard format...")
            standardized_data = dataset_loader.convert_to_standard_format()
        else:
            with open(standardized_path, 'r') as f:
                standardized_data = json.load(f)
        
        # Validate dataset
        is_valid, issues = dataset_loader.validate_data()
        if not is_valid:
            click.echo("⚠️  Dataset validation issues found:")
            for issue in issues[:5]:
                click.echo(f"  - {issue}")
        
        # Get dataset statistics
        stats = dataset_loader.get_dataset_stats()
        click.echo(f"✅ Dataset ready: {stats['total_documents']} documents")
        click.echo(f"   Average words per document: {stats['avg_words_per_doc']:.1f}")
        click.echo(f"   Average lines per document: {stats['avg_lines_per_doc']:.1f}")
        
        if dry_run:
            click.echo("\n🏃 Dry run mode - stopping here")
            return True
        
        # Step 2: Model Initialization
        click.echo("\n🤖 Step 2: OCR Model Initialization")
        click.echo("-" * 50)
        
        if app_config.ocr_model.name.lower() == 'easyocr':
            ocr_model = EasyOCRModel(
                languages=app_config.ocr_model.languages, 
                device=app_config.processing.device
            )
        elif app_config.ocr_model.name.lower() == 'paddleocr':
            ocr_model = PaddleOCRModel(
                languages=app_config.ocr_model.languages,
                device=app_config.processing.device
            )
        else:
            click.echo(f"❌ Unsupported OCR model: {app_config.ocr_model.name}")
            click.echo("   Supported models: easyocr, paddleocr")
            return False
        
        click.echo(f"✅ {app_config.ocr_model.name} model initialized on {app_config.processing.device}")
        
        # Step 3: Pipeline Setup
        click.echo("\n⚙️  Step 3: Pipeline Setup")
        click.echo("-" * 50)
        
        # Get pipeline configurations from the config manager
        transform_config, ocr_config, pipeline_config = config_manager.get_pipeline_configs()
        
        click.echo(f"✅ Pipeline configured from config file:")
        click.echo(f"   OCR Device: {ocr_config.device}")
        click.echo(f"   Transform Workers: {transform_config.num_workers}")
        click.echo(f"   Transform Batch Size: {transform_config.batch_size}")
        click.echo(f"   OCR Batch Size: {ocr_config.batch_size}")
        click.echo(f"   Quality Enhancement: {transform_config.quality_enhancement}")
        click.echo(f"   Image Normalization: {transform_config.normalize}")
        click.echo(f"   Max Image Size: {transform_config.max_image_size}")
        click.echo(f"   Continuous Processing: {pipeline_config.enable_continuous_processing}")
        
        # Step 4: System Monitoring
        click.echo("\n📊 Step 4: Starting System Monitoring")
        click.echo("-" * 50)
        
        system_monitor = SystemMonitor(
            log_dir=app_config.monitoring.log_dir, 
            sampling_interval=app_config.monitoring.sampling_interval
        )
        performance_tracker = PerformanceTracker()
        
        # Initialize GPU monitoring for GPU/MPS devices
        gpu_monitor = None
        if app_config.processing.device in ['gpu', 'cuda', 'mps']:
            try:
                gpu_monitor = GPUMonitor(device_type=app_config.processing.device)
                gpu_monitor.start_monitoring()
                click.echo(f"✅ GPU monitoring active for {app_config.processing.device}")
            except Exception as e:
                click.echo(f"⚠️  GPU monitoring failed: {e}")
                gpu_monitor = None
        
        system_monitor.start()
        performance_tracker.start_tracking()
        
        click.echo("✅ System monitoring active")
        
        # Step 5: Document Processing
        click.echo("\n🔄 Step 5: Processing Documents")
        click.echo("-" * 50)
        
        # Prepare documents for processing
        documents_to_process = []
        ground_truth_data = {}
        
        doc_count = 0
        for doc_id, doc_data in standardized_data.items():
            if doc_count >= app_config.dataset.num_documents:
                break
                
            try:
                image, gt_data, image_path = dataset_loader.prepare_for_ocr(doc_id)
                documents_to_process.append((doc_id, image, image_path))
                ground_truth_data[doc_id] = gt_data['text']
                doc_count += 1
            except Exception as e:
                logging.warning(f"Could not prepare document {doc_id}: {e}")
                continue
        
        click.echo(f"Processing {len(documents_to_process)} documents...")
        
        # Convert documents to pipeline format (image_path, doc_id)
        pipeline_inputs = []
        for doc_id, image, image_path in documents_to_process:
            if image_path:  # Use file path if available
                pipeline_inputs.append((image_path, doc_id))
            else:
                # Save temporary image file for cases where only array is available
                import tempfile
                import cv2
                temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                os.close(temp_fd)
                cv2.imwrite(temp_path, image)
                pipeline_inputs.append((temp_path, doc_id))
        
        # Process with pipeline
        with PipelineCoordinator(pipeline_config, ocr_model) as coordinator:
            processing_results = coordinator.process_images(pipeline_inputs)
        
        # Record performance and convert results
        successful_results = []
        failed_results = []
        
        for result in processing_results:
            # Convert OCRResult to format expected by rest of code
            processing_time = result.processing_time + result.transform_time
            performance_tracker.record_document(
                processing_time,
                success=result.success
            )
            
            if result.success:
                successful_results.append(result)
            else:
                failed_results.append(result)
            
            # Collect GPU metrics sample during processing
            if gpu_monitor:
                gpu_monitor.collect_sample()
        
        performance_tracker.stop_tracking()
        system_monitor.stop()
        
        # Stop GPU monitoring and get metrics
        gpu_metrics = None
        if gpu_monitor:
            gpu_monitor.stop_monitoring()
            gpu_metrics = gpu_monitor.get_summary()
            click.echo(f"✅ GPU monitoring completed: {gpu_metrics.get('samples_collected', 0)} samples")
        
        # Calculate statistics (simplified from BatchProcessor)
        total_results = len(processing_results)
        successful_count = len(successful_results)
        success_rate = successful_count / total_results if total_results > 0 else 0
        
        if successful_results:
            processing_times = [r.processing_time + r.transform_time for r in successful_results]
            total_time = sum(processing_times)
            docs_per_second = successful_count / total_time if total_time > 0 else 0
        else:
            docs_per_second = 0
        
        batch_stats = {
            'success_rate': success_rate,
            'docs_per_second': docs_per_second,
            'total_processed': total_results,
            'successful': successful_count,
            'failed': len(failed_results)
        }
        
        performance_metrics = performance_tracker.get_metrics()
        monitoring_summary = system_monitor.get_summary()
        
        click.echo(f"✅ Processing completed:")
        click.echo(f"   Success rate: {batch_stats.get('success_rate', 0)*100:.1f}%")
        click.echo(f"   Throughput: {batch_stats.get('docs_per_second', 0)*60:.1f} docs/minute")
        
        # Step 6: Accuracy Evaluation
        click.echo("\n🎯 Step 6: Accuracy Evaluation")
        click.echo("-" * 50)
        
        evaluator = EvaluationMetrics(
            normalize_text=app_config.evaluation.normalize_text,
            lowercase=app_config.evaluation.lowercase,
            remove_punctuation=app_config.evaluation.remove_punctuation
        )
        
        # Prepare evaluation data
        eval_documents = []
        confidences = []
        
        for result in processing_results:
            if result.success and result.doc_id in ground_truth_data:
                eval_documents.append((
                    result.doc_id,
                    ground_truth_data[result.doc_id],
                    result.ocr_text
                ))
                confidences.append(result.confidence_score)
        
        if eval_documents:
            evaluation_results = evaluator.evaluate_batch(eval_documents, confidences)
            evaluation_metrics = evaluator.get_aggregate_metrics()
            
            click.echo(f"✅ Accuracy evaluation completed:")
            click.echo(f"   Average word accuracy: {evaluation_metrics['word_accuracy']['mean']*100:.1f}%")
            click.echo(f"   Average character accuracy: {evaluation_metrics['char_accuracy']['mean']*100:.1f}%")
            click.echo(f"   Documents >90% accuracy: {evaluation_metrics.get('docs_above_90_word_acc', 0)}")
        else:
            evaluation_results = []
            evaluation_metrics = {}
            click.echo("⚠️  No successful results for evaluation")
        
        # Step 7: Cost Analysis
        click.echo("\n💰 Step 7: Cost Analysis")
        click.echo("-" * 50)
        
        cost_analyzer = CostAnalyzer(
            hardware_costs=app_config.cost_analysis_hardware,
            power_consumption=app_config.cost_analysis_power,
            cloud_costs=app_config.cost_analysis_cloud
        )
        
        total_processing_time = sum(r.processing_time for r in processing_results if r.success)
        avg_cpu_utilization = monitoring_summary.get('cpu', {}).get('avg_percent', 0) / 100.0
        uses_gpu = app_config.processing.device in ['gpu', 'cuda', 'mps']
        
        cost_analysis = cost_analyzer.analyze_processing_session(
            processing_time_seconds=total_processing_time,
            num_documents=len([r for r in processing_results if r.success]),
            cpu_utilization=avg_cpu_utilization,
            uses_gpu=uses_gpu,
            session_name=f"Demo_{len(documents_to_process)}_docs"
        )
        
        cost_analyzer.print_analysis_summary(cost_analysis)
        
        # Step 8: Report Generation
        if app_config.reporting.generate_html or app_config.reporting.generate_csv or app_config.reporting.generate_plots:
            click.echo("\n📋 Step 8: Report Generation")
            click.echo("-" * 50)
            
            report_generator = ReportGenerator(output_dir=output_path / "reports")
            
            # Generate plots if enabled
            plot_paths = {}
            if app_config.reporting.generate_plots:
                plot_paths = report_generator.create_performance_plots(
                    processing_results, 
                    evaluation_results, 
                    monitoring_summary
                )
            
            # Generate visual comparison report
            visual_comparison_data = report_generator.create_visual_comparison_report(
                processing_results=processing_results,
                ground_truth_data=ground_truth_data,
                max_documents=10
            )
            
            # Generate cost comparison plot
            if cost_analysis:
                cost_plot = report_generator.create_cost_comparison_plot(cost_analysis)
                if cost_plot:
                    plot_paths['cost_comparison'] = cost_plot
            
            # Generate HTML report if enabled
            html_report_path = None
            if app_config.reporting.generate_html:
                # Add model name to the report
                html_report_path = report_generator.generate_html_report(
                    processing_results,
                    evaluation_metrics,
                    cost_analysis,
                    monitoring_summary,
                    plot_paths,
                    visual_comparison_data,
                    ground_truth_data,
                    model_name=app_config.ocr_model.name
                )
            
            # Generate GPU performance report for GPU/MPS devices
            gpu_performance_report_path = None
            if gpu_metrics and app_config.processing.device in ['gpu', 'cuda', 'mps']:
                try:
                    gpu_report_generator = GPUPerformanceReportGenerator(output_dir=output_path / "reports")
                    
                    # Convert processing results to format expected by GPU report generator
                    gpu_processing_results = [
                        {
                            'success': r.success,
                            'processing_time': r.processing_time,
                            'transform_time': r.transform_time,
                            'doc_id': r.doc_id
                        }
                        for r in processing_results
                    ]
                    
                    # Configuration dict for GPU report
                    gpu_config = {
                        'batch_size': app_config.processing.batch_size,
                        'num_workers': app_config.processing.num_workers,
                        'quality_enhancement': transform_config.quality_enhancement,
                        'device': app_config.processing.device
                    }
                    
                    gpu_performance_report_path = gpu_report_generator.generate_performance_report(
                        processing_results=gpu_processing_results,
                        gpu_metrics=gpu_metrics,
                        system_metrics=monitoring_summary,
                        device_type=app_config.processing.device,
                        model_name=app_config.ocr_model.name,
                        config=gpu_config
                    )
                    
                    click.echo(f"✅ GPU performance report generated: {gpu_performance_report_path.name}")
                    
                except Exception as e:
                    click.echo(f"⚠️  GPU performance report failed: {e}")
                    logging.warning(f"GPU performance report generation failed: {e}")
            
            # Generate CSV summary if enabled
            csv_path = None
            if app_config.reporting.generate_csv:
                csv_path = report_generator.generate_csv_summary(
                    processing_results,
                    evaluation_results,
                    model_name=app_config.ocr_model.name
                )
            
            # Generate executive summary if enabled
            exec_summary = None
            exec_summary_path = None
            if app_config.reporting.generate_executive_summary:
                exec_summary = report_generator.generate_executive_summary(
                    evaluation_metrics,
                    cost_analysis,
                    monitoring_summary
                )
            
                # Save executive summary
                exec_summary_path = output_path / "executive_summary.json"
                with open(exec_summary_path, 'w') as f:
                    json.dump(exec_summary, f, indent=2, default=str)
            
            click.echo(f"✅ Reports generated:")
            if html_report_path:
                click.echo(f"   HTML Report: {html_report_path}")
                if visual_comparison_data and visual_comparison_data['visualization_paths']:
                    click.echo(f"   📸 Visual Comparison: {len(visual_comparison_data['visualization_paths'])} documents with OCR overlays")
            if csv_path:
                click.echo(f"   CSV Summary: {csv_path}")
            if exec_summary_path:
                click.echo(f"   Executive Summary: {exec_summary_path}")
            
            # Save complete results
            complete_results = {
                'processing_results': [
                    {
                        'doc_id': r.doc_id,
                        'success': r.success,
                        'processing_time': r.processing_time,
                        'ocr_text': r.ocr_text,
                        'error_message': r.error_message,
                        'confidence_score': r.confidence_score
                    } for r in processing_results
                ],
                'batch_statistics': batch_stats,
                'evaluation_metrics': evaluation_metrics,
                'cost_analysis': cost_analysis,
                'performance_metrics': performance_metrics,
                'monitoring_summary': monitoring_summary,
                'configuration': {
                    'device': app_config.processing.device,
                    'model': app_config.ocr_model.name,
                    'num_documents': app_config.dataset.num_documents,
                    'batch_size': app_config.processing.batch_size,
                    'num_workers': app_config.processing.num_workers
                }
            }
            
            results_path = output_path / "complete_results.json"
            with open(results_path, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            click.echo(f"   Complete Results: {results_path}")
        
        # Step 9: Summary and Recommendations
        click.echo("\n📈 Step 9: Summary and Recommendations")
        click.echo("-" * 50)
        
        # Print key findings
        successful_docs = len([r for r in processing_results if r.success])
        total_docs = len(processing_results)
        
        click.echo(f"🎯 Processing Summary:")
        click.echo(f"   Documents processed: {successful_docs}/{total_docs}")
        if total_docs > 0:
            click.echo(f"   Success rate: {successful_docs/total_docs*100:.1f}%")
        else:
            click.echo(f"   Success rate: 0.0%")
        click.echo(f"   Processing speed: {batch_stats.get('docs_per_second', 0)*60:.1f} docs/minute")
        
        if evaluation_metrics:
            avg_accuracy = evaluation_metrics['word_accuracy']['mean'] * 100
            click.echo(f"\n🎯 Accuracy Summary:")
            click.echo(f"   Average word accuracy: {avg_accuracy:.1f}%")
            
            if avg_accuracy >= 90:
                click.echo("   ✅ Excellent accuracy achieved!")
            elif avg_accuracy >= 80:
                click.echo("   ⚠️  Good accuracy - consider parameter tuning")
            else:
                click.echo("   ❌ Accuracy below 80% - review setup")
        
        if cost_analysis:
            savings = cost_analysis['cost_comparison']['local_vs_average_cloud']['savings']
            savings_pct = cost_analysis['cost_comparison']['local_vs_average_cloud']['savings_percentage']
            
            click.echo(f"\n💰 Cost Summary:")
            click.echo(f"   Savings vs cloud: ${savings:.4f} ({savings_pct:.1f}%)")
            
            if savings_pct > 50:
                click.echo("   💰 Significant cost savings achieved!")
            elif savings_pct > 0:
                click.echo("   💵 Moderate savings - consider scaling up")
            else:
                click.echo("   💸 Higher costs than cloud - review efficiency")
        
        # Step 10: Copy Results to GitHub Pages (docs directory)
        click.echo("\n📋 Step 10: Updating GitHub Pages")
        click.echo("-" * 50)
        
        try:
            # Copy results to docs directory for GitHub Pages
            import shutil
            
            # Find repo root by looking for docs directory
            current_dir = Path(__file__).resolve().parent
            repo_root = None
            
            # Walk up directories to find the one containing docs/
            for _ in range(10):  # Safety limit
                if (current_dir / "docs").exists():
                    repo_root = current_dir
                    break
                parent = current_dir.parent
                if parent == current_dir:  # Reached filesystem root
                    break
                current_dir = parent
            
            if repo_root is None:
                raise RuntimeError("Could not find repository root with docs/ directory")
                
            docs_dir = repo_root / "docs"
            docs_task1_dir = docs_dir / "demos" / "task-1"
            
            # Ensure docs directories exist
            docs_task1_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy main results files with model-specific naming
            if results_path.exists():
                shutil.copy2(results_path, docs_dir / f"{model_name}_complete_results.json")
                click.echo(f"   ✅ Complete results copied to docs/{model_name}_complete_results.json")
                
            # Copy executive summary if it exists
            exec_summary_path = output_path / "executive_summary.json"
            if exec_summary_path.exists():
                shutil.copy2(exec_summary_path, docs_dir / f"{model_name}_executive_summary.json")
                click.echo(f"   ✅ Executive summary copied to docs/{model_name}_executive_summary.json")
            
            # Copy reports directory contents
            reports_dir = output_path / "reports"
            if reports_dir.exists():
                # Copy all HTML reports and fix paths
                for html_file in reports_dir.glob("*.html"):
                    # Read HTML content
                    with open(html_file, 'r') as f:
                        html_content = f.read()
                    
                    # Fix plot paths for GitHub Pages (handle model-specific subdirectories)
                    html_content = html_content.replace(f'src="results/{model_name}/reports/plots/', f'src="{model_name}_plots/')
                    html_content = html_content.replace(f'src="results/{model_name}/reports/visualizations/', f'src="{model_name}_visualizations/')
                    html_content = html_content.replace('src="results/reports/plots/', f'src="{model_name}_plots/')  # Fallback for old format
                    html_content = html_content.replace('src="results/reports/visualizations/', f'src="{model_name}_visualizations/')  # Fallback for old format
                    
                    # Fix JavaScript visualization data paths
                    html_content = html_content.replace('"original_path": "data/funsd_subset/subset_100/images/', '"original_path": "original_images/')
                    html_content = html_content.replace(f'"overlay_path": "results/{model_name}/reports/visualizations/', f'"overlay_path": "{model_name}_visualizations/')
                    html_content = html_content.replace('"overlay_path": "results/reports/visualizations/', f'"overlay_path": "{model_name}_visualizations/')  # Fallback for old format
                    
                    # Write fixed content to docs directory
                    dest_file = docs_task1_dir / html_file.name
                    with open(dest_file, 'w') as f:
                        f.write(html_content)
                    
                    # Log special handling for GPU performance reports
                    if "performance" in html_file.name and app_config.processing.device in ['gpu', 'cuda', 'mps']:
                        click.echo(f"   🎮 GPU performance report copied: {html_file.name}")
                    else:
                        click.echo(f"   ✅ Report copied with fixed paths: {html_file.name}")
                
                # Copy all CSV summaries  
                for csv_file in reports_dir.glob("*.csv"):
                    shutil.copy2(csv_file, docs_task1_dir)
                    click.echo(f"   ✅ CSV summary copied: {csv_file.name}")
                
                # Copy plots directory to model-specific subdirectory
                plots_src = reports_dir / "plots"
                plots_dst = docs_task1_dir / f"{model_name}_plots"
                if plots_src.exists():
                    if plots_dst.exists():
                        shutil.rmtree(plots_dst)
                    shutil.copytree(plots_src, plots_dst)
                    click.echo(f"   ✅ Plots copied to docs/demos/task-1/{model_name}_plots/")
                
                # Copy visualizations directory to model-specific subdirectory
                viz_src = reports_dir / "visualizations"
                viz_dst = docs_task1_dir / f"{model_name}_visualizations"
                if viz_src.exists():
                    if viz_dst.exists():
                        shutil.rmtree(viz_dst)
                    shutil.copytree(viz_src, viz_dst)
                    click.echo(f"   ✅ Visualizations copied to docs/demos/task-1/{model_name}_visualizations/")
                    
                    # Copy corresponding original images to original_images directory
                    original_imgs_dst = docs_task1_dir / "original_images"
                    original_imgs_dst.mkdir(exist_ok=True)
                    
                    # Clear existing original images to ensure sync
                    for existing_img in original_imgs_dst.glob("*.png"):
                        existing_img.unlink()
                    
                    # Extract image names from visualization files and copy original images
                    for viz_file in viz_src.glob("*_ocr_overlay.png"):
                        # Extract original image name (remove _ocr_overlay suffix)
                        original_name = viz_file.name.replace("_ocr_overlay.png", ".png")
                        
                        # Find the original image in dataset
                        dataset_root = Path(__file__).parent / "data"
                        original_img_path = None
                        
                        # Search for the original image in dataset
                        for img_path in dataset_root.rglob(original_name):
                            if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                original_img_path = img_path
                                break
                        
                        if original_img_path:
                            dest_path = original_imgs_dst / original_name
                            shutil.copy2(original_img_path, dest_path)
                    
                    original_count = len(list(original_imgs_dst.glob("*.png")))
                    viz_count = len(list(viz_src.glob("*_ocr_overlay.png")))
                    click.echo(f"   ✅ Original images synchronized: {original_count}/{viz_count} images copied to docs/demos/task-1/original_images/")
                
                # Update the main task-1 index.html with latest report link
                index_path = docs_task1_dir / "index.html"
                if index_path.exists():
                    # Find the most recent HTML report
                    html_reports = list(docs_task1_dir.glob("ocr_processing_report_*.html"))
                    if html_reports:
                        latest_report = max(html_reports, key=lambda p: p.stat().st_mtime)
                        click.echo(f"   ✅ Latest report identified: {latest_report.name}")
                        
                        # Update index.html to point to latest report (basic replacement)
                        with open(index_path, 'r') as f:
                            content = f.read()
                        
                        # Replace old report references with new one
                        import re
                        content = re.sub(
                            r'ocr_processing_report_\d{8}_\d{6}\.html',
                            latest_report.name,
                            content
                        )
                        
                        with open(index_path, 'w') as f:
                            f.write(content)
                        click.echo(f"   ✅ Task-1 index.html updated with latest report link")
                        
                        # Also update main home page index.html
                        main_index_path = docs_dir / "index.html"
                        if main_index_path.exists():
                            with open(main_index_path, 'r') as f:
                                main_content = f.read()
                            
                            # Replace latest report link in home page
                            main_content = re.sub(
                                r'demos/task-1/ocr_processing_report_\d{8}_\d{6}\.html',
                                f'demos/task-1/{latest_report.name}',
                                main_content
                            )
                            
                            with open(main_index_path, 'w') as f:
                                f.write(main_content)
                            click.echo(f"   ✅ Home page index.html updated with latest report link")
            
            click.echo(f"🌐 GitHub Pages updated! Changes will be live at:")
            click.echo(f"   https://bot-nenb.github.io/ocr-research/demos/task-1/")
            
        except Exception as e:
            click.echo(f"⚠️  Warning: Failed to update GitHub Pages: {e}")
            logging.warning(f"Failed to copy results to docs directory: {e}")
        
        click.echo(f"\n🏁 Demo completed successfully!")
        click.echo(f"   Results saved to: {output_path}")
        
        logging.info("=" * 80)
        logging.info("OCR BATCH PROCESSING DEMO COMPLETED")
        logging.info("=" * 80)
        
        return True
        
    except Exception as e:
        logging.error(f"Demo failed with error: {e}", exc_info=True)
        click.echo(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main(standalone_mode=False)
    sys.exit(0 if success else 1)