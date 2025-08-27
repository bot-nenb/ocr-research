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
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from batch_processor.batch_processor import BatchProcessor, ProcessingConfig
from models.easyocr_model import EasyOCRModel
from utils.config_manager import ConfigManager
from utils.cost_analysis import CostAnalyzer
from utils.dataset_loader import FUNSDLoader
from utils.evaluation import EvaluationMetrics
from utils.monitoring import PerformanceTracker, SystemMonitor
from utils.reporting import ReportGenerator


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
    
    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(getattr(logging, logging_config.external_lib_level))
    logging.getLogger('PIL').setLevel(getattr(logging, logging_config.external_lib_level))
    logging.getLogger('easyocr').setLevel(getattr(logging, logging_config.external_lib_level))


@click.command()
@click.option('--device', default='auto', 
              help='Device to use: auto, cpu, gpu')
@click.option('--model', default='easyocr', 
              help='OCR model: easyocr, paddleocr')
@click.option('--num-documents', default=100, type=int,
              help='Number of documents to process')
@click.option('--num-workers', default=None, type=int,
              help='Number of worker processes/threads')
@click.option('--batch-size', default=10, type=int,
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
def main(device: str, model: str, num_documents: int, num_workers: Optional[int],
         batch_size: int, config: Optional[str], output_dir: str,
         dataset_dir: str, log_level: str, dry_run: bool,
         resume_from: Optional[str], generate_reports: bool, skip_download: bool):
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
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            num_documents=num_documents,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            generate_reports=generate_reports,
            model=model,
            log_level=log_level,
            skip_download=skip_download
        )
        
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
        else:
            click.echo("❌ Only EasyOCR is currently supported in this demo")
            return False
        
        click.echo(f"✅ {app_config.ocr_model.name} model initialized on {app_config.processing.device}")
        
        # Step 3: Batch Processing Setup
        click.echo("\n⚙️  Step 3: Batch Processing Setup")
        click.echo("-" * 50)
        
        batch_processor = BatchProcessor(app_config.processing, ocr_model)
        
        click.echo(f"✅ Batch processor configured:")
        click.echo(f"   Device: {app_config.processing.device}")
        click.echo(f"   Workers: {batch_processor.num_workers}")
        click.echo(f"   Batch size: {app_config.processing.batch_size}")
        
        # Step 4: System Monitoring
        click.echo("\n📊 Step 4: Starting System Monitoring")
        click.echo("-" * 50)
        
        system_monitor = SystemMonitor(
            log_dir=app_config.monitoring.log_dir, 
            sampling_interval=app_config.monitoring.sampling_interval
        )
        performance_tracker = PerformanceTracker()
        
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
        
        # Process batch
        processing_results = batch_processor.process_batch(
            documents_to_process, 
            ground_truth_data
        )
        
        # Record performance
        for result in processing_results:
            performance_tracker.record_document(
                result.processing_time,
                success=result.success
            )
        
        performance_tracker.stop_tracking()
        system_monitor.stop()
        
        # Get processing statistics
        batch_stats = batch_processor.get_statistics()
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
        uses_gpu = app_config.processing.device == 'gpu'
        
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
                html_report_path = report_generator.generate_html_report(
                    processing_results,
                    evaluation_metrics,
                    cost_analysis,
                    monitoring_summary,
                    plot_paths,
                    visual_comparison_data,
                    ground_truth_data
                )
            
            # Generate CSV summary if enabled
            csv_path = None
            if app_config.reporting.generate_csv:
                csv_path = report_generator.generate_csv_summary(
                    processing_results,
                    evaluation_results
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
        click.echo(f"   Success rate: {successful_docs/total_docs*100:.1f}%")
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