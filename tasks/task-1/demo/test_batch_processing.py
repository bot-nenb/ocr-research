#!/usr/bin/env python3
"""
Test script for batch processing, monitoring, and evaluation components.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.batch_processor.batch_processor import BatchProcessor, ProcessingConfig
from src.models.easyocr_model import EasyOCRModel
from src.utils.dataset_loader import FUNSDLoader
from src.utils.evaluation import EvaluationMetrics
from src.utils.monitoring import SystemMonitor, PerformanceTracker


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def test_integrated_pipeline():
    """Test the integrated OCR pipeline with batch processing, monitoring, and evaluation."""
    
    print("=" * 80)
    print("INTEGRATED OCR PIPELINE TEST")
    print("=" * 80)
    
    # Step 1: Load dataset
    print("\n1. Loading FUNSD dataset...")
    loader = FUNSDLoader(data_dir="data/funsd_subset")
    
    # Check if dataset exists, if not download it
    if not (loader.subset_path / 'images').exists():
        print("Dataset not found. Please run test_dataset_loader.py first to download.")
        return False
    
    # Load standardized annotations
    standardized_path = loader.processed_path / 'standardized_annotations.json'
    if not standardized_path.exists():
        print("Converting annotations to standard format...")
        standardized = loader.convert_to_standard_format()
    else:
        import json
        with open(standardized_path, 'r') as f:
            standardized = json.load(f)
    
    print(f"Loaded {len(standardized)} documents")
    
    # Step 2: Initialize OCR model
    print("\n2. Initializing OCR model...")
    try:
        ocr_model = EasyOCRModel(languages=['en'], device="cpu")
        print("EasyOCR model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize OCR model: {e}")
        return False
    
    # Step 3: Configure batch processing
    print("\n3. Configuring batch processor...")
    config = ProcessingConfig(
        device="cpu",  # Force CPU for testing
        num_workers=2,  # Use fewer workers for testing
        batch_size=5,  # Small batch for testing
        timeout=30,
        enable_progress=True,
        enable_monitoring=True
    )
    
    batch_processor = BatchProcessor(config, ocr_model)
    
    # Step 4: Initialize monitoring
    print("\n4. Starting system monitoring...")
    monitor = SystemMonitor(log_dir="logs")
    monitor.start()
    
    performance_tracker = PerformanceTracker()
    performance_tracker.start_tracking()
    
    # Step 5: Prepare test batch (use first 5 documents for quick test)
    print("\n5. Preparing test batch...")
    test_docs = []
    ground_truth = {}
    
    for i, (doc_id, doc_data) in enumerate(list(standardized.items())[:5]):
        try:
            image, gt = loader.prepare_for_ocr(doc_id)
            test_docs.append((doc_id, image))
            ground_truth[doc_id] = gt['text']
            print(f"  Loaded document {doc_id}")
        except Exception as e:
            print(f"  Failed to load {doc_id}: {e}")
            continue
    
    if not test_docs:
        print("No documents could be loaded for testing")
        return False
    
    # Step 6: Process batch
    print(f"\n6. Processing {len(test_docs)} documents...")
    print("-" * 40)
    
    # Show initial system stats
    monitor.print_live_stats()
    
    # Process documents
    results = batch_processor.process_batch(test_docs, ground_truth)
    
    # Record performance metrics
    for result in results:
        performance_tracker.record_document(
            result.processing_time, 
            success=result.success
        )
    
    performance_tracker.stop_tracking()
    
    # Show final system stats
    monitor.print_live_stats()
    monitor.stop()
    
    # Step 7: Evaluate results
    print("\n7. Evaluating OCR accuracy...")
    print("-" * 40)
    
    evaluator = EvaluationMetrics(normalize_text=True, lowercase=True)
    
    eval_docs = []
    eval_results = []  # Initialize to avoid UnboundLocalError
    for result in results:
        if result.success and result.doc_id in ground_truth:
            eval_docs.append((
                result.doc_id,
                ground_truth[result.doc_id],
                result.ocr_text
            ))
    
    if eval_docs:
        eval_results = evaluator.evaluate_batch(
            eval_docs,
            confidences=[r.confidence_score for r in results if r.success]
        )
        
        # Print evaluation summary
        evaluator.print_summary()
    else:
        print("No successful results to evaluate")
    
    # Step 8: Print performance summary
    print("\n8. Performance Summary")
    print("-" * 40)
    
    # Batch processing stats
    batch_stats = batch_processor.get_statistics()
    print("\nBatch Processing Statistics:")
    print(f"  Total processed: {batch_stats.get('total_processed', 0)}")
    print(f"  Successful: {batch_stats.get('successful', 0)}")
    print(f"  Failed: {batch_stats.get('failed', 0)}")
    print(f"  Success rate: {batch_stats.get('success_rate', 0)*100:.1f}%")
    print(f"  Device used: {batch_stats.get('device_used', 'unknown')}")
    print(f"  Workers: {batch_stats.get('num_workers', 0)}")
    
    if batch_stats.get('avg_processing_time'):
        print(f"  Avg time/doc: {batch_stats['avg_processing_time']:.2f}s")
        print(f"  Docs/second: {batch_stats.get('docs_per_second', 0):.2f}")
        print(f"  Docs/minute: {batch_stats.get('docs_per_second', 0)*60:.1f}")
    
    # Performance tracking stats
    perf_metrics = performance_tracker.get_metrics()
    if perf_metrics:
        print("\nPerformance Metrics:")
        print(f"  Total time: {perf_metrics.get('total_time_seconds', 0):.2f}s")
        print(f"  Throughput: {perf_metrics.get('throughput_docs_per_minute', 0):.1f} docs/min")
    
    # System monitoring stats
    monitor_summary = monitor.get_summary()
    if monitor_summary:
        print("\nSystem Resource Usage:")
        print(f"  Avg CPU: {monitor_summary['cpu']['avg_percent']:.1f}%")
        print(f"  Peak CPU: {monitor_summary['cpu']['max_percent']:.1f}%")
        print(f"  Avg Memory: {monitor_summary['memory']['avg_percent']:.1f}%")
        print(f"  Peak Memory: {monitor_summary['memory']['peak_used_gb']:.2f} GB")
        print(f"  Process Memory: {monitor_summary['memory']['peak_process_gb']:.2f} GB")
    
    # Step 9: Test worst/best performers
    if eval_results:
        print("\n9. Document Performance Analysis")
        print("-" * 40)
        
        worst = evaluator.get_worst_performers(2)
        if worst:
            print("\nWorst performing documents:")
            for doc in worst:
                print(f"  {doc.doc_id}: WER={doc.wer:.3f}, CER={doc.cer:.3f}")
        
        best = evaluator.get_best_performers(2)
        if best:
            print("\nBest performing documents:")
            for doc in best:
                print(f"  {doc.doc_id}: WER={doc.wer:.3f}, CER={doc.cer:.3f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return True


def test_evaluation_only():
    """Test evaluation metrics independently."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS TEST")
    print("=" * 60)
    
    evaluator = EvaluationMetrics()
    
    # Test cases
    test_cases = [
        ("doc1", "The quick brown fox", "The quick brown fox", 1.0),  # Perfect match
        ("doc2", "The quick brown fox", "The quik brown fox", 0.8),   # One typo
        ("doc3", "The quick brown fox", "quick brown fox", 0.7),      # Missing word
        ("doc4", "The quick brown fox", "The slow brown dog", 0.5),   # Multiple errors
        ("doc5", "Hello world", "Goodbye universe", 0.2),             # Completely different
    ]
    
    print("\nTesting individual documents:")
    for doc_id, reference, hypothesis, confidence in test_cases:
        result = evaluator.evaluate_document(doc_id, reference, hypothesis, confidence)
        print(f"\n{doc_id}:")
        print(f"  Reference:  '{reference}'")
        print(f"  Hypothesis: '{hypothesis}'")
        print(f"  WER: {result.wer:.3f}, CER: {result.cer:.3f}")
        print(f"  Word Acc: {result.word_accuracy:.1%}, Char Acc: {result.char_accuracy:.1%}")
    
    # Print aggregate metrics
    evaluator.print_summary()
    
    # Test correlation analysis
    correlation = evaluator.get_correlation_analysis()
    if correlation:
        print("\nCorrelation Analysis:")
        for key, value in correlation.items():
            if isinstance(value, dict):
                print(f"  {key}: r={value['correlation']:.3f}, p={value['p_value']:.3f}")
    
    return True


if __name__ == "__main__":
    # Test evaluation metrics first
    print("Testing evaluation metrics...")
    test_evaluation_only()
    
    # Then test integrated pipeline
    print("\n\nTesting integrated pipeline...")
    success = test_integrated_pipeline()
    
    sys.exit(0 if success else 1)