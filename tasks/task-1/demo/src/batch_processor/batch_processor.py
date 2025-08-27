"""
Batch Processing Framework with Device-Aware Parallelization

This module provides the core batch processing functionality for OCR tasks,
supporting both CPU multiprocessing and GPU batch processing.
"""

import logging
import multiprocessing as mp
import queue
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class ProcessingConfig:
    """Configuration for batch processing."""
    
    device: str = "auto"  # auto, cpu, gpu
    num_workers: int = None  # None = auto-detect
    batch_size: int = 10
    timeout: int = 30  # seconds per document
    max_retries: int = 3
    gpu_memory_limit: float = 0.8  # Use 80% of GPU memory
    enable_progress: bool = True
    enable_monitoring: bool = True


@dataclass
class ProcessingResult:
    """Result from processing a single document."""
    
    doc_id: str
    success: bool
    ocr_text: str = ""
    processing_time: float = 0.0
    error_message: str = ""
    device_used: str = ""
    worker_id: int = -1
    memory_used: float = 0.0
    confidence_score: float = 0.0
    # Visual comparison data
    image_path: str = ""
    text_lines: List = None
    bounding_boxes: List = None
    line_confidences: List = None
    image_size: Tuple = None


class DeviceManager:
    """Manages device selection and capabilities."""
    
    def __init__(self, preferred_device: str = "auto"):
        self.preferred_device = preferred_device
        self.has_gpu = torch.cuda.is_available()
        self.cpu_count = mp.cpu_count()
        
        self.device = self._select_device()
        
        logging.info(f"Device Manager initialized: {self.device}")
        if self.has_gpu:
            logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logging.info(f"CPU cores available: {self.cpu_count}")
    
    def _select_device(self) -> str:
        """Select the best device based on availability and preference."""
        if self.preferred_device == "gpu" and not self.has_gpu:
            logging.warning("GPU requested but not available, falling back to CPU")
            return "cpu"
        elif self.preferred_device == "auto":
            return "gpu" if self.has_gpu else "cpu"
        else:
            return self.preferred_device
    
    def get_optimal_workers(self, batch_size: int) -> int:
        """Get optimal number of workers for the selected device."""
        if self.device == "gpu":
            # For GPU, use fewer workers as GPU handles parallelism
            return min(4, self.cpu_count)
        else:
            # For CPU, use more workers
            return min(batch_size, self.cpu_count)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = {
            "device": self.device,
            "cpu_count": self.cpu_count,
            "has_gpu": self.has_gpu
        }
        
        if self.has_gpu:
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_reserved()
            })
        
        return info


class BatchProcessor:
    """Main batch processing engine with device-aware parallelization."""
    
    def __init__(self, config: ProcessingConfig, ocr_model: Any):
        """
        Initialize batch processor.
        
        Args:
            config: Processing configuration
            ocr_model: OCR model instance (EasyOCR or PaddleOCR wrapper)
        """
        self.config = config
        self.ocr_model = ocr_model
        self.device_manager = DeviceManager(config.device)
        
        # Set number of workers
        if config.num_workers is None:
            self.num_workers = self.device_manager.get_optimal_workers(config.batch_size)
        else:
            self.num_workers = config.num_workers
        
        self.results: List[ProcessingResult] = []
        self.is_running = False
        self.executor = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info(f"BatchProcessor initialized with {self.num_workers} workers on {self.device_manager.device}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logging.info("Received shutdown signal, stopping processing...")
        self.is_running = False
        if self.executor:
            self.executor.shutdown(wait=False)
        sys.exit(0)
    
    def process_batch(self, documents: List[Tuple[str, np.ndarray, str]], 
                     ground_truth: Optional[Dict[str, str]] = None) -> List[ProcessingResult]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of (doc_id, image_array, image_path) tuples
            ground_truth: Optional dictionary of ground truth text
            
        Returns:
            List of processing results
        """
        self.is_running = True
        self.results = []
        
        # Create progress bar if enabled
        pbar = None
        if self.config.enable_progress:
            pbar = tqdm(total=len(documents), desc="Processing documents")
        
        try:
            if self.device_manager.device == "gpu":
                results = self._process_gpu_batch(documents, pbar)
            else:
                results = self._process_cpu_batch(documents, pbar)
            
            self.results = results
            
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            traceback.print_exc()
            
        finally:
            if pbar:
                pbar.close()
            self.is_running = False
        
        return self.results
    
    def _process_cpu_batch(self, documents: List[Tuple[str, np.ndarray, str]], 
                          pbar: Optional[tqdm] = None) -> List[ProcessingResult]:
        """Process documents using CPU with thread pool (OCR models often can't be pickled)."""
        results = []
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickle issues
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            self.executor = executor
            
            # Submit all tasks
            future_to_doc = {}
            for doc_id, image, image_path in documents:
                if not self.is_running:
                    break
                    
                future = executor.submit(self._process_single_document, 
                                       doc_id, image, "cpu", 0, image_path)
                future_to_doc[future] = (doc_id, image, image_path)
            
            # Collect results
            for future in as_completed(future_to_doc):
                if not self.is_running:
                    break
                    
                doc_id, image, image_path = future_to_doc[future]
                
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Failed to process {doc_id}: {e}")
                    results.append(ProcessingResult(
                        doc_id=doc_id,
                        success=False,
                        error_message=str(e),
                        device_used="cpu"
                    ))
                
                if pbar:
                    pbar.update(1)
        
        return results
    
    def _process_gpu_batch(self, documents: List[Tuple[str, np.ndarray, str]], 
                          pbar: Optional[tqdm] = None) -> List[ProcessingResult]:
        """Process documents using GPU batching."""
        results = []
        
        # Process in smaller batches for GPU
        batch_size = min(self.config.batch_size, 32)  # GPU batch limit
        
        for i in range(0, len(documents), batch_size):
            if not self.is_running:
                break
                
            batch = documents[i:i+batch_size]
            
            # Process batch on GPU
            for doc_id, image, image_path in batch:
                try:
                    result = self._process_single_document(doc_id, image, "gpu", 0, image_path)
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"GPU processing failed for {doc_id}: {e}")
                    results.append(ProcessingResult(
                        doc_id=doc_id,
                        success=False,
                        error_message=str(e),
                        device_used="gpu"
                    ))
                
                if pbar:
                    pbar.update(1)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def _process_single_document(self, doc_id: str, image: np.ndarray, 
                                device: str, worker_id: int, image_path: str = "") -> ProcessingResult:
        """Process a single document."""
        start_time = time.time()
        
        try:
            # Run OCR - check for the actual method names
            if hasattr(self.ocr_model, 'process_image'):
                # Our EasyOCR wrapper
                result = self.ocr_model.process_image(image)
                
                # Validate result structure
                if not isinstance(result, dict):
                    raise ValueError(f"OCR model returned invalid result type: {type(result)}")
                
                ocr_text = result.get('text', '')
                
                # Use correct confidence key with fallback
                confidence = result.get('avg_confidence', 0.0)
                if confidence == 0.0 and 'confidences' in result and result['confidences']:
                    # Fallback: calculate from individual confidences if available
                    confidence = np.mean(result['confidences'])
                    logging.debug(f"Used fallback confidence calculation: {confidence}")
                
                # Handle error cases
                if 'error' in result:
                    logging.warning(f"OCR model returned error: {result['error']}")
                    
            elif hasattr(self.ocr_model, 'readtext'):
                # Raw EasyOCR
                result = self.ocr_model.readtext(image)
                if result and isinstance(result, list):
                    ocr_text = ' '.join([text[1] for text in result])
                    confidence = np.mean([text[2] for text in result]) if result else 0.0
                else:
                    ocr_text = ""
                    confidence = 0.0
                    
            elif hasattr(self.ocr_model, 'ocr'):
                # PaddleOCR
                result = self.ocr_model.ocr(image)
                if result and result[0]:
                    ocr_text = ' '.join([line[1][0] for line in result[0]])
                    confidence = np.mean([line[1][1] for line in result[0]])
                else:
                    ocr_text = ""
                    confidence = 0.0
            else:
                raise AttributeError(f"OCR model has no recognized method (checked: process_image, readtext, ocr)")
            
            processing_time = time.time() - start_time
            
            # Extract visual data from OCR result
            text_lines = result.get('text_lines', [])
            bounding_boxes = result.get('boxes', [])
            line_confidences = result.get('confidences', [])
            image_size = result.get('image_size', (0, 0))
            
            return ProcessingResult(
                doc_id=doc_id,
                success=True,
                ocr_text=ocr_text,
                processing_time=processing_time,
                device_used=device,
                worker_id=worker_id,
                confidence_score=confidence,
                # Visual comparison data
                image_path=image_path,
                text_lines=text_lines,
                bounding_boxes=bounding_boxes,
                line_confidences=line_confidences,
                image_size=image_size
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Error processing {doc_id}: {e}")
            
            return ProcessingResult(
                doc_id=doc_id,
                success=False,
                processing_time=processing_time,
                error_message=str(e),
                device_used=device,
                worker_id=worker_id
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        stats = {
            "total_processed": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "device_used": self.device_manager.device,
            "num_workers": self.num_workers
        }
        
        if successful:
            processing_times = [r.processing_time for r in successful]
            stats.update({
                "avg_processing_time": np.mean(processing_times),
                "min_processing_time": np.min(processing_times),
                "max_processing_time": np.max(processing_times),
                "total_processing_time": np.sum(processing_times),
                "docs_per_second": len(successful) / np.sum(processing_times),
                "avg_confidence": np.mean([r.confidence_score for r in successful])
            })
        
        return stats