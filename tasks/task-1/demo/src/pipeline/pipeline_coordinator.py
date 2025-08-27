"""
Pipeline Coordinator

This module provides the main coordination logic between the image transform
pipeline and the OCR reader. It manages the flow of images through the
processing stages.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from threading import Thread, Event

from .image_transform_pipeline import ImageTransformPipeline, ImageTransformConfig, TransformedImage
from .ocr_reader import OCRReader, OCRConfig, OCRResult


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    
    # Transform pipeline config
    transform_config: ImageTransformConfig = None
    
    # OCR reader config  
    ocr_config: OCRConfig = None
    
    # Coordination settings
    max_buffer_refill_workers: int = 1  # Workers for refilling transform buffer
    enable_continuous_processing: bool = True  # Keep transform pipeline running
    stats_update_interval: float = 5.0  # seconds
    
    def __post_init__(self):
        if self.transform_config is None:
            self.transform_config = ImageTransformConfig()
        if self.ocr_config is None:
            self.ocr_config = OCRConfig()


class PipelineCoordinator:
    """
    Coordinates the image transform pipeline and OCR reader.
    
    This class manages the flow of images through the processing pipeline:
    1. Images are loaded and transformed in parallel (CPU pipeline)
    2. Transformed images are buffered
    3. OCR reader processes images from the buffer (main thread only)
    4. Pipeline refills buffer when it gets low
    """
    
    def __init__(self, config: PipelineConfig, ocr_model: Any):
        """
        Initialize the pipeline coordinator.
        
        Args:
            config: Pipeline configuration
            ocr_model: OCR model instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.transform_pipeline = ImageTransformPipeline(config.transform_config)
        self.ocr_reader = OCRReader(config.ocr_config, ocr_model)
        
        # State management
        self._processing = False
        self._stop_event = Event()
        self._buffer_refill_executor = None
        
        # Statistics
        self._stats = {
            'total_images_processed': 0,
            'successful_ocr': 0,
            'failed_ocr': 0,
            'total_processing_time': 0.0,
            'transform_time': 0.0,
            'ocr_time': 0.0,
            'start_time': 0.0
        }
        
        self.logger.info("PipelineCoordinator initialized")
        self.logger.info(f"Transform batch size: {config.transform_config.batch_size}")
        self.logger.info(f"OCR batch size: {config.ocr_config.batch_size}")
    
    def _refill_buffer_worker(self, image_paths: List[Tuple[str, str]]) -> None:
        """
        Worker function to refill the transform buffer.
        
        Args:
            image_paths: List of (image_path, doc_id) tuples
        """
        try:
            self.logger.info(f"Refilling buffer with {len(image_paths)} images")
            start_time = time.time()
            
            self.transform_pipeline.fill_buffer(image_paths)
            
            refill_time = time.time() - start_time
            self._stats['transform_time'] += refill_time
            
            self.logger.info(f"Buffer refill completed in {refill_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Buffer refill failed: {e}")
    
    def _ensure_buffer_filled(self, 
                             remaining_images: List[Tuple[str, str]], 
                             min_buffer_size: int) -> List[Tuple[str, str]]:
        """
        Ensure the buffer has enough images for processing.
        
        Args:
            remaining_images: Images not yet processed
            min_buffer_size: Minimum buffer size to maintain
            
        Returns:
            Updated list of remaining images
        """
        current_buffer_size = self.transform_pipeline.get_buffer_size()
        
        if current_buffer_size < min_buffer_size and remaining_images:
            # Calculate how many images to process
            images_needed = max(
                min_buffer_size - current_buffer_size,
                self.config.transform_config.batch_size
            )
            
            images_to_process = remaining_images[:images_needed]
            remaining_images = remaining_images[images_needed:]
            
            # Submit buffer refill (async if continuous processing enabled)
            if self.config.enable_continuous_processing and self._buffer_refill_executor:
                # Async refill
                future = self._buffer_refill_executor.submit(
                    self._refill_buffer_worker, images_to_process
                )
                self.logger.info(f"Submitted {len(images_to_process)} images for async processing")
            else:
                # Sync refill
                self._refill_buffer_worker(images_to_process)
        
        return remaining_images
    
    def process_images(self, 
                      image_paths: List[Tuple[str, str]],
                      progress_callback: Optional[Callable[[Dict], None]] = None) -> List[OCRResult]:
        """
        Process a list of images through the complete pipeline.
        
        Args:
            image_paths: List of (image_path, doc_id) tuples
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of OCRResult objects
        """
        if not image_paths:
            self.logger.warning("No images provided for processing")
            return []
        
        self.logger.info(f"Starting pipeline processing of {len(image_paths)} images")
        
        self._processing = True
        self._stop_event.clear()
        self._stats['start_time'] = time.time()
        
        all_results = []
        remaining_images = image_paths.copy()
        
        try:
            # Initialize buffer refill executor if continuous processing enabled
            if self.config.enable_continuous_processing:
                self._buffer_refill_executor = ThreadPoolExecutor(
                    max_workers=self.config.max_buffer_refill_workers,
                    thread_name_prefix="buffer_refill"
                )
            
            # Initial buffer fill
            initial_batch_size = min(
                self.config.transform_config.batch_size * 2,  # Fill 2 batches initially
                len(remaining_images)
            )
            
            initial_images = remaining_images[:initial_batch_size]
            remaining_images = remaining_images[initial_batch_size:]
            
            self._refill_buffer_worker(initial_images)
            
            # Process images while there are items in buffer or remaining images
            while (not self.transform_pipeline.is_buffer_empty() or remaining_images) and not self._stop_event.is_set():
                
                # Ensure buffer has enough images
                remaining_images = self._ensure_buffer_filled(
                    remaining_images, 
                    self.config.ocr_config.batch_size
                )
                
                # Get batch from transform pipeline
                transform_batch = self.transform_pipeline.get_batch(
                    batch_size=self.config.ocr_config.batch_size,
                    timeout=10.0
                )
                
                if not transform_batch:
                    if remaining_images:
                        # Wait a bit for buffer to fill
                        time.sleep(0.5)
                        continue
                    else:
                        # No more images to process
                        break
                
                # Process with OCR reader
                ocr_start_time = time.time()
                ocr_results = self.ocr_reader.process_batch(transform_batch)
                ocr_time = time.time() - ocr_start_time
                
                # Update statistics
                self._stats['total_images_processed'] += len(ocr_results)
                self._stats['successful_ocr'] += sum(1 for r in ocr_results if r.success)
                self._stats['failed_ocr'] += sum(1 for r in ocr_results if not r.success)
                self._stats['ocr_time'] += ocr_time
                
                all_results.extend(ocr_results)
                
                # Progress callback
                if progress_callback:
                    progress_stats = self.get_processing_statistics()
                    progress_stats['completed'] = len(all_results)
                    progress_stats['remaining'] = len(remaining_images) + self.transform_pipeline.get_buffer_size()
                    progress_callback(progress_stats)
                
                self.logger.info(f"Processed batch: {len(ocr_results)} images, "
                               f"{len(all_results)}/{len(image_paths)} total completed")
            
            # Final statistics
            self._stats['total_processing_time'] = time.time() - self._stats['start_time']
            
            self.logger.info(f"Pipeline processing completed: {len(all_results)} results")
            self.logger.info(f"Success rate: {self._stats['successful_ocr']}/{self._stats['total_images_processed']}")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            raise
            
        finally:
            self._processing = False
            
            # Cleanup
            if self._buffer_refill_executor:
                self._buffer_refill_executor.shutdown(wait=True)
                self._buffer_refill_executor = None
    
    def stop_processing(self) -> None:
        """Stop the pipeline processing gracefully."""
        self.logger.info("Stopping pipeline processing...")
        self._stop_event.set()
        
        if self._buffer_refill_executor:
            self._buffer_refill_executor.shutdown(wait=False)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        current_time = time.time()
        elapsed_time = current_time - self._stats['start_time'] if self._stats['start_time'] > 0 else 0
        
        stats = {
            'pipeline_status': {
                'is_processing': self._processing,
                'elapsed_time': elapsed_time,
                'total_images_processed': self._stats['total_images_processed'],
                'successful_ocr': self._stats['successful_ocr'],
                'failed_ocr': self._stats['failed_ocr'],
                'success_rate': self._stats['successful_ocr'] / max(self._stats['total_images_processed'], 1)
            },
            'timing': {
                'total_processing_time': self._stats['total_processing_time'],
                'transform_time': self._stats['transform_time'],
                'ocr_time': self._stats['ocr_time'],
                'images_per_second': self._stats['total_images_processed'] / max(elapsed_time, 0.001)
            },
            'buffer_status': {
                'current_buffer_size': self.transform_pipeline.get_buffer_size(),
                'buffer_empty': self.transform_pipeline.is_buffer_empty()
            },
            'component_stats': {
                'transform_pipeline': self.transform_pipeline.get_statistics(),
                'ocr_reader': self.ocr_reader.get_statistics()
            }
        }
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            'total_images_processed': 0,
            'successful_ocr': 0,
            'failed_ocr': 0,
            'total_processing_time': 0.0,
            'transform_time': 0.0,
            'ocr_time': 0.0,
            'start_time': 0.0
        }
        self.logger.info("Statistics reset")
    
    def process_single_directory(self, 
                                directory_path: str,
                                image_extensions: List[str] = None,
                                progress_callback: Optional[Callable[[Dict], None]] = None) -> List[OCRResult]:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            image_extensions: List of image file extensions to process
            progress_callback: Optional progress callback
            
        Returns:
            List of OCRResult objects
        """
        if image_extensions is None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_paths = []
        for ext in image_extensions:
            pattern = f"*{ext}"
            files = list(directory.glob(pattern))
            files.extend(directory.glob(pattern.upper()))  # Also match uppercase
            
            for file_path in files:
                doc_id = file_path.stem  # Use filename without extension as doc_id
                image_paths.append((str(file_path), doc_id))
        
        self.logger.info(f"Found {len(image_paths)} images in {directory_path}")
        
        if not image_paths:
            self.logger.warning(f"No images found in {directory_path}")
            return []
        
        return self.process_images(image_paths, progress_callback)
    
    def shutdown(self) -> None:
        """Shutdown the pipeline coordinator and cleanup resources."""
        self.stop_processing()
        
        # Shutdown components
        self.transform_pipeline.shutdown()
        self.ocr_reader.cleanup()
        
        self.logger.info("PipelineCoordinator shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()