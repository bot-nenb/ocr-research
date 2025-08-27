"""
CPU Image Transform Pipeline with Parallel Processing

This module provides a parallel CPU pipeline for image preprocessing and transforms
using cv2. It loads and processes images in batches using thread or process pools.
"""

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from queue import Queue
from threading import Lock

import cv2
import numpy as np


@dataclass
class ImageTransformConfig:
    """Configuration for image transform pipeline."""
    
    batch_size: int = 10
    num_workers: int = 4
    use_processes: bool = False  # False = threads, True = processes
    max_image_size: Tuple[int, int] = (2048, 2048)
    normalize: bool = True
    grayscale_conversion: bool = False
    gaussian_blur: Optional[Tuple[int, int]] = None  # (ksize_x, ksize_y)
    adaptive_threshold: bool = False
    resize_images: bool = False
    target_size: Optional[Tuple[int, int]] = None
    quality_enhancement: bool = True
    timeout: float = 30.0  # seconds per image


@dataclass 
class TransformedImage:
    """Container for transformed image with metadata."""
    
    doc_id: str
    image: np.ndarray
    original_path: str
    processing_time: float
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    transforms_applied: List[str]
    success: bool = True
    error_message: str = ""


class ImageTransformPipeline:
    """
    Parallel CPU pipeline for image preprocessing and transforms.
    
    This pipeline loads images from disk and applies various CV2 transforms
    in parallel using either threads or processes. It maintains a buffer
    of transformed images that can be consumed by the OCR reader.
    """
    
    def __init__(self, config: ImageTransformConfig):
        """
        Initialize the image transform pipeline.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe buffer for transformed images
        self._buffer = Queue(maxsize=config.batch_size * 2)  # Allow some buffering
        self._buffer_lock = Lock()
        
        # Executor for parallel processing
        self.executor = None
        
        self.logger.info(f"ImageTransformPipeline initialized with {config.num_workers} workers")
        self.logger.info(f"Using {'processes' if config.use_processes else 'threads'} for parallelization")
    
    def _apply_transforms(self, image: np.ndarray, transforms_applied: List[str]) -> np.ndarray:
        """
        Apply configured transforms to an image.
        
        Args:
            image: Input image as numpy array
            transforms_applied: List to track applied transforms
            
        Returns:
            Transformed image
        """
        original_image = image.copy()
        
        try:
            # Resize if requested
            if self.config.resize_images and self.config.target_size:
                image = cv2.resize(image, self.config.target_size, interpolation=cv2.INTER_AREA)
                transforms_applied.append(f"resize_{self.config.target_size}")
            
            # Limit maximum image size
            if self.config.max_image_size:
                h, w = image.shape[:2]
                max_h, max_w = self.config.max_image_size
                
                if h > max_h or w > max_w:
                    # Calculate scale factor
                    scale = min(max_h / h, max_w / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    transforms_applied.append(f"max_size_limit_{new_w}x{new_h}")
            
            # Convert to grayscale if requested
            if self.config.grayscale_conversion and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                transforms_applied.append("grayscale")
                # Convert back to 3-channel for consistency
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                transforms_applied.append("gray_to_bgr")
            
            # Quality enhancement
            if self.config.quality_enhancement:
                # Noise reduction
                if len(image.shape) == 3:
                    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                else:
                    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
                transforms_applied.append("noise_reduction")
                
                # Enhance contrast using CLAHE
                if len(image.shape) == 3:
                    # Apply to each channel
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    transforms_applied.append("clahe_color")
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    image = clahe.apply(image)
                    transforms_applied.append("clahe_gray")
            
            # Gaussian blur if requested
            if self.config.gaussian_blur:
                ksize_x, ksize_y = self.config.gaussian_blur
                # Ensure odd kernel size
                ksize_x = ksize_x if ksize_x % 2 == 1 else ksize_x + 1
                ksize_y = ksize_y if ksize_y % 2 == 1 else ksize_y + 1
                image = cv2.GaussianBlur(image, (ksize_x, ksize_y), 0)
                transforms_applied.append(f"gaussian_blur_{ksize_x}x{ksize_y}")
            
            # Adaptive threshold (only for grayscale)
            if self.config.adaptive_threshold:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                
                # Apply adaptive threshold
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                
                # Convert back to 3-channel
                image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                transforms_applied.append("adaptive_threshold")
            
            # Normalize if requested
            if self.config.normalize:
                if image.dtype != np.uint8:
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    transforms_applied.append("normalize_uint8")
                else:
                    # Normalize histogram
                    if len(image.shape) == 3:
                        # Normalize each channel
                        for i in range(image.shape[2]):
                            image[:, :, i] = cv2.equalizeHist(image[:, :, i])
                    else:
                        image = cv2.equalizeHist(image)
                    transforms_applied.append("histogram_equalization")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error applying transforms: {e}")
            return original_image  # Return original on error
    
    def _load_and_transform_image(self, image_path: str, doc_id: str) -> TransformedImage:
        """
        Load and transform a single image.
        
        Args:
            image_path: Path to image file
            doc_id: Document identifier
            
        Returns:
            TransformedImage object
        """
        start_time = time.time()
        transforms_applied = []
        
        try:
            # Load image
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read image with cv2
            image = cv2.imread(str(image_path_obj))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            transforms_applied.append("loaded_with_cv2")
            
            # Apply transforms
            transformed_image = self._apply_transforms(image, transforms_applied)
            
            final_size = (transformed_image.shape[1], transformed_image.shape[0])
            processing_time = time.time() - start_time
            
            return TransformedImage(
                doc_id=doc_id,
                image=transformed_image,
                original_path=image_path,
                processing_time=processing_time,
                original_size=original_size,
                final_size=final_size,
                transforms_applied=transforms_applied,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing image {doc_id}: {e}")
            
            return TransformedImage(
                doc_id=doc_id,
                image=np.array([]),  # Empty array on error
                original_path=image_path,
                processing_time=processing_time,
                original_size=(0, 0),
                final_size=(0, 0),
                transforms_applied=transforms_applied,
                success=False,
                error_message=str(e)
            )
    
    def _worker_function(self, image_path: str, doc_id: str) -> TransformedImage:
        """
        Worker function for parallel processing.
        This function will be executed by threads or processes.
        """
        return self._load_and_transform_image(image_path, doc_id)
    
    def fill_buffer(self, image_paths: List[Tuple[str, str]]) -> None:
        """
        Fill the buffer with transformed images using parallel processing.
        
        Args:
            image_paths: List of (image_path, doc_id) tuples to process
        """
        if not image_paths:
            self.logger.warning("No image paths provided to fill buffer")
            return
        
        self.logger.info(f"Processing {len(image_paths)} images to fill buffer")
        
        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.config.num_workers) as executor:
            self.executor = executor
            
            # Submit all tasks
            future_to_info = {}
            for image_path, doc_id in image_paths:
                future = executor.submit(self._worker_function, image_path, doc_id)
                future_to_info[future] = (image_path, doc_id)
            
            # Collect results as they complete
            for future in as_completed(future_to_info, timeout=self.config.timeout):
                image_path, doc_id = future_to_info[future]
                
                try:
                    result = future.result(timeout=5.0)  # Short timeout since work is done
                    
                    # Add to buffer (thread-safe)
                    with self._buffer_lock:
                        # If buffer is full, this will block until space is available
                        self._buffer.put(result, timeout=10.0)
                    
                    if result.success:
                        self.logger.debug(f"Successfully processed {doc_id} in {result.processing_time:.2f}s")
                    else:
                        self.logger.warning(f"Failed to process {doc_id}: {result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"Worker failed for {doc_id}: {e}")
                    # Still put an error result in the buffer
                    error_result = TransformedImage(
                        doc_id=doc_id,
                        image=np.array([]),
                        original_path=image_path,
                        processing_time=0.0,
                        original_size=(0, 0),
                        final_size=(0, 0),
                        transforms_applied=[],
                        success=False,
                        error_message=str(e)
                    )
                    
                    with self._buffer_lock:
                        self._buffer.put(error_result, timeout=10.0)
        
        self.executor = None
        self.logger.info(f"Finished processing {len(image_paths)} images")
    
    def get_batch(self, batch_size: Optional[int] = None, timeout: float = 30.0) -> List[TransformedImage]:
        """
        Get a batch of transformed images from the buffer.
        
        Args:
            batch_size: Number of images to get (defaults to config batch_size)
            timeout: Maximum time to wait for images
            
        Returns:
            List of TransformedImage objects
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        batch = []
        start_time = time.time()
        
        while len(batch) < batch_size and (time.time() - start_time) < timeout:
            try:
                with self._buffer_lock:
                    if not self._buffer.empty():
                        result = self._buffer.get(timeout=1.0)
                        batch.append(result)
                    else:
                        # No more images available
                        break
            except:
                # Timeout or other queue error
                break
        
        self.logger.info(f"Retrieved batch of {len(batch)} images from buffer")
        return batch
    
    def get_buffer_size(self) -> int:
        """Get current number of items in buffer."""
        with self._buffer_lock:
            return self._buffer.qsize()
    
    def is_buffer_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._buffer_lock:
            return self._buffer.empty()
    
    def clear_buffer(self) -> None:
        """Clear all items from buffer."""
        with self._buffer_lock:
            while not self._buffer.empty():
                try:
                    self._buffer.get_nowait()
                except:
                    break
        self.logger.info("Buffer cleared")
    
    def shutdown(self) -> None:
        """Shutdown the pipeline and cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=False)
        self.clear_buffer()
        self.logger.info("ImageTransformPipeline shutdown complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "config": {
                "batch_size": self.config.batch_size,
                "num_workers": self.config.num_workers,
                "use_processes": self.config.use_processes,
                "max_image_size": self.config.max_image_size,
                "transforms_enabled": {
                    "normalize": self.config.normalize,
                    "grayscale_conversion": self.config.grayscale_conversion,
                    "gaussian_blur": self.config.gaussian_blur is not None,
                    "adaptive_threshold": self.config.adaptive_threshold,
                    "resize_images": self.config.resize_images,
                    "quality_enhancement": self.config.quality_enhancement
                }
            },
            "buffer_status": {
                "current_size": self.get_buffer_size(),
                "is_empty": self.is_buffer_empty()
            }
        }