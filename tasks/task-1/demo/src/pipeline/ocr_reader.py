"""
OCR Reader with Configurable Backend

This module provides a single-threaded OCR reader that processes images
from the transform pipeline. It runs only in the main thread/process
and supports both CPU and GPU backends.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .image_transform_pipeline import TransformedImage


@dataclass
class OCRConfig:
    """Configuration for OCR reader."""
    
    device: str = "auto"  # auto, cpu, gpu
    batch_size: int = 1  # Number of images to process per OCR call
    num_workers: int = 1  # For OCR models that support internal parallelization
    confidence_threshold: float = 0.0
    timeout_per_image: float = 30.0  # seconds
    
    # Model-specific kwargs
    model_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}


@dataclass
class OCRResult:
    """Result from OCR processing."""
    
    doc_id: str
    success: bool
    ocr_text: str = ""
    processing_time: float = 0.0
    error_message: str = ""
    confidence_score: float = 0.0
    device_used: str = ""
    
    # Visual data for analysis
    text_lines: List[str] = None
    bounding_boxes: List[List] = None
    line_confidences: List[float] = None
    image_size: Tuple[int, int] = None
    
    # Source image info
    original_path: str = ""
    transform_time: float = 0.0
    transforms_applied: List[str] = None
    
    def __post_init__(self):
        if self.text_lines is None:
            self.text_lines = []
        if self.bounding_boxes is None:
            self.bounding_boxes = []
        if self.line_confidences is None:
            self.line_confidences = []
        if self.transforms_applied is None:
            self.transforms_applied = []


class OCRReader:
    """
    Single-threaded OCR reader with configurable backend.
    
    This class processes images from the transform pipeline using
    various OCR models. It runs only in the main thread and supports
    both CPU and GPU processing.
    """
    
    def __init__(self, config: OCRConfig, ocr_model: Any):
        """
        Initialize OCR reader.
        
        Args:
            config: OCR configuration
            ocr_model: OCR model instance (EasyOCR, PaddleOCR, etc.)
        """
        self.config = config
        self.ocr_model = ocr_model
        self.logger = logging.getLogger(__name__)
        
        # Detect device
        self.device = self._detect_device()
        
        # Initialize model on correct device if possible
        self._setup_model_device()
        
        self.logger.info(f"OCRReader initialized on device: {self.device}")
        self.logger.info(f"Batch size: {config.batch_size}")
        self.logger.info(f"Model type: {type(ocr_model).__name__}")
    
    def _detect_device(self) -> str:
        """Detect and select the best available device."""
        if self.config.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "gpu"
            else:
                return "cpu"
        elif self.config.device == "gpu":
            if not TORCH_AVAILABLE:
                self.logger.warning("GPU requested but PyTorch not available, falling back to CPU")
                return "cpu"
            elif not torch.cuda.is_available():
                self.logger.warning("GPU requested but CUDA not available, falling back to CPU")
                return "cpu"
            else:
                return "gpu"
        else:
            return self.config.device
    
    def _setup_model_device(self) -> None:
        """Setup model on the correct device if supported."""
        try:
            # Try to move model to GPU if using EasyOCR or similar
            if self.device == "gpu" and hasattr(self.ocr_model, 'recognizer'):
                # EasyOCR models
                if hasattr(self.ocr_model.recognizer, 'model'):
                    self.ocr_model.recognizer.model = self.ocr_model.recognizer.model.cuda()
                    self.logger.info("Moved EasyOCR recognizer to GPU")
                    
                if hasattr(self.ocr_model, 'detector') and hasattr(self.ocr_model.detector, 'model'):
                    self.ocr_model.detector.model = self.ocr_model.detector.model.cuda()
                    self.logger.info("Moved EasyOCR detector to GPU")
                    
            elif self.device == "gpu" and hasattr(self.ocr_model, 'use_gpu'):
                # PaddleOCR models
                self.ocr_model.use_gpu = True
                self.logger.info("Enabled GPU for PaddleOCR")
                
        except Exception as e:
            self.logger.warning(f"Could not setup model device: {e}")
    
    def _process_single_image(self, transformed_image: TransformedImage) -> OCRResult:
        """
        Process a single transformed image with OCR.
        
        Args:
            transformed_image: Image from the transform pipeline
            
        Returns:
            OCRResult object
        """
        if not transformed_image.success:
            return OCRResult(
                doc_id=transformed_image.doc_id,
                success=False,
                error_message=f"Image transform failed: {transformed_image.error_message}",
                device_used=self.device,
                original_path=transformed_image.original_path,
                transform_time=transformed_image.processing_time,
                transforms_applied=transformed_image.transforms_applied
            )
        
        start_time = time.time()
        
        try:
            image = transformed_image.image
            
            # Run OCR based on model type
            if hasattr(self.ocr_model, 'process_image'):
                # Our custom wrapper models
                result = self.ocr_model.process_image(image)
                
                if not isinstance(result, dict):
                    raise ValueError(f"OCR model returned invalid result type: {type(result)}")
                
                ocr_text = result.get('text', '')
                confidence = result.get('avg_confidence', 0.0)
                
                # Extract visual data
                text_lines = result.get('text_lines', [])
                bounding_boxes = result.get('boxes', [])
                line_confidences = result.get('confidences', [])
                
                # Handle error cases
                if 'error' in result:
                    self.logger.warning(f"OCR model returned error: {result['error']}")
                    
            elif hasattr(self.ocr_model, 'readtext'):
                # Raw EasyOCR
                result = self.ocr_model.readtext(image)
                
                if result and isinstance(result, list):
                    text_lines = [text[1] for text in result]
                    ocr_text = ' '.join(text_lines)
                    line_confidences = [text[2] for text in result]
                    confidence = np.mean(line_confidences) if line_confidences else 0.0
                    bounding_boxes = [text[0] for text in result]
                else:
                    ocr_text = ""
                    confidence = 0.0
                    text_lines = []
                    bounding_boxes = []
                    line_confidences = []
                    
            elif hasattr(self.ocr_model, 'ocr'):
                # PaddleOCR
                result = self.ocr_model.ocr(image)
                
                if result and result[0]:
                    text_lines = [line[1][0] for line in result[0]]
                    ocr_text = ' '.join(text_lines)
                    line_confidences = [line[1][1] for line in result[0]]
                    confidence = np.mean(line_confidences) if line_confidences else 0.0
                    bounding_boxes = [line[0] for line in result[0]]
                else:
                    ocr_text = ""
                    confidence = 0.0
                    text_lines = []
                    bounding_boxes = []
                    line_confidences = []
                    
            else:
                raise AttributeError(f"OCR model has no recognized method (process_image, readtext, ocr)")
            
            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                self.logger.warning(f"Low confidence result for {transformed_image.doc_id}: {confidence}")
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                doc_id=transformed_image.doc_id,
                success=True,
                ocr_text=ocr_text,
                processing_time=processing_time,
                confidence_score=confidence,
                device_used=self.device,
                text_lines=text_lines,
                bounding_boxes=bounding_boxes,
                line_confidences=line_confidences,
                image_size=transformed_image.final_size,
                original_path=transformed_image.original_path,
                transform_time=transformed_image.processing_time,
                transforms_applied=transformed_image.transforms_applied
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"OCR processing failed for {transformed_image.doc_id}: {e}")
            
            return OCRResult(
                doc_id=transformed_image.doc_id,
                success=False,
                processing_time=processing_time,
                error_message=str(e),
                device_used=self.device,
                original_path=transformed_image.original_path,
                transform_time=transformed_image.processing_time,
                transforms_applied=transformed_image.transforms_applied
            )
    
    def _process_batch_optimized(self, transformed_images: List[TransformedImage]) -> List[OCRResult]:
        """
        Process a batch of images with potential optimization for batch-capable models.
        
        Args:
            transformed_images: List of transformed images
            
        Returns:
            List of OCRResult objects
        """
        # For now, process images individually
        # Future optimization: check if model supports true batch processing
        results = []
        
        for transformed_image in transformed_images:
            if not transformed_image.success:
                # Skip failed transforms but still create result
                result = OCRResult(
                    doc_id=transformed_image.doc_id,
                    success=False,
                    error_message=f"Transform failed: {transformed_image.error_message}",
                    device_used=self.device,
                    original_path=transformed_image.original_path,
                    transform_time=transformed_image.processing_time,
                    transforms_applied=transformed_image.transforms_applied
                )
                results.append(result)
                continue
            
            result = self._process_single_image(transformed_image)
            results.append(result)
            
            # Clear GPU cache periodically if using GPU
            if self.device == "gpu" and TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def process_batch(self, transformed_images: List[TransformedImage]) -> List[OCRResult]:
        """
        Process a batch of transformed images.
        
        Args:
            transformed_images: List of transformed images to process
            
        Returns:
            List of OCRResult objects
        """
        if not transformed_images:
            self.logger.warning("No images provided for OCR processing")
            return []
        
        self.logger.info(f"Processing OCR batch of {len(transformed_images)} images")
        start_time = time.time()
        
        # Process the batch
        results = self._process_batch_optimized(transformed_images)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        
        self.logger.info(f"OCR batch completed: {successful}/{len(results)} successful in {total_time:.2f}s")
        
        return results
    
    def process_images(self, 
                      transformed_images: List[TransformedImage],
                      batch_size: Optional[int] = None) -> List[OCRResult]:
        """
        Process multiple images, potentially in batches.
        
        Args:
            transformed_images: List of transformed images
            batch_size: Override batch size for this processing
            
        Returns:
            List of OCRResult objects
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(transformed_images), batch_size):
            batch = transformed_images[i:i + batch_size]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get OCR reader statistics."""
        stats = {
            "config": {
                "device": self.device,
                "batch_size": self.config.batch_size,
                "confidence_threshold": self.config.confidence_threshold,
                "timeout_per_image": self.config.timeout_per_image
            },
            "model_info": {
                "type": type(self.ocr_model).__name__,
                "has_process_image": hasattr(self.ocr_model, 'process_image'),
                "has_readtext": hasattr(self.ocr_model, 'readtext'),
                "has_ocr": hasattr(self.ocr_model, 'ocr')
            }
        }
        
        # Add device-specific info
        if self.device == "gpu" and TORCH_AVAILABLE and torch.cuda.is_available():
            stats["gpu_info"] = {
                "gpu_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_cached": torch.cuda.memory_reserved()
            }
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.device == "gpu" and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("OCRReader cleanup complete")