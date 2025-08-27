"""Simple PaddleOCR wrapper that bypasses version compatibility issues."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import time
import os
import torch
from .base_model import BaseOCRModel

logger = logging.getLogger(__name__)


class SimplePaddleOCRModel(BaseOCRModel):
    """Simplified PaddleOCR wrapper that avoids compatibility issues."""
    
    def __init__(
        self,
        device: str = "auto",
        languages: List[str] = None,
        gpu_memory_limit: Optional[float] = None,
        **kwargs
    ):
        """Initialize simple PaddleOCR model with minimal configuration."""
        super().__init__(
            model_name="PaddleOCR-Simple",
            device=device,
            languages=languages or ['en'],
            gpu_memory_limit=gpu_memory_limit,
            **kwargs
        )
        
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize PaddleOCR with minimal configuration to avoid compatibility issues."""
        try:
            # Import PaddleOCR only when needed
            from paddleocr import PaddleOCR
            
            # Use minimal configuration to avoid compatibility issues
            self._model = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
            
            logger.info(f"Simple PaddleOCR initialized successfully on {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize simple PaddleOCR: {e}")
            logger.warning("PaddleOCR has compatibility issues. Using fallback mode.")
            self._model = None
            
    def process_image(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single image with PaddleOCR."""
        start_time = time.time()
        
        if self._model is None:
            return {
                "text": "",
                "text_lines": [],
                "boxes": [],
                "confidences": [],
                "error": "PaddleOCR not available due to compatibility issues",
                "processing_time": time.time() - start_time,
                "model": "PaddleOCR-Simple",
                "device": str(self._device)
            }
        
        try:
            # Convert path to string
            image_path = str(Path(image_path).resolve())
            
            # Check if image exists
            if not Path(image_path).exists():
                raise ValueError(f"Cannot find image: {image_path}")
            
            # Perform OCR
            result = self._model.ocr(image_path, cls=True)
            
            # Parse results
            text_lines = []
            boxes = []
            confidences = []
            
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        box = line[0]
                        text_info = line[1]
                        
                        if text_info and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            boxes.append(box)
                            text_lines.append(text)
                            confidences.append(confidence)
            
            processing_time = time.time() - start_time
            
            return {
                "text": "\n".join(text_lines),
                "text_lines": text_lines,
                "boxes": boxes,
                "confidences": confidences,
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "num_detections": len(text_lines),
                "processing_time": processing_time,
                "image_size": "unknown",
                "model": "PaddleOCR-Simple",
                "device": str(self._device)
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "text": "",
                "text_lines": [],
                "boxes": [],
                "confidences": [],
                "error": str(e),
                "processing_time": time.time() - start_time,
                "model": "PaddleOCR-Simple",
                "device": str(self._device)
            }
    
    def benchmark(self, test_image: Union[str, Path], iterations: int = 10) -> Dict[str, Any]:
        """Benchmark the model performance."""
        if self._model is None:
            return {
                "error": "PaddleOCR not available",
                "model": "PaddleOCR-Simple",
                "device": str(self._device)
            }
            
        logger.info(f"Running Simple PaddleOCR benchmark on {self._device} with {iterations} iterations")
        
        # Warm up
        self.process_image(test_image)
        
        # Run benchmark
        times = []
        for i in range(iterations):
            result = self.process_image(test_image)
            times.append(result["processing_time"])
            
        return {
            "model": "PaddleOCR-Simple",
            "device": str(self._device),
            "iterations": iterations,
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "fps": 1.0 / np.mean(times),
            "device_info": self.get_device_info()
        }