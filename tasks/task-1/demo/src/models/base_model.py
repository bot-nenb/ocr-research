"""Base OCR model class with device detection and management."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class BaseOCRModel(ABC):
    """Abstract base class for OCR models with device management."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        languages: List[str] = None,
        gpu_memory_limit: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize base OCR model.
        
        Args:
            model_name: Name of the OCR model
            device: Device to use ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            languages: List of languages to support
            gpu_memory_limit: Maximum GPU memory to use (in GB)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.languages = languages or ["en"]
        self.gpu_memory_limit = gpu_memory_limit
        self._model = None
        self._device = self._setup_device(device)
        self.stats = {
            "total_images": 0,
            "total_time": 0.0,
            "errors": 0,
            "device": str(self._device)
        }
        
        logger.info(f"Initializing {model_name} on device: {self._device}")
        
    def _setup_device(self, device: str) -> str:
        """
        Setup and validate device selection.
        
        Args:
            device: Requested device ('auto', 'cpu', 'cuda', etc.)
            
        Returns:
            Selected device string
        """
        if device == "auto":
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                device = "cpu"
                logger.info("CUDA not available. Using CPU")
                
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning(f"CUDA requested but not available. Falling back to CPU")
                device = "cpu"
            else:
                # Validate specific GPU if index provided
                if ":" in device:
                    gpu_idx = int(device.split(":")[1])
                    if gpu_idx >= torch.cuda.device_count():
                        logger.warning(f"GPU {gpu_idx} not found. Using default GPU")
                        device = "cuda:0"
                        
        return device
    
    @abstractmethod
    def initialize_model(self):
        """Initialize the actual OCR model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def process_image(
        self, 
        image_path: Union[str, Path], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single image and return OCR results.
        
        Args:
            image_path: Path to the image file
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing OCR results with keys:
                - 'text': Extracted text
                - 'boxes': Bounding boxes
                - 'confidence': Confidence scores
                - 'processing_time': Time taken
        """
        pass
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing (if supported)
            **kwargs: Additional processing parameters
            
        Returns:
            List of OCR results for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, **kwargs)
                results.append(result)
                self.stats["total_images"] += 1
                self.stats["total_time"] += result.get("processing_time", 0)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                self.stats["errors"] += 1
                results.append({
                    "text": "",
                    "error": str(e),
                    "processing_time": 0
                })
                
        return results
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        info = {
            "device": str(self._device),
            "device_type": "GPU" if "cuda" in str(self._device) else "CPU"
        }
        
        if "cuda" in str(self._device):
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
            })
            
        return info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        if stats["total_images"] > 0:
            stats["avg_time_per_image"] = stats["total_time"] / stats["total_images"]
            stats["images_per_minute"] = 60 / stats["avg_time_per_image"] if stats["avg_time_per_image"] > 0 else 0
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "total_images": 0,
            "total_time": 0.0,
            "errors": 0,
            "device": str(self._device)
        }
        
    def cleanup(self):
        """Clean up resources and clear GPU memory if applicable."""
        if "cuda" in str(self._device):
            torch.cuda.empty_cache()
        logger.info(f"Cleaned up {self.model_name} resources")
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()