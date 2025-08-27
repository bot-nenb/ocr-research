"""EasyOCR model wrapper with CPU/GPU flexibility."""

import easyocr
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import time
import cv2
from .base_model import BaseOCRModel

logger = logging.getLogger(__name__)


class EasyOCRModel(BaseOCRModel):
    """EasyOCR wrapper with device-aware processing."""
    
    def __init__(
        self,
        device: str = "auto",
        languages: List[str] = None,
        gpu_memory_limit: Optional[float] = None,
        model_storage_directory: Optional[str] = None,
        download_enabled: bool = True,
        detector: bool = True,
        recognizer: bool = True,
        verbose: bool = False,
        quantize: bool = False,
        cudnn_benchmark: bool = False,
        **kwargs
    ):
        """
        Initialize EasyOCR model.
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda')
            languages: List of language codes (default: ['en'])
            gpu_memory_limit: Maximum GPU memory to use in GB
            model_storage_directory: Directory to store downloaded models
            download_enabled: Whether to download models if not present
            detector: Enable text detection
            recognizer: Enable text recognition
            verbose: Verbose output
            quantize: Use quantized models for faster CPU inference
            cudnn_benchmark: Enable cuDNN benchmark for GPU
            **kwargs: Additional parameters
        """
        super().__init__(
            model_name="EasyOCR",
            device=device,
            languages=languages,
            gpu_memory_limit=gpu_memory_limit,
            **kwargs
        )
        
        self.model_storage_directory = model_storage_directory
        self.download_enabled = download_enabled
        self.detector = detector
        self.recognizer = recognizer
        self.verbose = verbose
        self.quantize = quantize
        self.cudnn_benchmark = cudnn_benchmark
        
        # Initialize the model
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize EasyOCR reader with appropriate device settings."""
        try:
            # Determine if GPU should be used
            gpu = "cuda" in str(self._device)
            
            # Set up additional parameters
            kwargs = {
                "gpu": gpu,
                "verbose": self.verbose,
                "detector": self.detector,
                "recognizer": self.recognizer,
                "download_enabled": self.download_enabled,
            }
            
            # Add optional parameters
            if self.model_storage_directory:
                kwargs["model_storage_directory"] = self.model_storage_directory
                
            if self.quantize and not gpu:
                kwargs["quantize"] = True
                logger.info("Using quantized models for faster CPU inference")
                
            if gpu and self.cudnn_benchmark:
                kwargs["cudnn_benchmark"] = True
                logger.info("Enabled cuDNN benchmark for GPU optimization")
            
            # Create the reader
            self._model = easyocr.Reader(
                lang_list=self.languages,
                **kwargs
            )
            
            logger.info(f"EasyOCR initialized successfully on {self._device}")
            logger.info(f"Languages: {self.languages}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            # Try fallback to CPU if GPU fails
            if "cuda" in str(self._device):
                logger.warning("Falling back to CPU mode")
                self._device = "cpu"
                kwargs["gpu"] = False
                self._model = easyocr.Reader(
                    lang_list=self.languages,
                    **kwargs
                )
            else:
                raise
                
    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        detail: int = 1,
        decoder: str = 'greedy',
        beamWidth: int = 5,
        batch_size: int = 1,
        paragraph: bool = False,
        width_ths: float = 0.5,
        height_ths: float = 0.5,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single image with EasyOCR.
        
        Args:
            image_input: Path to the image file or numpy array
            detail: Level of detail (0=simple, 1=with polygon)
            decoder: Text decoder ('greedy' or 'beamSearch')
            beamWidth: Width for beam search decoder
            batch_size: Batch size for recognition
            paragraph: Combine results into paragraphs
            width_ths: Width threshold for text detection
            height_ths: Height threshold for text detection
            text_threshold: Text confidence threshold
            low_text: Low text score threshold
            link_threshold: Link threshold
            canvas_size: Maximum image dimension
            mag_ratio: Image magnification ratio
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        
        try:
            # Handle both file paths and numpy arrays
            if isinstance(image_input, (str, Path)):
                # Convert path to string
                image_path = str(Path(image_input).resolve())
                
                # Read image to check dimensions
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Cannot read image: {image_path}")
                # Use path for EasyOCR (it can be more efficient)
                ocr_input = image_path
            elif isinstance(image_input, np.ndarray):
                # Already have numpy array
                image = image_input
                # EasyOCR can also accept numpy arrays directly
                ocr_input = image
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
                
            height, width = image.shape[:2]
            
            # Adjust parameters based on device
            if "cpu" in str(self._device):
                # Optimize for CPU processing
                if width > 1920 or height > 1920:
                    # Resize large images for faster CPU processing
                    scale = min(1920/width, 1920/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height} for CPU processing")
                    
                # Use smaller canvas size for CPU
                canvas_size = min(canvas_size, 1920)
                
            # Perform OCR
            results = self._model.readtext(
                ocr_input,
                detail=detail,
                decoder=decoder,
                beamWidth=beamWidth,
                batch_size=batch_size,
                paragraph=paragraph,
                width_ths=width_ths,
                height_ths=height_ths,
                text_threshold=text_threshold,
                low_text=low_text,
                link_threshold=link_threshold,
                canvas_size=canvas_size,
                mag_ratio=mag_ratio,
                **kwargs
            )
            
            # Parse results
            text_lines = []
            boxes = []
            confidences = []
            
            for detection in results:
                if detail == 0:
                    # Simple format: (text, confidence)
                    text_lines.append(detection[0])
                    confidences.append(detection[1])
                else:
                    # Detailed format: (bbox, text, confidence)
                    boxes.append(detection[0])
                    text_lines.append(detection[1])
                    confidences.append(detection[2])
            
            processing_time = time.time() - start_time
            
            return {
                "text": "\n".join(text_lines),
                "text_lines": text_lines,
                "boxes": boxes,
                "confidences": confidences,
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "num_detections": len(text_lines),
                "processing_time": processing_time,
                "image_size": (width, height),
                "model": "EasyOCR",
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
                "model": "EasyOCR",
                "device": str(self._device)
            }
            
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images with optional batching.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            **kwargs: Additional parameters for process_image
            
        Returns:
            List of OCR results
        """
        # Determine optimal batch size based on device
        if batch_size is None:
            if "cuda" in str(self._device):
                batch_size = 4  # GPU can handle larger batches
            else:
                batch_size = 1  # CPU processes one at a time
                
        # Update kwargs with batch size
        kwargs['batch_size'] = batch_size
        
        # Process images
        results = super().process_batch(image_paths, batch_size=batch_size, **kwargs)
        
        return results
    
    def benchmark(self, test_image: Union[str, Path], iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark the model performance.
        
        Args:
            test_image: Path to test image
            iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running EasyOCR benchmark on {self._device} with {iterations} iterations")
        
        # Warm up
        self.process_image(test_image)
        
        # Run benchmark
        times = []
        for i in range(iterations):
            result = self.process_image(test_image)
            times.append(result["processing_time"])
            
        return {
            "model": "EasyOCR",
            "device": str(self._device),
            "iterations": iterations,
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "fps": 1.0 / np.mean(times),
            "device_info": self.get_device_info()
        }
        
    def set_languages(self, languages: List[str]):
        """
        Change the languages for OCR.
        
        Args:
            languages: New list of language codes
        """
        if set(languages) != set(self.languages):
            logger.info(f"Changing languages from {self.languages} to {languages}")
            self.languages = languages
            self.initialize_model()
            
    def optimize_for_device(self):
        """Optimize model settings for current device."""
        if "cuda" in str(self._device):
            # GPU optimizations
            if hasattr(self._model, 'detector') and self._model.detector:
                self._model.detector.eval()
            if hasattr(self._model, 'recognizer') and self._model.recognizer:
                self._model.recognizer.eval()
            logger.info("Applied GPU optimizations")
        else:
            # CPU optimizations
            if self.quantize:
                logger.info("Using quantized models for CPU optimization")
            # Could add more CPU-specific optimizations here
            
    def get_supported_languages(self) -> List[str]:
        """Get list of all supported languages."""
        # EasyOCR supports 80+ languages
        return [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi', 'id', 
            'ms', 'fa', 'ar', 'hi', 'bn', 'ta', 'te', 'kn', 'mr', 
            'ne', 'ur', 'ru', 'bg', 'uk', 'be', 'sr', 'mk', 'et', 
            'lv', 'lt', 'pl', 'cs', 'sk', 'sl', 'hr', 'bs', 'sq', 
            'hu', 'ro', 'tr', 'el', 'he', 'de', 'fr', 'es', 'pt', 
            'it', 'nl', 'no', 'sv', 'da', 'fi', 'is', 'ga', 'cy'
        ]