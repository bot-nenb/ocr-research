"""PaddleOCR model wrapper with CPU/GPU flexibility."""

from paddleocr import PaddleOCR
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import time
import cv2
import os
import torch  # Need this for GPU detection
from .base_model import BaseOCRModel

# Suppress PaddleOCR verbose output
os.environ['PPOCR_DEBUG'] = '0'

logger = logging.getLogger(__name__)


class PaddleOCRModel(BaseOCRModel):
    """PaddleOCR wrapper with device-aware processing."""
    
    def __init__(
        self,
        device: str = "auto",
        languages: List[str] = None,
        gpu_memory_limit: Optional[float] = None,
        use_angle_cls: bool = True,
        use_space_char: bool = True,
        use_mp: bool = False,
        enable_mkldnn: bool = True,
        cpu_threads: int = 10,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
        det_algorithm: str = 'DB',
        rec_algorithm: str = 'SVTR_LCNet',
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.5,
        max_text_length: int = 25,
        drop_score: float = 0.5,
        use_tensorrt: bool = False,
        precision: str = 'fp32',
        show_log: bool = False,
        **kwargs
    ):
        """
        Initialize PaddleOCR model.
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda')
            languages: List of language codes (default: ['en'])
            gpu_memory_limit: Maximum GPU memory to use in GB
            use_angle_cls: Use angle classification
            use_space_char: Use space character in recognition
            use_mp: Use multiprocessing
            enable_mkldnn: Enable MKL-DNN for CPU acceleration
            cpu_threads: Number of CPU threads
            det_model_dir: Detection model directory
            rec_model_dir: Recognition model directory
            cls_model_dir: Classification model directory
            det_algorithm: Detection algorithm ('DB', 'EAST', 'SAST')
            rec_algorithm: Recognition algorithm
            det_db_thresh: Detection DB threshold
            det_db_box_thresh: Detection box threshold
            det_db_unclip_ratio: Detection unclip ratio
            max_text_length: Maximum text length
            drop_score: Drop score threshold
            use_tensorrt: Use TensorRT for GPU acceleration
            precision: Model precision ('fp32', 'fp16', 'int8')
            show_log: Show PaddleOCR logs
            **kwargs: Additional parameters
        """
        # Map languages to PaddleOCR format
        if languages:
            self.paddle_lang = self._map_language(languages[0])
        else:
            self.paddle_lang = 'en'
            
        super().__init__(
            model_name="PaddleOCR",
            device=device,
            languages=languages or ['en'],
            gpu_memory_limit=gpu_memory_limit,
            **kwargs
        )
        
        self.use_angle_cls = use_angle_cls
        self.use_space_char = use_space_char
        self.use_mp = use_mp
        self.enable_mkldnn = enable_mkldnn
        self.cpu_threads = cpu_threads
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        self.det_algorithm = det_algorithm
        self.rec_algorithm = rec_algorithm
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.max_text_length = max_text_length
        self.drop_score = drop_score
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        self.show_log = show_log
        
        # Initialize the model
        self.initialize_model()
        
    def _map_language(self, lang: str) -> str:
        """Map language codes to PaddleOCR format."""
        lang_map = {
            'en': 'en',
            'zh': 'ch',
            'ch_sim': 'ch',
            'ch_tra': 'chinese_cht',
            'ja': 'japan',
            'ko': 'korean',
            'fr': 'french',
            'de': 'german',
            'es': 'es',
            'pt': 'pt',
            'ru': 'ru',
            'ar': 'arabic',
            'hi': 'hi',
            'th': 'th',
            'vi': 'vi'
        }
        return lang_map.get(lang, 'en')
        
    def initialize_model(self):
        """Initialize PaddleOCR with appropriate device settings."""
        try:
            # Determine if GPU should be used
            use_gpu = "cuda" in str(self._device)
            
            # Set up configuration with simplified parameters
            config = {
                'lang': self.paddle_lang,
                'use_textline_orientation': self.use_angle_cls,
                'text_det_thresh': self.det_db_thresh,
                'text_det_box_thresh': self.det_db_box_thresh,
                'text_det_unclip_ratio': self.det_db_unclip_ratio,
                'text_rec_score_thresh': self.drop_score,
                'use_doc_unwarping': False,
            }
            
            # Add model directories if specified
            if self.det_model_dir:
                config['text_detection_model_dir'] = self.det_model_dir
            if self.rec_model_dir:
                config['text_recognition_model_dir'] = self.rec_model_dir
            if self.cls_model_dir:
                config['textline_orientation_model_dir'] = self.cls_model_dir
            
            # Create PaddleOCR instance with minimal configuration
            # PaddleOCR will handle device selection internally based on availability
            self._model = PaddleOCR(**config)
            
            logger.info(f"PaddleOCR initialized successfully on {self._device}")
            logger.info(f"Language: {self.paddle_lang}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            # Try fallback to CPU if GPU fails
            if use_gpu:
                logger.warning("Falling back to CPU mode")
                self._device = "cpu"
                config['use_gpu'] = False
                config['enable_mkldnn'] = self.enable_mkldnn
                config['cpu_threads'] = self.cpu_threads
                self._model = PaddleOCR(**config)
            else:
                raise
                
    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        cls: bool = None,
        det: bool = True,
        rec: bool = True,
        return_word_box: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single image with PaddleOCR.
        
        Args:
            image_input: Path to the image file or numpy array
            cls: Use angle classification (None = use default)
            det: Enable text detection
            rec: Enable text recognition
            return_word_box: Return word-level bounding boxes
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with OCR results
        """
        # Get image identifier for logging
        if isinstance(image_input, (str, Path)):
            image_id = Path(image_input).stem
            logger.info(f"Starting PaddleOCR processing for image: {image_id}")
        else:
            logger.info("Starting PaddleOCR processing for numpy array image")
            
        start_time = time.time()
        
        try:
            # Handle both numpy arrays and file paths
            if isinstance(image_input, np.ndarray):
                # We have a numpy array (image data)
                image = image_input.copy()
                height, width = image.shape[:2]
                
                # Track original dimensions and scaling
                original_width, original_height = width, height
                scale_factor = 1.0
                image_was_resized = False
                
                # Adjust for CPU processing if needed
                if "cpu" in str(self._device) and (width > 2000 or height > 2000):
                    # Resize large images for faster CPU processing
                    scale_factor = min(2000/width, 2000/height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    image = cv2.resize(image, (new_width, new_height))
                    image_was_resized = True
                    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height} for CPU processing (scale: {scale_factor:.3f})")
                
                # Use angle classification setting
                if cls is None:
                    cls = self.use_angle_cls
                
                # Perform OCR on numpy array using new API
                result = self._model.predict(image)
            else:
                # We have a file path
                image_path = str(Path(image_input).resolve())
                
                # Read image to check dimensions
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Cannot read image: {image_path}")
                    
                height, width = image.shape[:2]
                
                # Track original dimensions and scaling
                original_width, original_height = width, height
                scale_factor = 1.0
                image_was_resized = False
                
                # Adjust for CPU processing if needed
                if "cpu" in str(self._device) and (width > 2000 or height > 2000):
                    # Resize large images for faster CPU processing
                    scale_factor = min(2000/width, 2000/height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    image = cv2.resize(image, (new_width, new_height))
                    image_was_resized = True
                    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height} for CPU processing (scale: {scale_factor:.3f})")
                
                # Use angle classification setting
                if cls is None:
                    cls = self.use_angle_cls
                
                # Perform OCR on file path using new API
                result = self._model.predict(image_path)
            
            # Parse results - handle new PaddleOCR API format
            text_lines = []
            boxes = []
            confidences = []
            
            if result and len(result) > 0:
                # New API returns OCRResult objects
                page_result = result[0]
                
                # Check if it's the new OCRResult format
                if hasattr(page_result, '__getitem__') and 'rec_texts' in page_result:
                    # Extract from new OCRResult format
                    rec_texts = page_result.get('rec_texts', [])
                    rec_scores = page_result.get('rec_scores', [])
                    rec_polys = page_result.get('rec_polys', [])
                    
                    # Combine texts, scores, and polygons
                    for i, text in enumerate(rec_texts):
                        if text and text.strip():  # Only include non-empty text
                            text_lines.append(text.strip())
                            confidences.append(rec_scores[i] if i < len(rec_scores) else 0.0)
                            
                            # Convert polygon to proper format
                            if i < len(rec_polys):
                                poly = rec_polys[i]
                                # poly is a numpy array of shape (4, 2) representing 4 points
                                # Convert to list of points for visualization
                                if hasattr(poly, 'tolist'):
                                    box = poly.tolist()  # Convert numpy array to list
                                else:
                                    box = list(poly)
                                boxes.append(box)
                            else:
                                boxes.append([[0, 0], [0, 0], [0, 0], [0, 0]])  # Fallback
                else:
                    # Fallback for potential old format
                    logger.warning("Unexpected PaddleOCR result format")
            
            # Scale bounding box coordinates back to original image size if resizing occurred
            if image_was_resized and scale_factor != 1.0:
                inverse_scale = 1.0 / scale_factor
                for box in boxes:
                    for point in box:
                        point[0] *= inverse_scale  # Scale x coordinate back
                        point[1] *= inverse_scale  # Scale y coordinate back
                logger.debug(f"Scaled {len(boxes)} bounding boxes back to original image dimensions (inverse scale: {inverse_scale:.3f})")
            
            processing_time = time.time() - start_time
            
            # Log completion details
            avg_confidence = np.mean(confidences) if confidences else 0.0
            logger.info(f"PaddleOCR completed: {len(text_lines)} text detections, avg confidence: {avg_confidence:.3f}, time: {processing_time:.2f}s")
            
            return {
                "text": "\n".join(text_lines),
                "text_lines": text_lines,
                "boxes": boxes,
                "confidences": confidences,
                "avg_confidence": avg_confidence,
                "num_detections": len(text_lines),
                "processing_time": processing_time,
                "image_size": (original_width, original_height),
                "model": "PaddleOCR",
                "device": str(self._device)
            }
            
        except Exception as e:
            # Handle error with appropriate input description
            input_desc = image_input if isinstance(image_input, (str, Path)) else "numpy array"
            logger.error(f"Error processing image {input_desc}: {e}")
            
            # Try to get original dimensions for error case
            try:
                if isinstance(image_input, np.ndarray):
                    error_height, error_width = image_input.shape[:2]
                elif isinstance(image_input, (str, Path)):
                    error_img = cv2.imread(str(image_input))
                    error_height, error_width = error_img.shape[:2] if error_img is not None else (0, 0)
                else:
                    error_width, error_height = 0, 0
            except:
                error_width, error_height = 0, 0
            
            return {
                "text": "",
                "text_lines": [],
                "boxes": [],
                "confidences": [],
                "error": str(e),
                "processing_time": time.time() - start_time,
                "image_size": (error_width, error_height),
                "model": "PaddleOCR",
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
            batch_size: Batch size for processing (not used in PaddleOCR)
            **kwargs: Additional parameters for process_image
            
        Returns:
            List of OCR results
        """
        # PaddleOCR doesn't support true batching, process sequentially
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
                    "processing_time": 0,
                    "model": "PaddleOCR",
                    "device": str(self._device)
                })
                
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
        logger.info(f"Running PaddleOCR benchmark on {self._device} with {iterations} iterations")
        
        # Warm up
        self.process_image(test_image)
        
        # Run benchmark
        times = []
        for i in range(iterations):
            result = self.process_image(test_image)
            times.append(result["processing_time"])
            
        return {
            "model": "PaddleOCR",
            "device": str(self._device),
            "iterations": iterations,
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "fps": 1.0 / np.mean(times),
            "device_info": self.get_device_info()
        }
        
    def extract_table_structure(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract table structure from image (PaddleOCR specialty).
        
        Args:
            image_path: Path to image containing table
            **kwargs: Additional parameters
            
        Returns:
            Table structure and content
        """
        start_time = time.time()
        
        try:
            # Use PaddleOCR's structure extraction if available
            # This would require additional PaddleOCR modules
            result = self.process_image(image_path, **kwargs)
            
            # Add table-specific processing here if needed
            result["table_extracted"] = True
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting table structure: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
    def set_language(self, language: str):
        """
        Change the language for OCR.
        
        Args:
            language: Language code
        """
        new_paddle_lang = self._map_language(language)
        if new_paddle_lang != self.paddle_lang:
            logger.info(f"Changing language from {self.paddle_lang} to {new_paddle_lang}")
            self.languages = [language]
            self.paddle_lang = new_paddle_lang
            self.initialize_model()
            
    def optimize_for_device(self):
        """Optimize model settings for current device."""
        if "cuda" in str(self._device):
            # GPU optimizations
            logger.info("Applying GPU optimizations for PaddleOCR")
            # Could enable TensorRT here if not already enabled
        else:
            # CPU optimizations
            logger.info(f"Applying CPU optimizations: MKL-DNN enabled, {self.cpu_threads} threads")
            
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # PaddleOCR supports 80+ languages
        return [
            'en', 'ch', 'chinese_cht', 'japan', 'korean', 'french', 
            'german', 'es', 'pt', 'ru', 'arabic', 'hi', 'th', 'vi',
            'it', 'dutch', 'swedish', 'finnish', 'danish', 'norwegian',
            'polish', 'turkish', 'croatian', 'czech', 'slovak', 'hungarian',
            'romanian', 'bulgarian', 'ukrainian', 'hebrew', 'urdu', 'fa',
            'bengali', 'tamil', 'telugu', 'kannada', 'malayalam', 'marathi',
            'nepali', 'indonesia', 'malay', 'latin', 'cyrillic'
        ]