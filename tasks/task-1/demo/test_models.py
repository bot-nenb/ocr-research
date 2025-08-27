#!/usr/bin/env python3
"""Test script to verify EasyOCR and PaddleOCR models are working correctly."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.easyocr_model import EasyOCRModel
from models.paddleocr_model import PaddleOCRModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image():
    """Create a simple test image with text."""
    # Create image with white background
    width, height = 800, 200
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add text
    text = "Hello World! This is a test image for OCR processing."
    # Use default font (may vary by system)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Get text bbox and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw text
    draw.text(position, text, fill='black', font=font)
    
    # Save image
    test_image_path = Path("test_image.png")
    image.save(test_image_path)
    logger.info(f"Created test image: {test_image_path}")
    
    return test_image_path


def test_easyocr():
    """Test EasyOCR model."""
    logger.info("=" * 50)
    logger.info("Testing EasyOCR...")
    logger.info("=" * 50)
    
    try:
        # Initialize model
        model = EasyOCRModel(device="cpu", languages=["en"])
        
        # Get device info
        device_info = model.get_device_info()
        logger.info(f"Device info: {device_info}")
        
        # Create test image
        test_image = create_test_image()
        
        # Process image
        result = model.process_image(test_image)
        
        # Display results
        logger.info(f"Detected text: {result['text']}")
        logger.info(f"Number of detections: {result['num_detections']}")
        logger.info(f"Average confidence: {result['avg_confidence']:.3f}")
        logger.info(f"Processing time: {result['processing_time']:.3f} seconds")
        
        # Run quick benchmark
        if result['num_detections'] > 0:
            logger.info("\nRunning quick benchmark (3 iterations)...")
            benchmark = model.benchmark(test_image, iterations=3)
            logger.info(f"Average processing time: {benchmark['avg_time']:.3f} seconds")
            logger.info(f"FPS: {benchmark['fps']:.2f}")
        
        # Get stats
        stats = model.get_stats()
        logger.info(f"\nModel stats: {stats}")
        
        # Cleanup
        model.cleanup()
        test_image.unlink()
        
        logger.info("✅ EasyOCR test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ EasyOCR test failed: {e}")
        return False


def test_paddleocr():
    """Test PaddleOCR model."""
    logger.info("=" * 50)
    logger.info("Testing PaddleOCR...")
    logger.info("=" * 50)
    
    try:
        # Initialize model
        model = PaddleOCRModel(device="cpu", languages=["en"])
        
        # Get device info
        device_info = model.get_device_info()
        logger.info(f"Device info: {device_info}")
        
        # Create test image
        test_image = create_test_image()
        
        # Process image
        result = model.process_image(test_image)
        
        # Display results
        logger.info(f"Detected text: {result['text']}")
        logger.info(f"Number of detections: {result['num_detections']}")
        logger.info(f"Average confidence: {result['avg_confidence']:.3f}")
        logger.info(f"Processing time: {result['processing_time']:.3f} seconds")
        
        # Run quick benchmark
        if result['num_detections'] > 0:
            logger.info("\nRunning quick benchmark (3 iterations)...")
            benchmark = model.benchmark(test_image, iterations=3)
            logger.info(f"Average processing time: {benchmark['avg_time']:.3f} seconds")
            logger.info(f"FPS: {benchmark['fps']:.2f}")
        
        # Get stats
        stats = model.get_stats()
        logger.info(f"\nModel stats: {stats}")
        
        # Cleanup
        model.cleanup()
        test_image.unlink()
        
        logger.info("✅ PaddleOCR test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ PaddleOCR test failed: {e}")
        return False


def compare_models():
    """Compare both models on the same image."""
    logger.info("=" * 50)
    logger.info("Comparing EasyOCR vs PaddleOCR...")
    logger.info("=" * 50)
    
    # Create test image
    test_image = create_test_image()
    
    results = {}
    
    # Test EasyOCR
    try:
        logger.info("\nProcessing with EasyOCR...")
        easyocr_model = EasyOCRModel(device="cpu", languages=["en"])
        easyocr_result = easyocr_model.process_image(test_image)
        results["EasyOCR"] = {
            "text": easyocr_result["text"],
            "confidence": easyocr_result["avg_confidence"],
            "time": easyocr_result["processing_time"],
            "detections": easyocr_result["num_detections"]
        }
        easyocr_model.cleanup()
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
        results["EasyOCR"] = {"error": str(e)}
    
    # Test PaddleOCR
    try:
        logger.info("\nProcessing with PaddleOCR...")
        paddle_model = PaddleOCRModel(device="cpu", languages=["en"])
        paddle_result = paddle_model.process_image(test_image)
        results["PaddleOCR"] = {
            "text": paddle_result["text"],
            "confidence": paddle_result["avg_confidence"],
            "time": paddle_result["processing_time"],
            "detections": paddle_result["num_detections"]
        }
        paddle_model.cleanup()
    except Exception as e:
        logger.error(f"PaddleOCR failed: {e}")
        results["PaddleOCR"] = {"error": str(e)}
    
    # Display comparison
    logger.info("\n" + "=" * 50)
    logger.info("COMPARISON RESULTS:")
    logger.info("=" * 50)
    
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        if "error" in result:
            logger.info(f"  Error: {result['error']}")
        else:
            logger.info(f"  Text: {result['text']}")
            logger.info(f"  Confidence: {result['confidence']:.3f}")
            logger.info(f"  Processing time: {result['time']:.3f} seconds")
            logger.info(f"  Detections: {result['detections']}")
    
    # Cleanup
    test_image.unlink()
    
    logger.info("\n✅ Comparison completed!")


def main():
    """Main test function."""
    logger.info("Starting OCR model tests...")
    
    # Test individual models
    easyocr_success = test_easyocr()
    paddleocr_success = test_paddleocr()
    
    # Compare models
    if easyocr_success and paddleocr_success:
        compare_models()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"EasyOCR: {'✅ PASSED' if easyocr_success else '❌ FAILED'}")
    logger.info(f"PaddleOCR: {'✅ PASSED' if paddleocr_success else '❌ FAILED'}")
    
    if easyocr_success and paddleocr_success:
        logger.info("\n🎉 All tests passed! Both models are working correctly.")
        return 0
    else:
        logger.info("\n⚠️ Some tests failed. Please check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())