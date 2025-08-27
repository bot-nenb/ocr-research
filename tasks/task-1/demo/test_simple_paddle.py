#!/usr/bin/env python3
"""Test the simple PaddleOCR model."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.paddleocr_simple import SimplePaddleOCRModel

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
    text = "Hello World! This is a test image for PaddleOCR."
    # Use default font
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
    test_image_path = Path("test_paddle_image.png")
    image.save(test_image_path)
    logger.info(f"Created test image: {test_image_path}")
    
    return test_image_path


def test_simple_paddleocr():
    """Test the simple PaddleOCR model."""
    logger.info("=" * 50)
    logger.info("Testing Simple PaddleOCR...")
    logger.info("=" * 50)
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Initialize model
        model = SimplePaddleOCRModel(device="cpu")
        
        # Get device info
        device_info = model.get_device_info()
        logger.info(f"Device info: {device_info}")
        
        # Process image
        result = model.process_image(test_image)
        
        # Display results
        if "error" not in result:
            logger.info(f"Detected text: {result['text']}")
            logger.info(f"Number of detections: {result['num_detections']}")
            logger.info(f"Average confidence: {result['avg_confidence']:.3f}")
            logger.info(f"Processing time: {result['processing_time']:.3f} seconds")
            
            # Run quick benchmark if successful
            if result['num_detections'] > 0:
                logger.info("\nRunning quick benchmark (3 iterations)...")
                benchmark = model.benchmark(test_image, iterations=3)
                if "error" not in benchmark:
                    logger.info(f"Average processing time: {benchmark['avg_time']:.3f} seconds")
                    logger.info(f"FPS: {benchmark['fps']:.2f}")
        else:
            logger.warning(f"PaddleOCR error: {result['error']}")
        
        # Get stats
        stats = model.get_stats()
        logger.info(f"\nModel stats: {stats}")
        
        # Cleanup
        model.cleanup()
        test_image.unlink()
        
        if "error" not in result:
            logger.info("✅ Simple PaddleOCR test completed successfully!")
            return True
        else:
            logger.info("⚠️ Simple PaddleOCR has compatibility issues but wrapper works.")
            return False
        
    except Exception as e:
        logger.error(f"❌ Simple PaddleOCR test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("Starting Simple PaddleOCR test...")
    
    success = test_simple_paddleocr()
    
    if success:
        logger.info("🎉 Simple PaddleOCR is working correctly!")
        return 0
    else:
        logger.info("⚠️ Simple PaddleOCR has issues but error handling works.")
        return 1


if __name__ == "__main__":
    sys.exit(main())