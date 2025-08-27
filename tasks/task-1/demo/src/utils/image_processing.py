"""
Image Processing Utilities for OCR Visualization

This module provides utilities for creating OCR overlay visualizations,
including bounding box drawing and text overlay on images.
"""

import base64
import io
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional
import colorsys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class OCRVisualizer:
    """Creates visualizations of OCR results overlaid on images."""
    
    def __init__(self, font_path: Optional[str] = None, font_size: int = 20):
        """
        Initialize the OCR visualizer.
        
        Args:
            font_path: Path to font file for text rendering
            font_size: Size of font for text rendering
        """
        self.font_size = font_size
        self.logger = logging.getLogger(__name__)
        
        # Try to load a font
        try:
            if font_path and Path(font_path).exists():
                self.font = ImageFont.truetype(font_path, font_size)
            else:
                # Try to load default fonts
                try:
                    self.font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                    except:
                        self.font = ImageFont.load_default()
        except Exception as e:
            self.logger.warning(f"Could not load font: {e}. Using default font.")
            self.font = ImageFont.load_default()
    
    def generate_colors(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """
        Generate visually distinct colors for bounding boxes.
        
        Args:
            n_colors: Number of colors to generate
            
        Returns:
            List of RGB color tuples
        """
        colors = []
        for i in range(n_colors):
            # Use HSV color space for better color distribution
            hue = i / n_colors
            saturation = 0.8
            value = 0.9
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to 0-255 range
            rgb = tuple(int(c * 255) for c in rgb)
            colors.append(rgb)
        
        return colors
    
    def draw_bounding_boxes(self, 
                           image: Union[np.ndarray, str, Path],
                           bounding_boxes: List[List],
                           text_lines: List[str],
                           confidences: List[float],
                           show_confidence: bool = True,
                           box_thickness: int = 2,
                           text_color: Tuple[int, int, int] = (255, 255, 255),
                           box_alpha: float = 0.3) -> Image.Image:
        """
        Draw OCR bounding boxes and text on an image.
        
        Args:
            image: Input image (numpy array, file path, or PIL Image)
            bounding_boxes: List of bounding box coordinates from OCR
            text_lines: List of detected text lines
            confidences: List of confidence scores
            show_confidence: Whether to show confidence scores
            box_thickness: Thickness of bounding box lines
            text_color: Color of text labels
            box_alpha: Transparency of bounding boxes
            
        Returns:
            PIL Image with OCR overlay
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert from BGR to RGB if it's a CV2 image
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Create drawing context
        draw = ImageDraw.Draw(pil_image)
        
        # Generate colors for bounding boxes
        colors = self.generate_colors(len(bounding_boxes))
        
        for i, (bbox, text, confidence) in enumerate(zip(bounding_boxes, text_lines, confidences)):
            if not bbox:  # Skip empty bounding boxes
                continue
                
            try:
                # Convert bounding box to rectangle coordinates
                if len(bbox) == 4 and all(isinstance(coord, (list, tuple)) for coord in bbox):
                    # EasyOCR format: list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    points = [(int(point[0]), int(point[1])) for point in bbox]
                    # Get bounding rectangle
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Draw polygon for exact bounding box
                    draw.polygon(points, outline=colors[i % len(colors)], width=box_thickness)
                    
                elif len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                    # Simple rectangle format: [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    draw.rectangle([x_min, y_min, x_max, y_max], 
                                 outline=colors[i % len(colors)], width=box_thickness)
                else:
                    self.logger.warning(f"Unsupported bounding box format: {bbox}")
                    continue
                
                # Prepare text label
                if show_confidence:
                    label = f"{text} ({confidence:.2f})"
                else:
                    label = text
                
                # Limit label length to prevent overcrowding
                if len(label) > 30:
                    label = label[:27] + "..."
                
                # Calculate text position (above bounding box)
                text_x = x_min
                text_y = max(0, y_min - self.font_size - 5)
                
                # Draw text background for better visibility
                try:
                    bbox_text = draw.textbbox((text_x, text_y), label, font=self.font)
                    text_bg_coords = [bbox_text[0] - 2, bbox_text[1] - 2, 
                                    bbox_text[2] + 2, bbox_text[3] + 2]
                    draw.rectangle(text_bg_coords, fill=(0, 0, 0, 180))
                except AttributeError:
                    # Fallback for older PIL versions
                    text_width, text_height = draw.textsize(label, font=self.font)
                    text_bg_coords = [text_x - 2, text_y - 2, 
                                    text_x + text_width + 2, text_y + text_height + 2]
                    draw.rectangle(text_bg_coords, fill=(0, 0, 0, 180))
                
                # Draw text
                draw.text((text_x, text_y), label, fill=text_color, font=self.font)
                
            except Exception as e:
                self.logger.warning(f"Error drawing bounding box {i}: {e}")
                continue
        
        return pil_image
    
    def create_comparison_image(self,
                              original_image: Union[np.ndarray, str, Path],
                              ocr_overlay: Image.Image,
                              side_by_side: bool = True) -> Image.Image:
        """
        Create a comparison image showing original and OCR overlay.
        
        Args:
            original_image: Original image
            ocr_overlay: Image with OCR overlay
            side_by_side: If True, show side by side; if False, overlay
            
        Returns:
            Combined comparison image
        """
        # Load original image
        if isinstance(original_image, (str, Path)):
            original_pil = Image.open(original_image).convert('RGB')
        elif isinstance(original_image, np.ndarray):
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            else:
                original_rgb = original_image
            original_pil = Image.fromarray(original_rgb)
        else:
            original_pil = original_image.convert('RGB')
        
        if side_by_side:
            # Create side-by-side comparison
            width1, height1 = original_pil.size
            width2, height2 = ocr_overlay.size
            
            # Scale images to same height
            target_height = max(height1, height2)
            scale1 = target_height / height1
            scale2 = target_height / height2
            
            new_width1 = int(width1 * scale1)
            new_width2 = int(width2 * scale2)
            
            original_resized = original_pil.resize((new_width1, target_height), Image.Resampling.LANCZOS)
            ocr_resized = ocr_overlay.resize((new_width2, target_height), Image.Resampling.LANCZOS)
            
            # Create combined image
            total_width = new_width1 + new_width2
            combined = Image.new('RGB', (total_width, target_height))
            combined.paste(original_resized, (0, 0))
            combined.paste(ocr_resized, (new_width1, 0))
            
            # Add labels
            draw = ImageDraw.Draw(combined)
            draw.text((10, 10), "Original", fill=(255, 255, 255), font=self.font)
            draw.text((new_width1 + 10, 10), "OCR Overlay", fill=(255, 255, 255), font=self.font)
            
            return combined
        else:
            # Simple overlay
            return ocr_overlay
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string for HTML embedding.
        
        Args:
            image: PIL Image
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    
    def save_visualization(self,
                          image: Union[np.ndarray, str, Path],
                          bounding_boxes: List[List],
                          text_lines: List[str],
                          confidences: List[float],
                          output_path: Union[str, Path],
                          show_confidence: bool = True) -> str:
        """
        Create and save OCR visualization.
        
        Args:
            image: Input image
            bounding_boxes: OCR bounding boxes
            text_lines: OCR text lines
            confidences: OCR confidence scores
            output_path: Path to save visualization
            show_confidence: Whether to show confidence scores
            
        Returns:
            Path to saved visualization
        """
        # Create visualization
        overlay_image = self.draw_bounding_boxes(
            image, bounding_boxes, text_lines, confidences, show_confidence
        )
        
        # Save image
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_image.save(output_path)
        
        self.logger.info(f"OCR visualization saved to {output_path}")
        return str(output_path)


class TextDiffHighlighter:
    """Creates highlighted text comparisons between OCR and ground truth."""
    
    def __init__(self):
        """Initialize text diff highlighter."""
        self.logger = logging.getLogger(__name__)
    
    def compute_word_diff(self, text1: str, text2: str) -> Tuple[List[str], List[str]]:
        """
        Compute word-level differences between two texts.
        
        Args:
            text1: Reference text
            text2: Comparison text
            
        Returns:
            Tuple of (highlighted_text1, highlighted_text2) with HTML markup
        """
        import difflib
        
        words1 = text1.split()
        words2 = text2.split()
        
        # Compute sequence matcher
        matcher = difflib.SequenceMatcher(None, words1, words2)
        
        highlighted1 = []
        highlighted2 = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Same words in both
                highlighted1.extend(words1[i1:i2])
                highlighted2.extend(words2[j1:j2])
            elif tag == 'delete':
                # Words only in text1 (deletions)
                for word in words1[i1:i2]:
                    highlighted1.append(f'<span class="diff-deleted">{word}</span>')
            elif tag == 'insert':
                # Words only in text2 (insertions)
                for word in words2[j1:j2]:
                    highlighted2.append(f'<span class="diff-inserted">{word}</span>')
            elif tag == 'replace':
                # Different words
                for word in words1[i1:i2]:
                    highlighted1.append(f'<span class="diff-changed">{word}</span>')
                for word in words2[j1:j2]:
                    highlighted2.append(f'<span class="diff-changed">{word}</span>')
        
        return ' '.join(highlighted1), ' '.join(highlighted2)
    
    def compute_char_diff(self, text1: str, text2: str) -> Tuple[str, str]:
        """
        Compute character-level differences between two texts.
        
        Args:
            text1: Reference text
            text2: Comparison text
            
        Returns:
            Tuple of (highlighted_text1, highlighted_text2) with HTML markup
        """
        import difflib
        
        matcher = difflib.SequenceMatcher(None, text1, text2)
        
        highlighted1 = []
        highlighted2 = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                highlighted1.append(text1[i1:i2])
                highlighted2.append(text2[j1:j2])
            elif tag == 'delete':
                highlighted1.append(f'<span class="diff-deleted">{text1[i1:i2]}</span>')
            elif tag == 'insert':
                highlighted2.append(f'<span class="diff-inserted">{text2[j1:j2]}</span>')
            elif tag == 'replace':
                highlighted1.append(f'<span class="diff-changed">{text1[i1:i2]}</span>')
                highlighted2.append(f'<span class="diff-changed">{text2[j1:j2]}</span>')
        
        return ''.join(highlighted1), ''.join(highlighted2)
    
    def create_diff_html(self, 
                        ground_truth: str, 
                        ocr_text: str,
                        doc_id: str = "",
                        include_stats: bool = True) -> str:
        """
        Create HTML diff comparison between ground truth and OCR text.
        
        Args:
            ground_truth: Ground truth text
            ocr_text: OCR output text
            doc_id: Document identifier
            include_stats: Whether to include difference statistics
            
        Returns:
            HTML string with side-by-side diff comparison
        """
        # Compute word-level diff
        highlighted_gt, highlighted_ocr = self.compute_word_diff(ground_truth, ocr_text)
        
        # Calculate statistics if requested
        stats_html = ""
        if include_stats:
            import Levenshtein
            
            # Calculate edit distance and similarity
            edit_distance = Levenshtein.distance(ground_truth, ocr_text)
            similarity = 1 - (edit_distance / max(len(ground_truth), len(ocr_text), 1))
            
            words_gt = len(ground_truth.split())
            words_ocr = len(ocr_text.split())
            
            stats_html = f"""
            <div class="diff-stats">
                <h4>Comparison Statistics</h4>
                <p><strong>Character Edit Distance:</strong> {edit_distance}</p>
                <p><strong>Similarity:</strong> {similarity:.1%}</p>
                <p><strong>Ground Truth Words:</strong> {words_gt}</p>
                <p><strong>OCR Words:</strong> {words_ocr}</p>
            </div>
            """
        
        # Create HTML
        html = f"""
        <div class="text-diff-container">
            <h3>Text Comparison{' - ' + doc_id if doc_id else ''}</h3>
            {stats_html}
            <div class="diff-comparison">
                <div class="diff-column">
                    <h4>Ground Truth</h4>
                    <div class="diff-text">{highlighted_gt}</div>
                </div>
                <div class="diff-column">
                    <h4>OCR Result</h4>
                    <div class="diff-text">{highlighted_ocr}</div>
                </div>
            </div>
        </div>
        """
        
        return html


def create_ocr_visualization_report(processing_results: List,
                                  ground_truth_data: dict,
                                  output_dir: Path,
                                  max_documents: int = 10) -> dict:
    """
    Create OCR visualization report for multiple documents.
    
    Args:
        processing_results: List of ProcessingResult objects
        ground_truth_data: Dictionary of ground truth text
        output_dir: Output directory for visualizations
        max_documents: Maximum number of documents to visualize
        
    Returns:
        Dictionary with paths to generated visualizations
    """
    visualizer = OCRVisualizer()
    diff_highlighter = TextDiffHighlighter()
    
    visualization_paths = {}
    text_diffs = {}
    
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Process successful results
    successful_results = [r for r in processing_results if r.success and r.bounding_boxes]
    
    # Sort by confidence score (worst first for better debugging)
    successful_results.sort(key=lambda x: x.confidence_score)
    
    for i, result in enumerate(successful_results[:max_documents]):
        try:
            # Create OCR overlay visualization
            if result.image_path and Path(result.image_path).exists():
                overlay_path = vis_dir / f"{result.doc_id}_ocr_overlay.png"
                
                visualizer.save_visualization(
                    image=result.image_path,
                    bounding_boxes=result.bounding_boxes,
                    text_lines=result.text_lines,
                    confidences=result.line_confidences,
                    output_path=overlay_path,
                    show_confidence=True
                )
                
                visualization_paths[result.doc_id] = {
                    'overlay_path': str(overlay_path),
                    'original_path': result.image_path
                }
            
            # Create text diff if ground truth available
            if result.doc_id in ground_truth_data:
                gt_text = ground_truth_data[result.doc_id]
                diff_html = diff_highlighter.create_diff_html(
                    ground_truth=gt_text,
                    ocr_text=result.ocr_text,
                    doc_id=result.doc_id,
                    include_stats=True
                )
                text_diffs[result.doc_id] = diff_html
            
        except Exception as e:
            logging.error(f"Error creating visualization for {result.doc_id}: {e}")
            continue
    
    return {
        'visualization_paths': visualization_paths,
        'text_diffs': text_diffs,
        'output_directory': str(vis_dir)
    }