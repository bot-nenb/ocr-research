"""
FUNSD Dataset Loader and Preparation Utilities

This module handles downloading, loading, and preparing the FUNSD dataset
for OCR benchmarking. FUNSD (Form Understanding in Noisy Scanned Documents)
is a dataset for form understanding in noisy scanned documents.
"""

import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class FUNSDLoader:
    """Loads and prepares the FUNSD dataset for OCR evaluation."""
    
    FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    
    def __init__(self, data_dir: str = "data/funsd_subset"):
        """
        Initialize FUNSD dataset loader.
        
        Args:
            data_dir: Directory to store the dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_path = self.data_dir / "dataset"
        self.subset_path = self.data_dir / "subset_100"
        self.processed_path = self.data_dir / "processed"
        
    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download the FUNSD dataset if not already present.
        
        Args:
            force_download: Force re-download even if dataset exists
            
        Returns:
            True if download successful or dataset already exists
        """
        zip_path = self.data_dir / "funsd.zip"
        
        if self.dataset_path.exists() and not force_download:
            print(f"Dataset already exists at {self.dataset_path}")
            return True
            
        print(f"Downloading FUNSD dataset from {self.FUNSD_URL}")
        print("Note: FUNSD is publicly available for research purposes")
        
        try:
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"Download progress: {percent:.1f}%", end='\r')
            
            urlretrieve(self.FUNSD_URL, zip_path, reporthook=download_progress)
            print("\nDownload complete!")
            
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            zip_path.unlink()
            print("Dataset extracted successfully")
            
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return False
    
    def parse_annotations(self, annotation_path: Path) -> Dict:
        """
        Parse FUNSD JSON annotations.
        
        Args:
            annotation_path: Path to annotation JSON file
            
        Returns:
            Dictionary containing parsed annotations
        """
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        parsed = {
            'form': data.get('form', []),
            'text_lines': [],
            'words': [],
            'boxes': []
        }
        
        for item in data.get('form', []):
            text = item.get('text', '')
            box = item.get('box', [])
            words = item.get('words', [])
            
            if text and box:
                parsed['text_lines'].append(text)
                parsed['boxes'].append(box)
                
                for word_item in words:
                    word_text = word_item.get('text', '')
                    word_box = word_item.get('box', [])
                    if word_text and word_box:
                        parsed['words'].append({
                            'text': word_text,
                            'box': word_box
                        })
        
        return parsed
    
    def select_subset(self, num_documents: int = 100, seed: int = 42) -> List[Path]:
        """
        Select a diverse subset of documents for testing.
        
        Args:
            num_documents: Number of documents to select
            seed: Random seed for reproducible selection
            
        Returns:
            List of paths to selected documents
        """
        random.seed(seed)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Please download first.")
        
        all_images = []
        
        for split in ['training_data', 'testing_data']:
            split_path = self.dataset_path / split / 'images'
            if split_path.exists():
                images = list(split_path.glob('*.png'))
                all_images.extend(images)
        
        if len(all_images) < num_documents:
            print(f"Warning: Only {len(all_images)} images available, using all")
            selected = all_images
        else:
            selected = random.sample(all_images, num_documents)
        
        print(f"Selected {len(selected)} documents for subset")
        
        self.subset_path.mkdir(parents=True, exist_ok=True)
        subset_images_dir = self.subset_path / 'images'
        subset_annotations_dir = self.subset_path / 'annotations'
        subset_images_dir.mkdir(exist_ok=True)
        subset_annotations_dir.mkdir(exist_ok=True)
        
        for img_path in tqdm(selected, desc="Copying subset"):
            shutil.copy2(img_path, subset_images_dir / img_path.name)
            
            ann_path = img_path.parent.parent / 'annotations' / img_path.name.replace('.png', '.json')
            if ann_path.exists():
                shutil.copy2(ann_path, subset_annotations_dir / ann_path.name)
        
        return selected
    
    def convert_to_standard_format(self) -> Dict:
        """
        Convert FUNSD annotations to standardized format for evaluation.
        
        Returns:
            Dictionary with standardized annotations
        """
        if not self.subset_path.exists():
            raise FileNotFoundError(f"Subset not found at {self.subset_path}. Please create subset first.")
        
        standardized = {}
        annotations_dir = self.subset_path / 'annotations'
        
        for ann_file in annotations_dir.glob('*.json'):
            doc_id = ann_file.stem
            parsed = self.parse_annotations(ann_file)
            
            standardized[doc_id] = {
                'image_path': str(self.subset_path / 'images' / f"{doc_id}.png"),
                'ground_truth_text': ' '.join(parsed['text_lines']),
                'words': parsed['words'],
                'bounding_boxes': parsed['boxes'],
                'num_words': len(parsed['words']),
                'num_lines': len(parsed['text_lines'])
            }
        
        output_path = self.processed_path / 'standardized_annotations.json'
        self.processed_path.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(standardized, f, indent=2, ensure_ascii=False)
        
        print(f"Standardized annotations saved to {output_path}")
        return standardized
    
    def validate_data(self) -> Tuple[bool, List[str]]:
        """
        Validate data integrity and check for issues.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.subset_path.exists():
            issues.append("Subset directory does not exist")
            return False, issues
        
        images_dir = self.subset_path / 'images'
        annotations_dir = self.subset_path / 'annotations'
        
        if not images_dir.exists():
            issues.append("Images directory does not exist")
        
        if not annotations_dir.exists():
            issues.append("Annotations directory does not exist")
        
        if len(issues) > 0:
            return False, issues
        
        image_files = set(f.stem for f in images_dir.glob('*.png'))
        annotation_files = set(f.stem for f in annotations_dir.glob('*.json'))
        
        missing_annotations = image_files - annotation_files
        missing_images = annotation_files - image_files
        
        if missing_annotations:
            issues.append(f"Missing annotations for {len(missing_annotations)} images")
        
        if missing_images:
            issues.append(f"Missing images for {len(missing_images)} annotations")
        
        for img_file in images_dir.glob('*.png'):
            try:
                img = Image.open(img_file)
                img.verify()
            except Exception as e:
                issues.append(f"Corrupt image {img_file.name}: {e}")
        
        for ann_file in annotations_dir.glob('*.json'):
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'form' not in data:
                        issues.append(f"Missing 'form' key in {ann_file.name}")
            except Exception as e:
                issues.append(f"Invalid JSON in {ann_file.name}: {e}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            print("Data validation passed!")
        else:
            print(f"Data validation found {len(issues)} issue(s)")
        
        return is_valid, issues
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_documents': 0,
            'total_words': 0,
            'total_lines': 0,
            'avg_words_per_doc': 0,
            'avg_lines_per_doc': 0,
            'image_dimensions': []
        }
        
        if not self.subset_path.exists():
            return stats
        
        annotations_dir = self.subset_path / 'annotations'
        images_dir = self.subset_path / 'images'
        
        for ann_file in annotations_dir.glob('*.json'):
            stats['total_documents'] += 1
            parsed = self.parse_annotations(ann_file)
            stats['total_words'] += len(parsed['words'])
            stats['total_lines'] += len(parsed['text_lines'])
            
            img_path = images_dir / f"{ann_file.stem}.png"
            if img_path.exists():
                img = Image.open(img_path)
                stats['image_dimensions'].append(img.size)
        
        if stats['total_documents'] > 0:
            stats['avg_words_per_doc'] = stats['total_words'] / stats['total_documents']
            stats['avg_lines_per_doc'] = stats['total_lines'] / stats['total_documents']
        
        return stats
    
    def prepare_for_ocr(self, doc_id: str) -> Tuple[np.ndarray, Dict, str]:
        """
        Prepare a document for OCR processing.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Tuple of (image_array, ground_truth_data, image_path)
        """
        img_path = self.subset_path / 'images' / f"{doc_id}.png"
        ann_path = self.subset_path / 'annotations' / f"{doc_id}.json"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations not found: {ann_path}")
        
        image = cv2.imread(str(img_path))
        parsed = self.parse_annotations(ann_path)
        
        ground_truth = {
            'text': ' '.join(parsed['text_lines']),
            'words': parsed['words'],
            'boxes': parsed['boxes']
        }
        
        return image, ground_truth, str(img_path)


def main():
    """Main function to demonstrate dataset loading."""
    loader = FUNSDLoader(data_dir="data/funsd_subset")
    
    print("Step 1: Downloading FUNSD dataset...")
    if not loader.download_dataset():
        print("Failed to download dataset. Please check your internet connection.")
        return
    
    print("\nStep 2: Selecting subset of 100 documents...")
    selected = loader.select_subset(num_documents=100)
    
    print("\nStep 3: Converting to standardized format...")
    standardized = loader.convert_to_standard_format()
    
    print("\nStep 4: Validating data integrity...")
    is_valid, issues = loader.validate_data()
    
    if not is_valid:
        print("Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("\nStep 5: Computing dataset statistics...")
    stats = loader.get_dataset_stats()
    
    print("\nDataset Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Total lines: {stats['total_lines']}")
    print(f"  Avg words/doc: {stats['avg_words_per_doc']:.1f}")
    print(f"  Avg lines/doc: {stats['avg_lines_per_doc']:.1f}")
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()