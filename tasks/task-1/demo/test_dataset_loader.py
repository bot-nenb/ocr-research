#!/usr/bin/env python3
"""Test script for FUNSD dataset loader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dataset_loader import FUNSDLoader


def test_dataset_loader():
    """Test the FUNSD dataset loader functionality."""
    
    print("=" * 60)
    print("FUNSD Dataset Loader Test")
    print("=" * 60)
    
    loader = FUNSDLoader(data_dir="data/funsd_subset")
    
    print("\n1. Testing dataset download...")
    print("-" * 40)
    success = loader.download_dataset()
    if success:
        print("✓ Dataset download successful")
    else:
        print("✗ Dataset download failed")
        return False
    
    print("\n2. Testing subset selection...")
    print("-" * 40)
    try:
        selected = loader.select_subset(num_documents=100)
        print(f"✓ Selected {len(selected)} documents")
    except Exception as e:
        print(f"✗ Subset selection failed: {e}")
        return False
    
    print("\n3. Testing annotation parsing...")
    print("-" * 40)
    try:
        standardized = loader.convert_to_standard_format()
        print(f"✓ Converted {len(standardized)} documents to standard format")
        
        if standardized:
            first_doc = next(iter(standardized.values()))
            print(f"  Sample document has {first_doc['num_words']} words and {first_doc['num_lines']} lines")
    except Exception as e:
        print(f"✗ Annotation parsing failed: {e}")
        return False
    
    print("\n4. Testing data validation...")
    print("-" * 40)
    try:
        is_valid, issues = loader.validate_data()
        if is_valid:
            print("✓ Data validation passed")
        else:
            print(f"⚠ Data validation found {len(issues)} issues:")
            for issue in issues[:5]:
                print(f"  - {issue}")
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        return False
    
    print("\n5. Testing dataset statistics...")
    print("-" * 40)
    try:
        stats = loader.get_dataset_stats()
        print("✓ Dataset statistics computed:")
        print(f"  - Total documents: {stats['total_documents']}")
        print(f"  - Total words: {stats['total_words']}")
        print(f"  - Average words per document: {stats['avg_words_per_doc']:.1f}")
        print(f"  - Average lines per document: {stats['avg_lines_per_doc']:.1f}")
    except Exception as e:
        print(f"✗ Statistics computation failed: {e}")
        return False
    
    print("\n6. Testing document preparation for OCR...")
    print("-" * 40)
    try:
        if standardized:
            doc_id = next(iter(standardized.keys()))
            image, ground_truth = loader.prepare_for_ocr(doc_id)
            print(f"✓ Prepared document '{doc_id}' for OCR")
            print(f"  - Image shape: {image.shape}")
            print(f"  - Ground truth text length: {len(ground_truth['text'])} characters")
            print(f"  - Number of words: {len(ground_truth['words'])}")
    except Exception as e:
        print(f"✗ Document preparation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_dataset_loader()
    sys.exit(0 if success else 1)