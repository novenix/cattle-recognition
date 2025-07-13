"""
Extract individual cow crops from detection datasets for identification training.

This script processes YOLO format datasets and extracts bounding box regions
containing individual cows for use in contrastive learning.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from typing import List, Tuple, Dict


class CowCropExtractor:
    """Extract cow crops from YOLO detection datasets."""
    
    def __init__(self, 
                 input_dataset_path: str,
                 output_path: str,
                 min_crop_size: int = 64,
                 max_crop_size: int = 512,
                 padding_ratio: float = 0.1):
        """
        Initialize the cow crop extractor.
        
        Args:
            input_dataset_path: Path to YOLO dataset (with train/valid/test folders)
            output_path: Path to save extracted crops
            min_crop_size: Minimum crop size in pixels
            max_crop_size: Maximum crop size in pixels  
            padding_ratio: Padding around bounding box (0.1 = 10% padding)
        """
        self.input_path = Path(input_dataset_path)
        self.output_path = Path(output_path)
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.padding_ratio = padding_ratio
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "train").mkdir(exist_ok=True)
        (self.output_path / "valid").mkdir(exist_ok=True)
        (self.output_path / "test").mkdir(exist_ok=True)
        
        self.extraction_stats = {
            "total_images": 0,
            "total_crops": 0,
            "skipped_crops": 0,
            "train_crops": 0,
            "valid_crops": 0,
            "test_crops": 0
        }
    
    def parse_yolo_annotation(self, label_path: str, img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Parse YOLO format annotation file.
        
        Args:
            label_path: Path to .txt label file
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) in pixel coordinates
        """
        bboxes = []
        
        if not os.path.exists(label_path):
            return bboxes
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert from YOLO format to pixel coordinates
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    bboxes.append((x1, y1, x2, y2))
        
        return bboxes
    
    def apply_padding(self, bbox: Tuple[int, int, int, int], 
                     img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Apply padding to bounding box.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            img_width: Image width
            img_height: Image height
            
        Returns:
            Padded bounding box clamped to image boundaries
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate current bbox dimensions
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate padding
        pad_x = int(bbox_width * self.padding_ratio)
        pad_y = int(bbox_height * self.padding_ratio)
        
        # Apply padding and clamp to image boundaries
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_width, x2 + pad_x)
        y2 = min(img_height, y2 + pad_y)
        
        return (x1, y1, x2, y2)
    
    def is_valid_crop(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if crop meets size requirements.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            True if crop is valid, False otherwise
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Check minimum size
        if width < self.min_crop_size or height < self.min_crop_size:
            return False
            
        # Check maximum size
        if width > self.max_crop_size or height > self.max_crop_size:
            return False
            
        # Check aspect ratio (should be reasonable for cows)
        aspect_ratio = width / height
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
            
        return True
    
    def extract_crop(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract crop from image.
        
        Args:
            image: Input image as numpy array
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def process_split(self, split_name: str):
        """
        Process a dataset split (train, valid, test).
        
        Args:
            split_name: Name of the split to process
        """
        split_path = self.input_path / split_name
        images_path = split_path / "images"
        labels_path = split_path / "labels"
        
        if not images_path.exists() or not labels_path.exists():
            print(f"Skipping {split_name}: missing images or labels directory")
            return
        
        # Get all image files
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        print(f"\nProcessing {split_name} split: {len(image_files)} images")
        
        crop_count = 0
        for img_path in tqdm(image_files, desc=f"Extracting {split_name} crops"):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            img_height, img_width = image.shape[:2]
            self.extraction_stats["total_images"] += 1
            
            # Get corresponding label file
            label_file = labels_path / (img_path.stem + ".txt")
            
            # Parse annotations
            bboxes = self.parse_yolo_annotation(str(label_file), img_width, img_height)
            
            # Extract crops for each bounding box
            for i, bbox in enumerate(bboxes):
                # Apply padding
                padded_bbox = self.apply_padding(bbox, img_width, img_height)
                
                # Validate crop
                if not self.is_valid_crop(padded_bbox):
                    self.extraction_stats["skipped_crops"] += 1
                    continue
                
                # Extract crop
                crop = self.extract_crop(image, padded_bbox)
                
                # Save crop
                crop_filename = f"{img_path.stem}_cow_{i:02d}.jpg"
                crop_path = self.output_path / split_name / crop_filename
                
                cv2.imwrite(str(crop_path), crop)
                crop_count += 1
                self.extraction_stats["total_crops"] += 1
                self.extraction_stats[f"{split_name}_crops"] += 1
        
        print(f"Extracted {crop_count} crops from {split_name}")
    
    def extract_all_crops(self):
        """Extract crops from all dataset splits."""
        print("Starting cow crop extraction...")
        print(f"Input dataset: {self.input_path}")
        print(f"Output directory: {self.output_path}")
        print(f"Crop size range: {self.min_crop_size}-{self.max_crop_size} pixels")
        print(f"Padding ratio: {self.padding_ratio}")
        
        # Process each split
        for split in ["train", "valid", "test"]:
            self.process_split(split)
        
        # Save extraction statistics
        stats_path = self.output_path / "extraction_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.extraction_stats, f, indent=2)
        
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
        print(f"Total images processed: {self.extraction_stats['total_images']}")
        print(f"Total crops extracted: {self.extraction_stats['total_crops']}")
        print(f"Crops skipped (invalid): {self.extraction_stats['skipped_crops']}")
        print(f"Train crops: {self.extraction_stats['train_crops']}")
        print(f"Valid crops: {self.extraction_stats['valid_crops']}")
        print(f"Test crops: {self.extraction_stats['test_crops']}")
        print(f"Statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract cow crops from YOLO detection dataset")
    parser.add_argument("--input", required=True, help="Path to input YOLO dataset")
    parser.add_argument("--output", required=True, help="Path to output crops directory")
    parser.add_argument("--min-size", type=int, default=64, help="Minimum crop size")
    parser.add_argument("--max-size", type=int, default=512, help="Maximum crop size")
    parser.add_argument("--padding", type=float, default=0.1, help="Padding ratio around bbox")
    
    args = parser.parse_args()
    
    # Create extractor and run
    extractor = CowCropExtractor(
        input_dataset_path=args.input,
        output_path=args.output,
        min_crop_size=args.min_size,
        max_crop_size=args.max_size,
        padding_ratio=args.padding
    )
    
    extractor.extract_all_crops()


if __name__ == "__main__":
    main()