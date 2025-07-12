#!/usr/bin/env python3
"""
Data preparation utilities for cattle recognition project.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import json

def create_combined_dataset(source_datasets: List[Path], output_path: Path, 
                          train_ratio: float = 0.7, val_ratio: float = 0.2, 
                          test_ratio: float = 0.1) -> None:
    """
    Combine multiple YOLO datasets into a single dataset with proper splits.
    
    Args:
        source_datasets: List of paths to source datasets
        output_path: Path where combined dataset will be saved
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation  
        test_ratio: Ratio of data for testing
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    all_images = []
    all_labels = []
    
    # Collect all images and labels from source datasets
    for dataset_path in source_datasets:
        for split in ['train', 'valid', 'test']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                images = list(images_dir.glob('*.[jp][pn]g')) + list(images_dir.glob('*.jpeg'))
                for img_path in images:
                    label_path = labels_dir / (img_path.stem + '.txt')
                    if label_path.exists():
                        all_images.append(img_path)
                        all_labels.append(label_path)
    
    print(f"Found {len(all_images)} image-label pairs")
    
    # Shuffle and split data
    import random
    random.seed(42)
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    
    n_total = len(combined)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = combined[:n_train]
    val_data = combined[n_train:n_train + n_val]
    test_data = combined[n_train + n_val:]
    
    # Copy files to new structure
    for split_name, split_data in [('train', train_data), ('valid', val_data), ('test', test_data)]:
        for i, (img_path, label_path) in enumerate(split_data):
            # Copy image
            new_img_path = output_path / split_name / 'images' / f"{split_name}_{i:06d}{img_path.suffix}"
            shutil.copy2(img_path, new_img_path)
            
            # Copy label
            new_label_path = output_path / split_name / 'labels' / f"{split_name}_{i:06d}.txt"
            shutil.copy2(label_path, new_label_path)
    
    print(f"Created combined dataset:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Valid: {len(val_data)} samples") 
    print(f"  Test: {len(test_data)} samples")

def analyze_class_distribution(dataset_path: Path) -> Dict:
    """Analyze class distribution across dataset splits."""
    class_stats = {}
    
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_path / split / 'labels'
        if not labels_dir.exists():
            continue
            
        class_counts = {}
        total_objects = 0
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_objects += 1
        
        class_stats[split] = {
            'class_counts': class_counts,
            'total_objects': total_objects,
            'num_images': len(list(labels_dir.glob('*.txt')))
        }
    
    return class_stats

def create_data_yaml(dataset_path: Path, class_names: List[str]) -> None:
    """Create data.yaml file for YOLO training."""
    yaml_content = {
        'train': str(dataset_path / 'train' / 'images'),
        'val': str(dataset_path / 'valid' / 'images'),
        'test': str(dataset_path / 'test' / 'images'),
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_file = dataset_path / 'data.yaml'
    
    # Write YAML content manually (avoid dependency on PyYAML)
    with open(yaml_file, 'w') as f:
        f.write(f"train: {yaml_content['train']}\n")
        f.write(f"val: {yaml_content['val']}\n")
        f.write(f"test: {yaml_content['test']}\n")
        f.write(f"nc: {yaml_content['nc']}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
    
    print(f"Created data.yaml at {yaml_file}")

if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent.parent.parent / "data" / "detection"
    
    if data_dir.exists():
        datasets = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"Found datasets: {[d.name for d in datasets]}")
        
        # Analyze each dataset
        for dataset in datasets:
            print(f"\nAnalyzing {dataset.name}:")
            stats = analyze_class_distribution(dataset)
            for split, info in stats.items():
                if info['num_images'] > 0:
                    print(f"  {split}: {info['num_images']} images, {info['total_objects']} objects")
                    print(f"    Classes: {info['class_counts']}")
    else:
        print("No datasets found. Please run download_datasets.py first.")