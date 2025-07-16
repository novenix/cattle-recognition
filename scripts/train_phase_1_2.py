#!/usr/bin/env python3
"""
Complete pipeline script for Phase 1.2: Identification Data Preparation and Training.

This script orchestrates the complete Phase 1.2 process:
1. Extract cow crops from detection datasets
2. Train contrastive learning model for identification
3. Validate the trained model

Usage:
    python scripts/train_phase_1_2.py --help
"""

"""
python scripts/train_phase_1_2.py
--dataset-path data/detection/cow-counting-v3
--epochs 50
--batch-size 32
--feature-dim 512
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    result = subprocess.run(command, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"❌ ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    else:
        print(f"✅ SUCCESS: {description} completed")


def main():
    parser = argparse.ArgumentParser(description="Complete Phase 1.2 pipeline for cattle identification")
    
    # Data paths
    parser.add_argument("--dataset-path", required=True, 
                       help="Path to cow-counting-v3 dataset")
    parser.add_argument("--crops-output", default="data/identification/cow_crops",
                       help="Output directory for extracted crops")
    parser.add_argument("--model-output", default="models/identification",
                       help="Output directory for trained models")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--feature-dim", type=int, default=512,
                       help="Feature vector dimension")
    
    # Crop extraction parameters
    parser.add_argument("--min-crop-size", type=int, default=64,
                       help="Minimum crop size in pixels")
    parser.add_argument("--max-crop-size", type=int, default=512,
                       help="Maximum crop size in pixels")
    parser.add_argument("--padding", type=float, default=0.1,
                       help="Padding ratio around bounding boxes")
    
    # System parameters
    parser.add_argument("--device", default="cuda",
                       help="Device for training (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Control flags
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip crop extraction (use existing crops)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (only extract crops)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.dataset_path).exists():
        print(f"❌ ERROR: Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)
    
    # Create output directories
    Path(args.crops_output).mkdir(parents=True, exist_ok=True)
    Path(args.model_output).mkdir(parents=True, exist_ok=True)
    
    print("🐄 CATTLE IDENTIFICATION TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Crops output: {args.crops_output}")
    print(f"Model output: {args.model_output}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Device: {args.device}")
    
    # Step 1: Extract cow crops
    if not args.skip_extraction:
        extract_cmd = (
            f"python src/data_preparation/extract_cow_crops.py "
            f"--input {args.dataset_path} "
            f"--output {args.crops_output} "
            f"--min-size {args.min_crop_size} "
            f"--max-size {args.max_crop_size} "
            f"--padding {args.padding}"
        )
        
        run_command(extract_cmd, "Extracting cow crops from detection dataset")
    else:
        print("\n⏭️  SKIPPING crop extraction (--skip-extraction flag)")
    
    # Step 2: Train identification model
    if not args.skip_training:
        train_cmd = (
            f"python src/training/train_identification.py "
            f"--crops-dir {args.crops_output} "
            f"--output-dir {args.model_output} "
            f"--epochs {args.epochs} "
            f"--batch-size {args.batch_size} "
            f"--learning-rate {args.learning_rate} "
            f"--feature-dim {args.feature_dim} "
            f"--device {args.device} "
            f"--num-workers {args.num_workers}"
        )
        
        run_command(train_cmd, "Training contrastive identification model")
    else:
        print("\n⏭️  SKIPPING model training (--skip-training flag)")
    
    # Step 3: Summary
    print("\n" + "="*60)
    print("🎉 PHASE 1.2 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    if not args.skip_extraction:
        crops_stats_file = Path(args.crops_output) / "extraction_stats.json"
        if crops_stats_file.exists():
            import json
            with open(crops_stats_file, 'r') as f:
                stats = json.load(f)
            print(f"📊 Extraction Stats:")
            print(f"   - Total crops: {stats['total_crops']}")
            print(f"   - Train crops: {stats['train_crops']}")
            print(f"   - Valid crops: {stats['valid_crops']}")
            print(f"   - Test crops: {stats['test_crops']}")
    
    if not args.skip_training:
        model_file = Path(args.model_output) / "best_model.pth"
        if model_file.exists():
            print(f"🤖 Model saved: {model_file}")
            print(f"📈 Training curves: {Path(args.model_output) / 'training_curves.png'}")
    
    print("\n📋 Next Steps:")
    print("1. Evaluate model performance with validation metrics")
    print("2. Test the identification pipeline with sample images")
    print("3. Integrate with detection model for Phase 4 (Tracking Logic)")
    print("4. Proceed to Phase 2: System Capability Training")
    
    print(f"\n💡 Quick test command:")
    print(f"python src/inference/identification_model.py")


if __name__ == "__main__":
    main()