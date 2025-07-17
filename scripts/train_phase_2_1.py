#!/usr/bin/env python3
"""
Phase 2.1 Training Script: Detection Capability Training

This script orchestrates the complete Phase 2.1 process:
1. Train cattle detection model using YOLO
2. Validate the trained model
3. Export for deployment (Phase 3 preparation)

Usage:
    python scripts/train_phase_2_1.py --dataset data/detection/cattle-detection-v3
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
    parser = argparse.ArgumentParser(description="Complete Phase 2.1 pipeline for cattle detection")
    
    # Data arguments
    parser.add_argument("--dataset", required=True,
                       help="Path to YOLO format detection dataset")
    parser.add_argument("--model-output", default="models/detection",
                       help="Output directory for trained models")
    
    # Model arguments
    parser.add_argument("--model-size", default="s", choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLO model size (s=small recommended for balance)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--device", default="auto",
                       help="Training device (auto, cpu, cuda)")
    
    # Control flags
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (only validate existing model)")
    parser.add_argument("--export-format", default="onnx",
                       help="Export format for deployment")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.dataset).exists():
        print(f"❌ ERROR: Dataset path does not exist: {args.dataset}")
        sys.exit(1)
        
    # Create output directory
    Path(args.model_output).mkdir(parents=True, exist_ok=True)
    
    print("🐄 CATTLE DETECTION TRAINING PIPELINE - PHASE 2.1")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Model output: {args.model_output}")
    print(f"Model size: YOLO v11{args.model_size}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    
    # Step 1: Train detection model
    if not args.skip_training:
        train_cmd = (
            f"python src/training/train_detection.py "
            f"--dataset {args.dataset} "
            f"--model-size {args.model_size} "
            f"--epochs {args.epochs} "
            f"--batch-size {args.batch_size} "
            f"--imgsz {args.imgsz} "
            f"--device {args.device} "
            f"--project {args.model_output} "
            f"--name cattle_detector_v11{args.model_size} "
            f"--validate "
            f"--export "
            f"--export-format {args.export_format}"
        )
        
        run_command(train_cmd, "Training cattle detection model")
        
        # Model paths
        model_dir = Path(args.model_output) / f"cattle_detector_v11{args.model_size}"
        best_model = model_dir / "weights" / "best.pt"
        exported_model = model_dir / "weights" / f"best.{args.export_format}"
        
    else:
        print("\n⏭️  SKIPPING training (--skip-training flag)")
        
        # Look for existing model
        model_dir = Path(args.model_output) / f"cattle_detector_v11{args.model_size}"
        best_model = model_dir / "weights" / "best.pt"
        
        if not best_model.exists():
            print(f"❌ ERROR: No existing model found at {best_model}")
            sys.exit(1)
    
    # Step 2: Summary
    print("\n" + "="*70)
    print("🎉 PHASE 2.1 DETECTION TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    if best_model.exists():
        print(f"🤖 Detection Model: {best_model}")
        print(f"📈 Training logs: {model_dir}")
        
        # Check if exported model exists
        exported_candidates = [
            model_dir / "weights" / f"best.{args.export_format}",
            model_dir / f"best.{args.export_format}",
        ]
        
        exported_model = None
        for candidate in exported_candidates:
            if candidate.exists():
                exported_model = candidate
                break
                
        if exported_model:
            print(f"📦 Exported Model: {exported_model}")
    
    # Integration guidance
    print("\n📋 Next Steps - Integration with Phase 1.2:")
    print("1. You now have both required AI engines:")
    print(f"   🔍 Detection Model: {best_model}")
    print(f"   🏷️  Identification Model: models/identification/best_model.pth")
    print("\n2. Ready for Phase 4: Tracking Logic Development")
    print("   - Implement main tracking loop")
    print("   - Integrate detection + identification")
    print("   - Add similarity threshold calibration")
    
    print("\n💡 Quick test commands:")
    print(f"# Test detection:")
    print(f"python -c \"from ultralytics import YOLO; YOLO('{best_model}').predict('path/to/image.jpg', show=True)\"")
    print(f"\n# Validate model:")
    print(f"python src/training/train_detection.py --dataset {args.dataset} --validate")
    
    print(f"\n🎯 Phase 2 Status:")
    print(f"   ✅ Phase 2.1: Detection Capability Training (COMPLETED)")
    print(f"   ✅ Phase 2.2: Differentiation Capability Training (COMPLETED)")
    print(f"   ➡️  Ready for Phase 4: Tracking Logic Development")


if __name__ == "__main__":
    main()