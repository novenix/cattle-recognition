#!/usr/bin/env python3
"""
Detection Model Training Script for Phase 2.1: Detection Capability Training.

This script trains a YOLO model for cattle detection using the cattle-detection datasets.
It implements the detection capability required for the cattle recognition system.

Usage:
    python src/training/train_detection.py --dataset data/detection/cattle-detection-v3 --epochs 100

Architecture:
    - YOLO v11 for real-time cattle detection
    - Single class: 'cow'  
    - Output: Bounding boxes with confidence scores
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


class CattleDetectionTrainer:
    """
    Trainer class for cattle detection model using YOLO.
    
    This implements Phase 2.1 of the cattle recognition system:
    Training a model specialized in locating and drawing bounding boxes 
    around all cattle in an image or video frame.
    """
    
    def __init__(self, dataset_path, model_size='n', device='auto'):
        """
        Initialize the detection trainer.
        
        Args:
            dataset_path (str): Path to YOLO format dataset
            model_size (str): YOLO model size ('n', 's', 'm', 'l', 'x')
            device (str): Device for training ('auto', 'cpu', 'cuda')
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.device = device
        
        # Validate dataset
        self.data_yaml = self.dataset_path / 'data.yaml'
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset configuration not found: {self.data_yaml}")
            
        # Load dataset configuration
        with open(self.data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
            
        print(f"📊 Dataset: {self.dataset_path}")
        print(f"🎯 Classes: {self.data_config['nc']} - {self.data_config['names']}")
        print(f"🤖 Model: YOLOv11{self.model_size}")
        print(f"⚡ Device: {self.device}")
        
    def setup_model(self):
        """Initialize YOLO model."""
        model_name = f"yolo11{self.model_size}.pt"
        print(f"🔄 Loading YOLO model: {model_name}")
        
        # Initialize model - will download pretrained weights if not present
        self.model = YOLO(model_name)
        
        # Modify model for single class if needed
        if self.data_config['nc'] != 80:  # COCO has 80 classes
            print(f"📝 Adapting model for {self.data_config['nc']} classes")
            
    def train(self, epochs=100, imgsz=640, batch_size=16, workers=8, 
              patience=50, save_period=10, project='models/detection', 
              name='cattle_detector'):
        """
        Train the detection model.
        
        Args:
            epochs (int): Number of training epochs
            imgsz (int): Image size for training
            batch_size (int): Batch size
            workers (int): Number of dataloader workers
            patience (int): Early stopping patience
            save_period (int): Save checkpoint every N epochs
            project (str): Project directory
            name (str): Experiment name
        """
        print(f"\n{'='*60}")
        print(f"🚀 STARTING CATTLE DETECTION TRAINING")
        print(f"{'='*60}")
        
        # Create output directory
        output_dir = Path(project) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        train_args = {
            'data': str(self.data_yaml),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': self.device,
            'workers': workers,
            'patience': patience,
            'save_period': save_period,
            'project': project,
            'name': name,
            'verbose': True,
            'plots': True,
            'cache': True,  # Cache images for faster training
            'save': True,
            'save_txt': True,  # Save labels in txt format
            'save_conf': True,  # Save confidences in txt format
        }
        
        print(f"📋 Training Configuration:")
        for key, value in train_args.items():
            print(f"   {key}: {value}")
        print()
        
        # Start training
        start_time = datetime.now()
        print(f"⏰ Training started at: {start_time}")
        
        try:
            # Train the model
            results = self.model.train(**train_args)
            
            # Training completed
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"\n✅ Training completed successfully!")
            print(f"⏰ Duration: {duration}")
            print(f"📁 Results saved in: {output_dir}")
            
            # Save training summary
            self._save_training_summary(output_dir, train_args, duration, results)
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise
            
    def _save_training_summary(self, output_dir, train_args, duration, results):
        """Save training summary and metadata."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset': str(self.dataset_path),
            'model_size': self.model_size,
            'training_duration': str(duration),
            'training_args': train_args,
            'dataset_config': self.data_config,
            'best_model_path': str(output_dir / 'weights' / 'best.pt'),
            'last_model_path': str(output_dir / 'weights' / 'last.pt'),
        }
        
        # Add metrics if available
        if hasattr(results, 'results_dict'):
            summary['final_metrics'] = results.results_dict
            
        # Save summary
        summary_file = output_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📊 Training summary saved: {summary_file}")
        
    def validate(self, model_path=None, dataset_split='val'):
        """
        Validate the trained model.
        
        Args:
            model_path (str): Path to model weights (None for best from training)
            dataset_split (str): Dataset split to validate on ('val', 'test')
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
            
        print(f"\n🔍 Validating model on {dataset_split} set...")
        
        # Prepare validation data path
        val_data = dict(self.data_config)
        if dataset_split == 'test':
            val_data['val'] = val_data['test']
            
        # Save temporary config
        temp_config = Path('temp_val_config.yaml')
        with open(temp_config, 'w') as f:
            yaml.dump(val_data, f)
            
        try:
            # Run validation
            results = model.val(data=str(temp_config), verbose=True)
            
            print(f"✅ Validation completed:")
            print(f"   mAP50: {results.box.map50:.4f}")
            print(f"   mAP50-95: {results.box.map:.4f}")
            print(f"   Precision: {results.box.mp:.4f}")
            print(f"   Recall: {results.box.mr:.4f}")
            
            return results
            
        finally:
            # Cleanup
            if temp_config.exists():
                temp_config.unlink()
                
    def export_model(self, model_path, format='onnx', optimize=True):
        """
        Export model for deployment (Phase 3: Real-World Optimization).
        
        Args:
            model_path (str): Path to trained model
            format (str): Export format ('onnx', 'torchscript', 'tensorrt', etc.)
            optimize (bool): Apply optimizations for deployment
        """
        print(f"\n🚀 Exporting model for deployment...")
        print(f"   Format: {format}")
        print(f"   Optimize: {optimize}")
        
        model = YOLO(model_path)
        
        # Export model
        exported_path = model.export(
            format=format,
            optimize=optimize,
            verbose=True
        )
        
        print(f"✅ Model exported: {exported_path}")
        return exported_path


def main():
    parser = argparse.ArgumentParser(description="Train cattle detection model (Phase 2.1)")
    
    # Dataset arguments
    parser.add_argument("--dataset", required=True,
                       help="Path to YOLO format dataset directory")
    parser.add_argument("--model-size", default="n", choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--device", default="auto",
                       help="Training device (auto, cpu, cuda)")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of dataloader workers")
    
    # Output arguments
    parser.add_argument("--project", default="models/detection",
                       help="Project directory for saving results")
    parser.add_argument("--name", default="cattle_detector",
                       help="Experiment name")
    
    # Control flags
    parser.add_argument("--validate", action="store_true",
                       help="Run validation after training")
    parser.add_argument("--export", action="store_true",
                       help="Export model after training")
    parser.add_argument("--export-format", default="onnx",
                       help="Export format for deployment")
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not Path(args.dataset).exists():
        print(f"❌ Dataset path does not exist: {args.dataset}")
        sys.exit(1)
        
    print("🐄 CATTLE DETECTION TRAINING - PHASE 2.1")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = CattleDetectionTrainer(
            dataset_path=args.dataset,
            model_size=args.model_size,
            device=args.device
        )
        
        # Setup model
        trainer.setup_model()
        
        # Train model
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            workers=args.workers,
            project=args.project,
            name=args.name
        )
        
        # Get best model path
        best_model = Path(args.project) / args.name / 'weights' / 'best.pt'
        
        # Validation
        if args.validate and best_model.exists():
            print(f"\n🔍 Running validation...")
            trainer.validate(str(best_model))
            
        # Export model
        if args.export and best_model.exists():
            print(f"\n📦 Exporting model...")
            trainer.export_model(str(best_model), format=args.export_format)
            
        print(f"\n🎉 PHASE 2.1 DETECTION TRAINING COMPLETED!")
        print(f"📁 Best model: {best_model}")
        print(f"📊 Results: {Path(args.project) / args.name}")
        
        print(f"\n📋 Next Steps:")
        print(f"1. Integrate with Phase 1.2 identification model")
        print(f"2. Implement Phase 4: Tracking Logic Development")
        print(f"3. Test complete detection + identification pipeline")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()