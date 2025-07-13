"""
Contrastive learning training script for cattle identification.

This script implements a SimCLR-style contrastive learning approach to train
a feature extractor that can generate distinctive embeddings for cattle re-identification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import random
from PIL import Image
import matplotlib.pyplot as plt


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with image augmentations."""
    
    def __init__(self, image_dir: str, transform=None, augment=None):
        """
        Initialize contrastive dataset.
        
        Args:
            image_dir: Directory containing cow crop images
            transform: Base transforms (resize, normalize)
            augment: Augmentation transforms for creating positive pairs
        """
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        self.transform = transform
        self.augment = augment
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Return two augmented versions of the same image for contrastive learning.
        
        Returns:
            tuple: (image1, image2) - two augmented versions of the same image
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply base transform
        if self.transform:
            image = self.transform(image)
        
        # Create two different augmented versions
        if self.augment:
            image1 = self.augment(image)
            image2 = self.augment(image)
        else:
            image1 = image
            image2 = image
            
        return image1, image2, idx


class ContrastiveAugmentation:
    """Strong augmentation pipeline for creating positive pairs."""
    
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform(x)


class CattleEncoder(nn.Module):
    """ResNet-based encoder for cattle feature extraction."""
    
    def __init__(self, backbone='resnet50', feature_dim=512, pretrained=True):
        """
        Initialize cattle encoder.
        
        Args:
            backbone: Backbone architecture ('resnet50')
            feature_dim: Dimension of output feature vector
            pretrained: Use ImageNet pretrained weights
        """
        super(CattleEncoder, self).__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            if pretrained:
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.backbone = resnet50(weights=None)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim)
        )
        
        # Add L2 normalization
        self.normalize = nn.functional.normalize
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: Input image batch [B, 3, H, W]
            
        Returns:
            Normalized feature vector [B, feature_dim]
        """
        # Extract features
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        
        # L2 normalize
        embeddings = self.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    
    def __init__(self, temperature=0.07, batch_size=None):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for softmax
            batch_size: Batch size (used to create labels)
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
    
    def forward(self, features1, features2):
        """
        Compute InfoNCE loss.
        
        Args:
            features1: Features from first augmentation [B, D]
            features2: Features from second augmentation [B, D]
            
        Returns:
            InfoNCE loss
        """
        batch_size = features1.shape[0]
        
        # Concatenate features
        features = torch.cat([features1, features2], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # [2B, 2B]
        
        # Create labels (positive pairs are at positions (i, i+B) and (i+B, i))
        labels = torch.arange(batch_size).to(features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2B]
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss


class ContrastiveTrainer:
    """Trainer for contrastive learning."""
    
    def __init__(self, 
                 model: CattleEncoder,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 temperature: float = 0.07):
        """
        Initialize trainer.
        
        Args:
            model: CattleEncoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            temperature: Temperature for InfoNCE loss
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = InfoNCELoss(temperature=temperature)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (img1, img2, _) in enumerate(pbar):
            img1, img2 = img1.to(self.device), img2.to(self.device)
            
            # Forward pass
            features1 = self.model(img1)
            features2 = self.model(img2)
            
            # Compute loss
            loss = self.criterion(features1, features2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for img1, img2, _ in tqdm(self.val_loader, desc="Validating"):
                img1, img2 = img1.to(self.device), img2.to(self.device)
                
                # Forward pass
                features1 = self.model(img1)
                features2 = self.model(img2)
                
                # Compute loss
                loss = self.criterion(features1, features2)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, num_epochs: int, save_dir: str):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Print metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'feature_dim': self.model.projection_head[-1].out_features
                }
                torch.save(checkpoint, save_path / 'best_model.pth')
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
                torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves(save_path / 'training_curves.png')
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved to: {save_path}")
    
    def plot_training_curves(self, save_path: str):
        """Plot and save training curves."""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 1, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_data_loaders(crops_dir: str, batch_size: int = 64, num_workers: int = 4):
    """Create training and validation data loaders."""
    
    # Base transforms
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])
    
    # Augmentation
    augmentation = ContrastiveAugmentation(size=224)
    
    # Datasets
    train_dataset = ContrastiveDataset(
        image_dir=os.path.join(crops_dir, 'train'),
        transform=base_transform,
        augment=augmentation
    )
    
    val_dataset = ContrastiveDataset(
        image_dir=os.path.join(crops_dir, 'valid'),
        transform=base_transform,
        augment=augmentation
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train cattle identification model with contrastive learning")
    parser.add_argument("--crops-dir", required=True, help="Directory containing extracted cow crops")
    parser.add_argument("--output-dir", required=True, help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--feature-dim", type=int, default=512, help="Feature dimension")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss")
    parser.add_argument("--device", default="cuda", help="Device to train on")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        crops_dir=args.crops_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = CattleEncoder(
        backbone='resnet50',
        feature_dim=args.feature_dim,
        pretrained=True
    )
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        temperature=args.temperature
    )
    
    # Train model
    trainer.train(
        num_epochs=args.epochs,
        save_dir=args.output_dir
    )


if __name__ == "__main__":
    main()