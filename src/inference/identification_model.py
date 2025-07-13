"""
Inference pipeline for cattle identification using trained contrastive learning model.

This module provides functionality to load a trained model and extract feature vectors
from cow images for use in the tracking system.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
import json


class CattleIdentificationModel:
    """Inference model for cattle identification."""
    
    def __init__(self, model_path: str, device: str = 'auto', feature_dim: int = 512):
        """
        Initialize the identification model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            feature_dim: Feature dimension of the model
        """
        self.device = self._get_device(device)
        self.feature_dim = feature_dim
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup preprocessing
        self.preprocess = self._create_preprocessing_pipeline()
        
        print(f"Cattle identification model loaded on {self.device}")
        print(f"Feature dimension: {self.feature_dim}")
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for inference."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, model_path: str):
        """Load the trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract feature dimension from checkpoint if available
        if 'feature_dim' in checkpoint:
            self.feature_dim = checkpoint['feature_dim']
        
        # Reconstruct model architecture
        from ..training.train_identification import CattleEncoder
        model = CattleEncoder(
            backbone='resnet50',
            feature_dim=self.feature_dim,
            pretrained=False  # We're loading trained weights
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def _create_preprocessing_pipeline(self):
        """Create preprocessing pipeline for input images."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess input image for inference.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if necessary
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply preprocessing
        tensor = self.preprocess(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def extract_features(self, images: Union[torch.Tensor, List]) -> np.ndarray:
        """
        Extract feature vectors from images.
        
        Args:
            images: Input images (tensor or list of images)
            
        Returns:
            Feature vectors as numpy array [N, feature_dim]
        """
        with torch.no_grad():
            if isinstance(images, list):
                # Process multiple images
                batch_tensors = []
                for img in images:
                    tensor = self.preprocess_image(img)
                    batch_tensors.append(tensor)
                
                batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            else:
                # Single tensor input
                batch_tensor = images.to(self.device)
            
            # Extract features
            features = self.model(batch_tensor)
            
            # Convert to numpy
            return features.cpu().numpy()
    
    def get_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            features1: First feature vector [feature_dim]
            features2: Second feature vector [feature_dim]
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        features1 = features1 / np.linalg.norm(features1)
        features2 = features2 / np.linalg.norm(features2)
        
        # Compute cosine similarity
        similarity = np.dot(features1, features2)
        return float(similarity)
    
    def find_most_similar(self, query_features: np.ndarray, 
                         database_features: List[np.ndarray]) -> Tuple[int, float]:
        """
        Find the most similar feature vector in a database.
        
        Args:
            query_features: Query feature vector [feature_dim]
            database_features: List of database feature vectors
            
        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        if not database_features:
            return -1, 0.0
        
        best_similarity = -1.0
        best_index = -1
        
        for i, db_features in enumerate(database_features):
            similarity = self.get_similarity(query_features, db_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i
        
        return best_index, best_similarity


class CattleDatabase:
    """Database for storing and managing cattle feature vectors."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize cattle database.
        
        Args:
            similarity_threshold: Threshold for considering two features as the same cow
        """
        self.similarity_threshold = similarity_threshold
        self.features_db = []  # List of feature vectors
        self.cow_ids = []      # List of corresponding cow IDs
        self.next_cow_id = 1   # Next available cow ID
        
    def add_cow(self, features: np.ndarray, cow_id: int = None) -> int:
        """
        Add a new cow to the database.
        
        Args:
            features: Feature vector for the cow
            cow_id: Specific cow ID (if None, auto-assign)
            
        Returns:
            Assigned cow ID
        """
        if cow_id is None:
            cow_id = self.next_cow_id
            self.next_cow_id += 1
        else:
            self.next_cow_id = max(self.next_cow_id, cow_id + 1)
        
        self.features_db.append(features.copy())
        self.cow_ids.append(cow_id)
        
        return cow_id
    
    def identify_cow(self, features: np.ndarray, model: CattleIdentificationModel) -> Tuple[int, float]:
        """
        Identify a cow or register as new if no match found.
        
        Args:
            features: Feature vector of the cow to identify
            model: Identification model for similarity computation
            
        Returns:
            Tuple of (cow_id, similarity_score)
        """
        if not self.features_db:
            # First cow in database
            cow_id = self.add_cow(features)
            return cow_id, 1.0
        
        # Find most similar cow in database
        best_index, best_similarity = model.find_most_similar(features, self.features_db)
        
        if best_similarity >= self.similarity_threshold:
            # Found a match
            return self.cow_ids[best_index], best_similarity
        else:
            # No match found, register as new cow
            cow_id = self.add_cow(features)
            return cow_id, 0.0  # 0.0 indicates this is a new cow
    
    def get_database_stats(self) -> dict:
        """Get statistics about the database."""
        return {
            'total_cows': len(self.features_db),
            'similarity_threshold': self.similarity_threshold,
            'next_cow_id': self.next_cow_id
        }
    
    def save_database(self, filepath: str):
        """Save database to file."""
        data = {
            'similarity_threshold': self.similarity_threshold,
            'features_db': [f.tolist() for f in self.features_db],
            'cow_ids': self.cow_ids,
            'next_cow_id': self.next_cow_id
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_database(self, filepath: str):
        """Load database from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.similarity_threshold = data['similarity_threshold']
        self.features_db = [np.array(f) for f in data['features_db']]
        self.cow_ids = data['cow_ids']
        self.next_cow_id = data['next_cow_id']


def demo_identification_pipeline():
    """Demonstrate the identification pipeline."""
    print("Cattle Identification Pipeline Demo")
    print("=" * 50)
    
    # This would be replaced with actual model path
    model_path = "models/identification/best_model.pth"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_identification.py")
        return
    
    # Initialize identification model
    print("Loading identification model...")
    id_model = CattleIdentificationModel(model_path)
    
    # Initialize database
    print("Initializing cattle database...")
    database = CattleDatabase(similarity_threshold=0.8)
    
    # Demo with sample images (this would be replaced with actual cow crops)
    print("\nDemo completed!")
    print("Use this pipeline in your tracking system to identify cows.")


if __name__ == "__main__":
    demo_identification_pipeline()