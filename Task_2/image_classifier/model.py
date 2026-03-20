"""
Image Classification model for animal recognition.
Uses transfer learning with pre-trained models (ResNet, Vision Transformer).

This module provides:
- Model training with transfer learning and fine-tuning
- Inference for animal classification
- Model saving and loading
- Data augmentation and preprocessing
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt

# Optional progress bar for training
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class AnimalImageDataset(Dataset):
    """Dataset for animal image classification."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of class labels (integers)
            transform: Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AnimalImageClassifier:
    """Image classifier for animal recognition using transfer learning."""
    
    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "resnet50",
        pretrained: bool = True,
        device: str = None
    ):
        """
        Initialize image classifier.
        
        Args:
            num_classes: Number of animal classes
            model_name: Model architecture ('resnet50', 'resnet101', 'vit_b_16', etc.)
            pretrained: Whether to use pretrained weights
            device: Device to use ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if device is None:
            self.device = "cuda" if cuda_available else "cpu"
        else:
            self.device = device

        self.class_names = None
        
        # Use 'weights' parameter instead of deprecated 'pretrained'
        weights = None
        if pretrained:
            if model_name == "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V1
            elif model_name == "resnet101":
                weights = ResNet101_Weights.IMAGENET1K_V1
        
        # Load model with weights parameter
        if model_name.startswith("resnet"):
            if model_name == "resnet50":
                self.model = models.resnet50(weights=weights)
            elif model_name == "resnet101":
                self.model = models.resnet101(weights=weights)
            else:
                self.model = models.resnet50(weights=weights)
            
            # Modify final layer
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            # Default to ResNet50
            self.model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Define image transformations
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def set_class_names(self, class_names: List[str]):
        """Set class names for predictions."""
        self.class_names = class_names
    
    def train(
        self,
        train_image_paths: List[str],
        train_labels: List[int],
        val_image_paths: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        output_dir: str = "./models/image_classifier",
        freeze_backbone: bool = True,
        verbose: bool = True
    ):
        """
        Train the image classification model.
        
        Args:
            train_image_paths: List of training image paths
            train_labels: List of training labels
            val_image_paths: List of validation image paths
            val_labels: List of validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            output_dir: Directory to save trained model
            freeze_backbone: Whether to freeze backbone layers
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create datasets
        train_dataset = AnimalImageDataset(
            train_image_paths,
            train_labels,
            self.train_transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        
        # Optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Training loop
        self.model.train()
        train_history = {"loss": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0

            if verbose and tqdm is not None:
                iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            else:
                iterator = train_loader
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (images, labels) in enumerate(iterator, 1):
                try:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    if verbose and tqdm is None and batch_idx % 20 == 0:
                        # Print intermediate batch progress if tqdm is not available
                        print(f"  Batch {batch_idx}/{len(train_loader)} - loss: {loss.item():.4f}")
                
                except RuntimeError as e:
                    if "CUDA" in str(e) or "cuda" in str(e):
                        print(f"\n⚠ CUDA error encountered during training: {str(e)[:100]}...")
                        print("  Switching to CPU and reloading model...")
                        self.device = "cpu"
                        self.model = self.model.to(self.device)
                        print("  Resuming training on CPU...")
                        # Retry this batch on CPU
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    else:
                        raise

            epoch_loss /= len(train_loader)
            train_history["loss"].append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

            if val_image_paths is not None:
                val_loss, val_acc = self.evaluate(
                    val_image_paths,
                    val_labels,
                    batch_size=batch_size
                )
                train_history["val_loss"].append(val_loss)
                train_history["val_acc"].append(val_acc)
                if verbose:
                    print(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            scheduler.step()
        
        # Save model
        self.save_model(output_dir)
        
        # Save training history
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(train_history, f)
        
        print(f"Model saved to {output_dir}")
    
    def evaluate(
        self,
        image_paths: List[str],
        labels: List[int],
        batch_size: int = 32
    ) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            image_paths: List of test image paths
            labels: List of test labels
            batch_size: Batch size for evaluation
        
        Returns:
            Tuple of (loss, accuracy)
        """
        dataset = AnimalImageDataset(image_paths, labels, self.val_transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels_batch in loader:
                images = images.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(
        self,
        image_path: str,
        top_k: int = 5,
        return_probabilities: bool = True
    ) -> Dict[str, any]:
        """
        Classify an image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            return_probabilities: Whether to return probability scores
        
        Returns:
            Dictionary with predictions
        """
        image = Image.open(image_path).convert("RGB")
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            class_name = self.class_names[idx] if self.class_names else f"Class {idx}"
            predictions.append({
                "class": class_name,
                "class_id": int(idx),
                "confidence": float(probabilities[idx])
            })
        
        return {
            "image_path": image_path,
            "top_prediction": predictions[0],
            "top_k_predictions": predictions if return_probabilities else None
        }
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Classify multiple images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for prediction
        
        Returns:
            List of predictions
        """
        dataset = AnimalImageDataset(
            image_paths,
            [0] * len(image_paths),  # Dummy labels
            self.val_transform
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                
                for idx, probs in enumerate(probabilities):
                    top_idx = np.argmax(probs)
                    class_name = self.class_names[top_idx] if self.class_names else f"Class {top_idx}"
                    all_predictions.append({
                        "class": class_name,
                        "confidence": float(probs[top_idx])
                    })
        
        return all_predictions
    
    def save_model(self, output_dir: str):
        """Save model checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pth"))
        
        # Save metadata
        metadata = {
            "num_classes": self.num_classes,
            "model_name": self.model_name,
            "class_names": self.class_names
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def load_model(self, model_dir: str):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "model.pth"),
            map_location=self.device
        ))
        
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.class_names = metadata.get("class_names")
        
        print(f"Model loaded from {model_dir}")
