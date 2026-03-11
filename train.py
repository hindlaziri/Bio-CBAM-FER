"""
Training Script for Bio-CBAM Model

Author: Hind Laziri
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple

from models.bio_cbam import create_bio_cbam
from dataset_scripts.dataset_loader import create_dataloaders


class Trainer:
    """Bio-CBAM Model Trainer"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, device: str = 'cuda',
                 learning_rate: float = 0.001, num_epochs: int = 100):
        """
        Args:
            model: Bio-CBAM model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use ('cuda' or 'cpu')
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, checkpoint_dir: str = './checkpoints'):
        """Train model for specified number of epochs"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
        
        # Save final model
        final_path = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_path)
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train Bio-CBAM model')
    parser.add_argument('--dataset', type=str, default='fer2013',
                       choices=['fer2013', 'ckplus', 'jaffe'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--num-classes', type=int, default=7,
                       choices=[4, 7], help='Number of emotion classes')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--use-fmri-prior', action='store_true', default=True,
                       help='Use fMRI-guided spatial priors')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model
    print(f"Creating Bio-CBAM model...")
    model = create_bio_cbam(
        num_classes=args.num_classes,
        backbone=args.backbone,
        use_fmri_prior=args.use_fmri_prior,
        device=args.device
    )
    
    # Create data loaders
    print(f"Loading {args.dataset} dataset...")
    dataloaders = create_dataloaders(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    # Train
    trainer.train(checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':
    main()
