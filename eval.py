"""
Evaluation Script for Bio-CBAM Model

Evaluates model performance on test sets with detailed metrics.

Author: Hind Laziri
Date: 2026
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from models.bio_cbam import create_bio_cbam
from dataset_scripts.dataset_loader import create_dataloaders


class Evaluator:
    """Bio-CBAM Model Evaluator"""
    
    def __init__(self, model: nn.Module, test_loader: DataLoader,
                 device: str = 'cuda', num_classes: int = 7):
        """
        Args:
            model: Bio-CBAM model
            test_loader: Test data loader
            device: Device to use
            num_classes: Number of emotion classes
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate(self) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(self.test_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'classification_report': classification_report(all_labels, all_preds, output_dict=True)
        }
        
        return metrics, all_preds, all_labels
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = './confusion_matrix.png'):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_metrics(self, metrics: Dict, save_path: str = './metrics.png'):
        """Plot evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].bar(['Accuracy'], [metrics['accuracy']])
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].text(0, metrics['accuracy'], f"{metrics['accuracy']:.4f}", ha='center')
        
        # Precision, Recall, F1
        scores = [metrics['precision'], metrics['recall'], metrics['f1']]
        axes[0, 1].bar(['Precision', 'Recall', 'F1'], scores)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_title('Detailed Metrics')
        for i, score in enumerate(scores):
            axes[0, 1].text(i, score, f"{score:.4f}", ha='center')
        
        # Loss
        axes[1, 0].bar(['Loss'], [metrics['loss']])
        axes[1, 0].set_title('Test Loss')
        axes[1, 0].text(0, metrics['loss'], f"{metrics['loss']:.4f}", ha='center')
        
        # Per-class metrics
        class_metrics = metrics['classification_report']
        classes = [str(i) for i in range(self.num_classes)]
        precisions = [class_metrics[c]['precision'] for c in classes]
        axes[1, 1].bar(classes, precisions)
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].set_title('Per-Class Precision')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_xlabel('Class')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Bio-CBAM model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='fer2013',
                       choices=['fer2013', 'ckplus', 'jaffe'],
                       help='Dataset to evaluate on')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--num-classes', type=int, default=7,
                       choices=[4, 7], help='Number of emotion classes')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print(f"Creating Bio-CBAM model...")
    model = create_bio_cbam(
        num_classes=args.num_classes,
        backbone=args.backbone,
        device=args.device
    )
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    
    # Create data loaders
    print(f"Loading {args.dataset} dataset...")
    dataloaders = create_dataloaders(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=dataloaders['test'],
        device=args.device,
        num_classes=args.num_classes
    )
    
    # Evaluate
    print("Evaluating model...")
    metrics, preds, labels = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Loss:      {metrics['loss']:.4f}")
    print("="*50)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    evaluator.plot_confusion_matrix(cm, 
                                   save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Plot metrics
    evaluator.plot_metrics(metrics, 
                          save_path=os.path.join(args.output_dir, 'metrics_plot.png'))
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
