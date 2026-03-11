"""Visualization utilities for Bio-CBAM attention maps and results"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import cv2


def visualize_attention_maps(image: np.ndarray, attention_map: np.ndarray,
                            title: str = 'Attention Map',
                            save_path: Optional[str] = None) -> None:
    """
    Visualize attention map overlaid on original image.
    
    Args:
        image: Original image (H, W, 3)
        attention_map: Attention map (H, W)
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    # Overlay
    attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    overlay = image.copy().astype(float)
    overlay[..., 0] = overlay[..., 0] * (1 - 0.5 * attention_normalized)
    overlay[..., 1] = overlay[..., 1] * (1 - 0.3 * attention_normalized)
    overlay[..., 2] = overlay[..., 2] * (1 - 0.3 * attention_normalized)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_fmri_prior(fmri_prior: np.ndarray, 
                        save_path: Optional[str] = None) -> None:
    """
    Visualize fMRI-derived spatial prior.
    
    Args:
        fmri_prior: fMRI prior map (H, W)
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize
    prior_norm = (fmri_prior - fmri_prior.min()) / (fmri_prior.max() - fmri_prior.min() + 1e-8)
    
    plt.imshow(prior_norm, cmap='viridis')
    plt.colorbar(label='Prior Strength')
    plt.title('fMRI-Guided Spatial Prior', fontsize=14, fontweight='bold')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_per_class_metrics(metrics: Dict, class_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        metrics: Dictionary with per-class metrics
        class_names: List of class names
        save_path: Path to save figure
    """
    if class_names is None:
        num_classes = len(metrics.get('precision', []))
        class_names = [str(i) for i in range(num_classes)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision
    axes[0].bar(class_names, metrics['precision'])
    axes[0].set_ylabel('Precision', fontsize=11)
    axes[0].set_title('Per-Class Precision', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Recall
    axes[1].bar(class_names, metrics['recall'])
    axes[1].set_ylabel('Recall', fontsize=11)
    axes[1].set_title('Per-Class Recall', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # F1-Score
    axes[2].bar(class_names, metrics['f1'])
    axes[2].set_ylabel('F1-Score', fontsize=11)
    axes[2].set_title('Per-Class F1-Score', fontsize=12, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_attention_grid(images: List[np.ndarray], 
                         attention_maps: List[np.ndarray],
                         predictions: List[int],
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Create grid of images with attention maps.
    
    Args:
        images: List of images
        attention_maps: List of attention maps
        predictions: List of predictions
        class_names: List of class names
        save_path: Path to save figure
    """
    num_samples = min(len(images), 8)
    
    if class_names is None:
        class_names = [str(i) for i in range(max(predictions) + 1)]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 6))
    
    for i in range(num_samples):
        # Image
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'Pred: {class_names[predictions[i]]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Attention map
        axes[1, i].imshow(attention_maps[i], cmap='hot')
        axes[1, i].axis('off')
    
    plt.suptitle('Attention Maps Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
