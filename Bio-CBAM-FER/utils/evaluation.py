"""Evaluation utilities for Bio-CBAM"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, List
import json


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    return metrics


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with per-class metrics
    """
    num_classes = len(np.unique(y_true))
    
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []
    
    for class_id in range(num_classes):
        mask = y_true == class_id
        support = mask.sum()
        support_per_class.append(int(support))
        
        if support == 0:
            precision_per_class.append(0.0)
            recall_per_class.append(0.0)
            f1_per_class.append(0.0)
        else:
            pred_mask = y_pred == class_id
            tp = (mask & pred_mask).sum()
            fp = (~mask & pred_mask).sum()
            fn = (mask & ~pred_mask).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            precision_per_class.append(float(precision))
            recall_per_class.append(float(recall))
            f1_per_class.append(float(f1))
    
    return {
        'precision': precision_per_class,
        'recall': recall_per_class,
        'f1': f1_per_class,
        'support': support_per_class
    }


def statistical_analysis(accuracies: List[float]) -> Dict:
    """
    Perform statistical analysis on multiple runs.
    
    Args:
        accuracies: List of accuracies from multiple runs
    
    Returns:
        Dictionary with statistical metrics
    """
    accuracies = np.array(accuracies)
    
    stats = {
        'mean': float(accuracies.mean()),
        'std': float(accuracies.std()),
        'min': float(accuracies.min()),
        'max': float(accuracies.max()),
        'median': float(np.median(accuracies)),
        'q25': float(np.percentile(accuracies, 25)),
        'q75': float(np.percentile(accuracies, 75)),
        'range': float(accuracies.max() - accuracies.min()),
        'cv': float(accuracies.std() / accuracies.mean() * 100),  # Coefficient of variation
        'num_runs': len(accuracies)
    }
    
    return stats


def compute_class_imbalance(y: np.ndarray) -> Dict:
    """
    Compute class imbalance metrics.
    
    Args:
        y: Labels
    
    Returns:
        Dictionary with imbalance metrics
    """
    unique, counts = np.unique(y, return_counts=True)
    
    imbalance = {
        'class_distribution': {str(int(c)): int(count) for c, count in zip(unique, counts)},
        'total_samples': int(y.shape[0]),
        'num_classes': int(len(unique)),
        'imbalance_ratio': float(counts.max() / counts.min()),
        'minority_class': int(unique[counts.argmin()]),
        'majority_class': int(unique[counts.argmax()])
    }
    
    return imbalance


def save_metrics(metrics: Dict, filepath: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary with metrics
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        filepath: Path to metrics file
    
    Returns:
        Dictionary with metrics
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def print_metrics(metrics: Dict) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary with metrics
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted):  {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    print("="*60)


def print_per_class_metrics(metrics: Dict, class_names: List[str] = None) -> None:
    """
    Print per-class metrics in a formatted way.
    
    Args:
        metrics: Dictionary with per-class metrics
        class_names: List of class names
    """
    num_classes = len(metrics['precision'])
    
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-"*80)
    
    for i in range(num_classes):
        print(f"{class_names[i]:<15} {metrics['precision'][i]:<15.4f} "
              f"{metrics['recall'][i]:<15.4f} {metrics['f1'][i]:<15.4f} "
              f"{metrics['support'][i]:<10}")
    
    print("="*80)
