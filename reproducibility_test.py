"""
Reproducibility Testing Script for Bio-CBAM

Tests model reproducibility across multiple runs with different random seeds.

Author: Hind Laziri
Date: 2026
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from typing import Dict, List
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.bio_cbam import create_bio_cbam
from dataset_scripts.dataset_loader import create_dataloaders


class ReproducibilityTester:
    """Test model reproducibility"""
    
    def __init__(self, device: str = 'cuda', num_classes: int = 7):
        """
        Args:
            device: Device to use
            num_classes: Number of emotion classes
        """
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.results = {
            'accuracies': [],
            'losses': [],
            'seeds': []
        }
    
    def evaluate_model(self, model: nn.Module, test_loader) -> float:
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def run_multiple_seeds(self, seeds: List[int], test_loader,
                          num_classes: int = 7, backbone: str = 'resnet50'):
        """
        Run evaluation with multiple random seeds.
        
        Args:
            seeds: List of random seeds to test
            test_loader: Test data loader
            num_classes: Number of emotion classes
            backbone: Backbone architecture
        """
        print(f"Running reproducibility test with {len(seeds)} seeds...")
        
        for seed in tqdm(seeds):
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create model
            model = create_bio_cbam(
                num_classes=num_classes,
                backbone=backbone,
                device=self.device
            )
            
            # Evaluate
            accuracy, loss = self.evaluate_model(model, test_loader)
            
            self.results['seeds'].append(seed)
            self.results['accuracies'].append(accuracy)
            self.results['losses'].append(loss)
        
        self._print_statistics()
    
    def _print_statistics(self):
        """Print reproducibility statistics"""
        accuracies = np.array(self.results['accuracies'])
        losses = np.array(self.results['losses'])
        
        print("\n" + "="*60)
        print("REPRODUCIBILITY TEST RESULTS")
        print("="*60)
        print(f"Number of runs: {len(accuracies)}")
        print(f"\nAccuracy Statistics:")
        print(f"  Mean:     {accuracies.mean():.4f}%")
        print(f"  Std Dev:  {accuracies.std():.4f}%")
        print(f"  Min:      {accuracies.min():.4f}%")
        print(f"  Max:      {accuracies.max():.4f}%")
        print(f"  Range:    {accuracies.max() - accuracies.min():.4f}%")
        
        print(f"\nLoss Statistics:")
        print(f"  Mean:     {losses.mean():.4f}")
        print(f"  Std Dev:  {losses.std():.4f}")
        print(f"  Min:      {losses.min():.4f}")
        print(f"  Max:      {losses.max():.4f}")
        print("="*60)
    
    def plot_results(self, save_path: str = './reproducibility_results.png'):
        """Plot reproducibility test results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.results['seeds'], self.results['accuracies'], 'o-', linewidth=2, markersize=8)
        axes[0].axhline(y=np.mean(self.results['accuracies']), color='r', linestyle='--', label='Mean')
        axes[0].fill_between(
            range(len(self.results['seeds'])),
            np.mean(self.results['accuracies']) - np.std(self.results['accuracies']),
            np.mean(self.results['accuracies']) + np.std(self.results['accuracies']),
            alpha=0.2, color='red'
        )
        axes[0].set_xlabel('Random Seed')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Accuracy Across Different Seeds')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Loss plot
        axes[1].plot(self.results['seeds'], self.results['losses'], 'o-', linewidth=2, markersize=8, color='orange')
        axes[1].axhline(y=np.mean(self.results['losses']), color='r', linestyle='--', label='Mean')
        axes[1].fill_between(
            range(len(self.results['seeds'])),
            np.mean(self.results['losses']) - np.std(self.results['losses']),
            np.mean(self.results['losses']) + np.std(self.results['losses']),
            alpha=0.2, color='red'
        )
        axes[1].set_xlabel('Random Seed')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Across Different Seeds')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    
    def save_results(self, save_path: str = './reproducibility_results.json'):
        """Save reproducibility test results"""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Bio-CBAM reproducibility')
    parser.add_argument('--dataset', type=str, default='fer2013',
                       choices=['fer2013', 'ckplus', 'jaffe'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--num-classes', type=int, default=7,
                       choices=[4, 7], help='Number of emotion classes')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of reproducibility test runs')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='./reproducibility',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate random seeds
    seeds = [42, 123, 456, 789, 999][:args.num_runs]
    
    # Create data loaders
    print(f"Loading {args.dataset} dataset...")
    dataloaders = create_dataloaders(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    # Create tester
    tester = ReproducibilityTester(
        device=args.device,
        num_classes=args.num_classes
    )
    
    # Run reproducibility test
    tester.run_multiple_seeds(
        seeds=seeds,
        test_loader=dataloaders['test'],
        num_classes=args.num_classes,
        backbone=args.backbone
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, 'reproducibility_results.json')
    tester.save_results(results_path)
    
    # Plot results
    plot_path = os.path.join(args.output_dir, 'reproducibility_plot.png')
    tester.plot_results(plot_path)
    
    print(f"\nReproducibility test completed!")
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
