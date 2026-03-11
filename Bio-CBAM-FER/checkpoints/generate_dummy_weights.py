"""
Generate dummy pre-trained weights for testing purposes.

This script creates a dummy model checkpoint that can be used
for testing evaluation and inference pipelines.

Author: Hind Laziri
Date: 2026
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_bio_cbam


def generate_dummy_weights(output_path: str = './checkpoints/best_model.pth',
                          num_classes: int = 7,
                          backbone: str = 'resnet50'):
    """
    Generate and save dummy pre-trained weights.
    
    Args:
        output_path: Path to save weights
        num_classes: Number of emotion classes
        backbone: Backbone architecture
    """
    print(f"Generating dummy pre-trained weights...")
    print(f"  - Model: Bio-CBAM with {backbone}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Output: {output_path}")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_bio_cbam(
        num_classes=num_classes,
        backbone=backbone,
        device=device
    )
    
    # Save weights
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Weights saved successfully!")
    print(f"  - File size: {os.path.getsize(output_path) / 1e6:.2f} MB")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")


if __name__ == '__main__':
    # Generate for different configurations
    configs = [
        ('best_model_resnet50_7class.pth', 7, 'resnet50'),
        ('best_model_resnet50_4class.pth', 4, 'resnet50'),
        ('best_model_resnet18_7class.pth', 7, 'resnet18'),
    ]
    
    for filename, num_classes, backbone in configs:
        output_path = os.path.join('./checkpoints', filename)
        generate_dummy_weights(output_path, num_classes, backbone)
        print()
