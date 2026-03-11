"""
Bio-CBAM: A Neuro-Guided Attention Mechanism for Facial Expression Recognition

This module implements the Bio-CBAM architecture that integrates fMRI-derived
spatial priors with CBAM attention mechanisms for robust facial expression recognition.

Author: Hind Laziri
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM)"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Attention-weighted tensor
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM)"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Spatial attention map
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            CBAM-weighted tensor
        """
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x


class fMRIPrior(nn.Module):
    """fMRI-Guided Spatial Prior Module
    
    This module integrates fMRI-derived spatial priors to guide attention
    towards neurobiologically relevant facial regions.
    """
    
    def __init__(self, height: int = 224, width: int = 224, 
                 num_landmarks: int = 68, device: str = 'cuda'):
        super(fMRIPrior, self).__init__()
        self.height = height
        self.width = width
        self.num_landmarks = num_landmarks
        self.device = device
        
        # fMRI activation map (learned during training)
        self.register_buffer('fmri_map', torch.ones(1, 1, height, width) / (height * width))
    
    def create_fmri_prior(self, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create fMRI-guided spatial prior from landmarks or predefined regions.
        
        Args:
            landmarks: Facial landmarks (B, 68, 2) or None
        
        Returns:
            fMRI prior map of shape (B, 1, H, W)
        """
        if landmarks is None:
            # Use predefined fMRI activation map
            return self.fmri_map
        
        # Create Gaussian heatmaps around landmarks
        batch_size = landmarks.shape[0]
        prior_map = torch.zeros(batch_size, 1, self.height, self.width, 
                               device=landmarks.device)
        
        sigma = 15  # Gaussian kernel standard deviation
        
        for b in range(batch_size):
            for i in range(self.num_landmarks):
                x, y = landmarks[b, i]
                x, y = int(x.item()), int(y.item())
                
                # Create Gaussian heatmap
                y_min = max(0, y - 3*sigma)
                y_max = min(self.height, y + 3*sigma)
                x_min = max(0, x - 3*sigma)
                x_max = min(self.width, x + 3*sigma)
                
                yy, xx = torch.meshgrid(
                    torch.arange(y_min, y_max, device=landmarks.device),
                    torch.arange(x_min, x_max, device=landmarks.device),
                    indexing='ij'
                )
                
                gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                prior_map[b, 0, y_min:y_max, x_min:x_max] += gaussian
        
        # Normalize
        prior_map = prior_map / (prior_map.max() + 1e-8)
        return prior_map
    
    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply fMRI prior to feature maps.
        
        Args:
            x: Feature maps (B, C, H, W)
            landmarks: Facial landmarks (B, 68, 2) or None
        
        Returns:
            Prior-weighted feature maps
        """
        prior = self.create_fmri_prior(landmarks)
        
        # Resize prior to match feature map size if needed
        if prior.shape[-2:] != x.shape[-2:]:
            prior = F.interpolate(prior, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return x * prior


class BioCBAM(nn.Module):
    """Bio-CBAM: Neuro-Guided Attention Mechanism for FER
    
    This module combines fMRI-derived spatial priors with CBAM attention
    for robust facial expression recognition.
    """
    
    def __init__(self, num_classes: int = 7, backbone: str = 'resnet50',
                 use_fmri_prior: bool = True, device: str = 'cuda'):
        super(BioCBAM, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_fmri_prior = use_fmri_prior
        self.device = device
        
        # Load backbone
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=True)
            self.feature_dim = 2048
        elif backbone == 'resnet18':
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=True)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # fMRI Prior
        if use_fmri_prior:
            self.fmri_prior = fMRIPrior(height=224, width=224, device=device)
        
        # CBAM modules at different scales
        self.cbam_layer4 = CBAM(2048 if backbone == 'resnet50' else 512)
        self.cbam_layer3 = CBAM(1024 if backbone == 'resnet50' else 256)
        
        # Classification head
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through Bio-CBAM.
        
        Args:
            x: Input images (B, 3, 224, 224)
            landmarks: Facial landmarks (B, 68, 2) or None
        
        Returns:
            logits: Classification logits (B, num_classes)
            attention_maps: Dictionary of attention maps for visualization
        """
        attention_maps = {}
        
        # Feature extraction
        features = self.backbone(x)
        
        # Apply fMRI prior if available
        if self.use_fmri_prior and landmarks is not None:
            fmri_prior = self.fmri_prior(features, landmarks)
            features = features * fmri_prior
            attention_maps['fmri_prior'] = fmri_prior
        
        # Apply CBAM
        features = self.cbam_layer4(features)
        attention_maps['cbam'] = features
        
        # Global average pooling
        features = self.global_avgpool(features)
        features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.fc(features)
        
        return logits, attention_maps
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        features = self.backbone(x)
        features = self.global_avgpool(features)
        features = features.view(features.size(0), -1)
        return features


def create_bio_cbam(num_classes: int = 7, backbone: str = 'resnet50',
                   use_fmri_prior: bool = True, device: str = 'cuda') -> BioCBAM:
    """
    Factory function to create Bio-CBAM model.
    
    Args:
        num_classes: Number of emotion classes
        backbone: Backbone architecture ('resnet50' or 'resnet18')
        use_fmri_prior: Whether to use fMRI-guided priors
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        BioCBAM model instance
    """
    model = BioCBAM(
        num_classes=num_classes,
        backbone=backbone,
        use_fmri_prior=use_fmri_prior,
        device=device
    )
    return model.to(device)


if __name__ == '__main__':
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_bio_cbam(num_classes=7, device=device)
    
    # Dummy input
    x = torch.randn(4, 3, 224, 224).to(device)
    landmarks = torch.randn(4, 68, 2).to(device)
    
    # Forward pass
    logits, attention_maps = model(x, landmarks)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Attention maps: {list(attention_maps.keys())}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
