"""
Generate fMRI-derived spatial priors for Bio-CBAM

This script creates fMRI-guided spatial attention maps based on
neurobiological evidence from facial emotion processing studies.

Author: Hind Laziri
Date: 2026
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple
import os


class fMRIPriorGenerator:
    """Generate fMRI-derived spatial priors"""
    
    # Key brain regions for emotion processing (normalized coordinates)
    EMOTION_REGIONS = {
        'amygdala': {'center': (0.35, 0.45), 'sigma': 0.15},
        'orbitofrontal': {'center': (0.5, 0.3), 'sigma': 0.12},
        'insula': {'center': (0.65, 0.4), 'sigma': 0.1},
        'superior_temporal': {'center': (0.7, 0.5), 'sigma': 0.12},
    }
    
    # Facial regions important for emotion recognition
    FACIAL_REGIONS = {
        'eyes': {'center': (0.5, 0.35), 'sigma': 0.15},
        'mouth': {'center': (0.5, 0.65), 'sigma': 0.12},
        'eyebrows': {'center': (0.5, 0.25), 'sigma': 0.1},
        'nose': {'center': (0.5, 0.5), 'sigma': 0.08},
    }
    
    def __init__(self, height: int = 224, width: int = 224):
        """
        Args:
            height: Height of prior maps
            width: Width of prior maps
        """
        self.height = height
        self.width = width
    
    def create_gaussian_prior(self, center: Tuple[float, float], 
                             sigma: float) -> np.ndarray:
        """
        Create Gaussian prior map.
        
        Args:
            center: Normalized center coordinates (0-1)
            sigma: Normalized standard deviation (0-1)
        
        Returns:
            Prior map (H, W)
        """
        # Convert normalized coordinates to pixel coordinates
        center_y = int(center[0] * self.height)
        center_x = int(center[1] * self.width)
        sigma_y = sigma * self.height
        sigma_x = sigma * self.width
        
        # Create coordinate grids
        y = np.arange(self.height)
        x = np.arange(self.width)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        
        # Create Gaussian
        prior = np.exp(-((yy - center_y)**2 / (2 * sigma_y**2) + 
                        (xx - center_x)**2 / (2 * sigma_x**2)))
        
        return prior
    
    def create_facial_prior(self) -> np.ndarray:
        """
        Create prior based on facial regions important for emotion recognition.
        
        Returns:
            Facial prior map (H, W)
        """
        prior = np.zeros((self.height, self.width))
        
        # Combine facial regions with weights
        weights = {
            'eyes': 0.4,
            'mouth': 0.3,
            'eyebrows': 0.2,
            'nose': 0.1
        }
        
        for region, params in self.FACIAL_REGIONS.items():
            region_prior = self.create_gaussian_prior(
                params['center'], 
                params['sigma']
            )
            prior += weights[region] * region_prior
        
        # Normalize
        prior = prior / (prior.max() + 1e-8)
        
        return prior
    
    def create_emotion_specific_prior(self, emotion: str) -> np.ndarray:
        """
        Create emotion-specific prior based on fMRI activation patterns.
        
        Args:
            emotion: Emotion class ('angry', 'happy', 'sad', 'fear', 'neutral', 'disgust', 'surprise')
        
        Returns:
            Emotion-specific prior map (H, W)
        """
        # Base facial prior
        prior = self.create_facial_prior()
        
        # Emotion-specific adjustments based on fMRI studies
        emotion_weights = {
            'angry': {'eyes': 0.5, 'eyebrows': 0.3, 'mouth': 0.2},
            'happy': {'mouth': 0.5, 'eyes': 0.3, 'eyebrows': 0.2},
            'sad': {'eyes': 0.4, 'mouth': 0.4, 'eyebrows': 0.2},
            'fear': {'eyes': 0.6, 'mouth': 0.3, 'eyebrows': 0.1},
            'neutral': {'eyes': 0.4, 'mouth': 0.3, 'nose': 0.3},
            'disgust': {'nose': 0.4, 'mouth': 0.4, 'eyes': 0.2},
            'surprise': {'eyes': 0.5, 'mouth': 0.4, 'eyebrows': 0.1}
        }
        
        if emotion in emotion_weights:
            prior = np.zeros((self.height, self.width))
            for region, weight in emotion_weights[emotion].items():
                if region in self.FACIAL_REGIONS:
                    region_prior = self.create_gaussian_prior(
                        self.FACIAL_REGIONS[region]['center'],
                        self.FACIAL_REGIONS[region]['sigma']
                    )
                    prior += weight * region_prior
            
            prior = prior / (prior.max() + 1e-8)
        
        return prior
    
    def create_combined_prior(self) -> np.ndarray:
        """
        Create combined prior from all regions.
        
        Returns:
            Combined prior map (H, W)
        """
        prior = np.zeros((self.height, self.width))
        
        # Add facial regions
        for region, params in self.FACIAL_REGIONS.items():
            region_prior = self.create_gaussian_prior(
                params['center'],
                params['sigma']
            )
            prior += 0.7 * region_prior
        
        # Normalize
        prior = prior / (prior.max() + 1e-8)
        
        return prior
    
    def apply_smoothing(self, prior: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Apply Gaussian smoothing to prior.
        
        Args:
            prior: Prior map
            sigma: Smoothing standard deviation
        
        Returns:
            Smoothed prior map
        """
        return gaussian_filter(prior, sigma=sigma)
    
    def save_prior(self, prior: np.ndarray, filepath: str) -> None:
        """
        Save prior to NPY file.
        
        Args:
            prior: Prior map
            filepath: Path to save file
        """
        np.save(filepath, prior)
        print(f"Prior saved to {filepath}")
    
    def load_prior(self, filepath: str) -> np.ndarray:
        """
        Load prior from NPY file.
        
        Args:
            filepath: Path to prior file
        
        Returns:
            Prior map
        """
        return np.load(filepath)


def generate_all_priors(output_dir: str = './priors', height: int = 224, width: int = 224):
    """
    Generate all fMRI priors and save to files.
    
    Args:
        output_dir: Directory to save priors
        height: Height of prior maps
        width: Width of prior maps
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = fMRIPriorGenerator(height=height, width=width)
    
    # Generate emotion-specific priors
    emotions = ['angry', 'happy', 'sad', 'fear', 'neutral', 'disgust', 'surprise']
    
    print("Generating fMRI-derived spatial priors...")
    
    for emotion in emotions:
        prior = generator.create_emotion_specific_prior(emotion)
        prior = generator.apply_smoothing(prior, sigma=2.0)
        
        filepath = os.path.join(output_dir, f'prior_{emotion}.npy')
        generator.save_prior(prior, filepath)
        print(f"✓ Generated prior for {emotion}")
    
    # Generate combined prior
    combined_prior = generator.create_combined_prior()
    combined_prior = generator.apply_smoothing(combined_prior, sigma=2.0)
    filepath = os.path.join(output_dir, 'prior_combined.npy')
    generator.save_prior(combined_prior, filepath)
    print(f"✓ Generated combined prior")
    
    # Generate facial prior
    facial_prior = generator.create_facial_prior()
    facial_prior = generator.apply_smoothing(facial_prior, sigma=2.0)
    filepath = os.path.join(output_dir, 'prior_facial.npy')
    generator.save_prior(facial_prior, filepath)
    print(f"✓ Generated facial prior")
    
    print(f"\nAll priors saved to {output_dir}")


if __name__ == '__main__':
    generate_all_priors(output_dir='./priors', height=224, width=224)
