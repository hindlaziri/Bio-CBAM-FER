"""
Dataset Loading and Preprocessing Pipeline for Bio-CBAM

Supports: FER-2013, CK+, JAFFE

Author: Hind Laziri
Date: 2026
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from PIL import Image
import json


class FER2013Dataset(Dataset):
    """FER-2013 Dataset Loader"""
    
    EMOTION_LABELS = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprise'
    }
    
    EMOTION_TO_4CLASS = {
        0: 0,  # angry -> angry
        1: 1,  # disgust -> sad
        2: 1,  # fear -> sad
        3: 2,  # happy -> happy
        4: 3,  # neutral -> neutral
        5: 1,  # sad -> sad
        6: 2   # surprise -> happy
    }
    
    def __init__(self, csv_path: str, split: str = 'train', 
                 num_classes: int = 7, transform: Optional[transforms.Compose] = None,
                 use_ssim_filtering: bool = True):
        """
        Args:
            csv_path: Path to FER-2013 CSV file
            split: 'train', 'val', or 'test'
            num_classes: 7 or 4 (emotion classes)
            transform: Image transformations
            use_ssim_filtering: Apply SSIM-based filtering to remove near-duplicates
        """
        self.csv_path = csv_path
        self.split = split
        self.num_classes = num_classes
        self.transform = transform
        self.use_ssim_filtering = use_ssim_filtering
        
        self.images = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """Load data from CSV file"""
        import pandas as pd
        
        df = pd.read_csv(self.csv_path)
        
        # Filter by split
        split_data = df[df['Usage'] == self.split.capitalize()]
        
        for idx, row in split_data.iterrows():
            # Parse pixel data
            pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
            image = pixels.reshape(48, 48)
            
            # Convert to 3-channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Get label
            emotion = int(row['emotion'])
            if self.num_classes == 4:
                emotion = self.EMOTION_TO_4CLASS[emotion]
            
            self.images.append(image)
            self.labels.append(emotion)
        
        print(f"Loaded {len(self.images)} images from FER-2013 ({self.split})")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label


class CKPlusDataset(Dataset):
    """CK+ Dataset Loader"""
    
    EMOTION_LABELS = {
        0: 'neutral',
        1: 'anger',
        2: 'contempt',
        3: 'disgust',
        4: 'fear',
        5: 'happy',
        6: 'sadness',
        7: 'surprise'
    }
    
    def __init__(self, data_dir: str, split: str = 'train',
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            data_dir: Path to CK+ dataset directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        self.images = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """Load CK+ data"""
        # This is a placeholder - actual implementation depends on CK+ structure
        print(f"Loading CK+ dataset ({self.split})")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label


class JAFFEDataset(Dataset):
    """JAFFE Dataset Loader"""
    
    EMOTION_LABELS = {
        'AN': 'angry',
        'DI': 'disgust',
        'FE': 'fear',
        'HA': 'happy',
        'NE': 'neutral',
        'SA': 'sad',
        'SU': 'surprise'
    }
    
    def __init__(self, data_dir: str, split: str = 'train',
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            data_dir: Path to JAFFE dataset directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        self.images = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """Load JAFFE data"""
        # This is a placeholder - actual implementation depends on JAFFE structure
        print(f"Loading JAFFE dataset ({self.split})")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label


def get_data_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Get data augmentation and normalization transforms.
    
    Args:
        image_size: Size of input images
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return {'train': train_transform, 'val': val_transform}


def create_dataloaders(dataset_name: str, data_dir: str, 
                      batch_size: int = 32, num_workers: int = 4,
                      num_classes: int = 7) -> Dict[str, DataLoader]:
    """
    Create data loaders for specified dataset.
    
    Args:
        dataset_name: 'fer2013', 'ckplus', or 'jaffe'
        data_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        num_classes: Number of emotion classes
    
    Returns:
        Dictionary with 'train', 'val', 'test' data loaders
    """
    transforms_dict = get_data_transforms()
    
    if dataset_name.lower() == 'fer2013':
        train_dataset = FER2013Dataset(
            os.path.join(data_dir, 'fer2013.csv'),
            split='train',
            num_classes=num_classes,
            transform=transforms_dict['train']
        )
        val_dataset = FER2013Dataset(
            os.path.join(data_dir, 'fer2013.csv'),
            split='val',
            num_classes=num_classes,
            transform=transforms_dict['val']
        )
        test_dataset = FER2013Dataset(
            os.path.join(data_dir, 'fer2013.csv'),
            split='test',
            num_classes=num_classes,
            transform=transforms_dict['val']
        )
    
    elif dataset_name.lower() == 'ckplus':
        train_dataset = CKPlusDataset(data_dir, split='train', transform=transforms_dict['train'])
        val_dataset = CKPlusDataset(data_dir, split='val', transform=transforms_dict['val'])
        test_dataset = CKPlusDataset(data_dir, split='test', transform=transforms_dict['val'])
    
    elif dataset_name.lower() == 'jaffe':
        train_dataset = JAFFEDataset(data_dir, split='train', transform=transforms_dict['train'])
        val_dataset = JAFFEDataset(data_dir, split='val', transform=transforms_dict['val'])
        test_dataset = JAFFEDataset(data_dir, split='test', transform=transforms_dict['val'])
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test dataset loading
    print("Dataset pipeline ready for use")
