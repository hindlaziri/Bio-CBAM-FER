# Model Checkpoints

This directory contains pre-trained Bio-CBAM model weights.

## Available Models

### Pre-trained Weights

- `best_model_resnet50_7class.pth`: Best model (ResNet-50, 7-class)
  - Accuracy: 74.8%
  - Size: ~100 MB
  - Dataset: FER-2013

- `best_model_resnet50_4class.pth`: Best model (ResNet-50, 4-class)
  - Accuracy: 94.7%
  - Size: ~100 MB
  - Dataset: FER-2013 (4-class subset)

- `best_model_resnet18_7class.pth`: Lightweight model (ResNet-18, 7-class)
  - Accuracy: 72.3%
  - Size: ~50 MB
  - Dataset: FER-2013

## Generating Dummy Weights

To generate dummy weights for testing:

```bash
python generate_dummy_weights.py
```

## Loading Weights

```python
import torch
from models import create_bio_cbam

# Create model
model = create_bio_cbam(num_classes=7, backbone='resnet50')

# Load weights
checkpoint = torch.load('checkpoints/best_model_resnet50_7class.pth')
model.load_state_dict(checkpoint)

# Use for inference
model.eval()
with torch.no_grad():
    logits, attention_maps = model(images)
```

## Training and Saving Weights

During training, the best model is automatically saved:

```bash
python train.py --checkpoint-dir ./checkpoints
```

This will create:
- `checkpoints/best_model.pth`: Best model based on validation accuracy
- `checkpoints/final_model.pth`: Final model after all epochs
- `checkpoints/training_history.json`: Training metrics

## Checkpoint Contents

Each `.pth` file contains:
- Model state dictionary (weights and biases)
- Compatible with PyTorch's `load_state_dict()`
- Can be loaded into any Bio-CBAM model with matching architecture

## Model Architecture Info

All models use:
- **Backbone**: ResNet-18 or ResNet-50 (pretrained on ImageNet)
- **Attention**: CBAM (Channel + Spatial)
- **fMRI Prior**: Spatial attention guided by fMRI priors
- **Input**: 224×224 RGB images
- **Output**: Emotion class logits

## Downloading Pre-trained Weights

Full pre-trained weights are available at:
- **Zenodo**: https://doi.org/10.5281/zenodo.18818259
- **GitHub**: https://github.com/hindlaziri/Bio-CBAM-FER/releases

## File Size Reference

| Model | Backbone | Classes | Size |
|-------|----------|---------|------|
| best_model_resnet50_7class.pth | ResNet-50 | 7 | ~100 MB |
| best_model_resnet50_4class.pth | ResNet-50 | 4 | ~100 MB |
| best_model_resnet18_7class.pth | ResNet-18 | 7 | ~50 MB |
