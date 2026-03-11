# Bio-CBAM: A Neuro-Guided Attention Mechanism for Robust Facial Expression Recognition

## Overview

Bio-CBAM integrates fMRI-derived spatial priors with Convolutional Block Attention Modules (CBAM) for facial expression recognition in unconstrained environments. This repository contains the complete implementation, preprocessing pipeline, and evaluation code for the paper submitted to **Multimedia Tools and Applications (MTAP)**.

## Key Features

- **Neuro-Guided Attention**: Integrates fMRI-derived spatial priors to guide attention towards neurobiologically relevant facial regions
- **CBAM Architecture**: Combines channel and spatial attention mechanisms for robust feature learning
- **Multi-Dataset Support**: Supports FER-2013, CK+, and JAFFE datasets
- **Comprehensive Evaluation**: Includes training, evaluation, and reproducibility testing scripts
- **High Performance**: Achieves 94.7% accuracy on FER-2013 4-class subset and 74.8% on 7-class benchmark
- **Reproducibility**: Validated across 5 independent runs with different random seeds

## Performance

| Dataset | Classes | Accuracy | Precision | Recall | F1-Score |
|---------|---------|----------|-----------|--------|----------|
| FER-2013 | 7-class | 74.8% | 75.2% | 74.8% | 74.9% |
| FER-2013 | 4-class | 94.7% | 94.8% | 94.7% | 94.7% |
| CK+ | 7-class | 96.3% | 96.5% | 96.3% | 96.4% |
| JAFFE | 7-class | 93.7% | 93.9% | 93.7% | 93.8% |

## Project Structure

```
Bio-CBAM-FER/
├── models/
│   ├── __init__.py
│   └── bio_cbam.py              # Bio-CBAM model implementation
├── dataset_scripts/
│   ├── __init__.py
│   └── dataset_loader.py        # Dataset loading and preprocessing
├── weights/
│   └── (pre-trained model weights)
├── docs/
│   ├── INSTALL.md               # Installation guide
│   └── USAGE.md                 # Usage guide
├── train.py                     # Training script
├── eval.py                      # Evaluation script
├── reproducibility_test.py      # Reproducibility testing
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore
```

## Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/hindlaziri/Bio-CBAM-FER.git
cd Bio-CBAM-FER

# Create virtual environment
conda create -n bio-cbam python=3.9
conda activate bio-cbam

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For detailed installation instructions, see [docs/INSTALL.md](docs/INSTALL.md).

## Usage

### Training

```bash
python train.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 0.001 \
    --backbone resnet50 \
    --device cuda
```

### Evaluation

```bash
python eval.py \
    --model-path ./checkpoints/best_model.pth \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --output-dir ./results
```

### Reproducibility Testing

```bash
python reproducibility_test.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --num-runs 5 \
    --output-dir ./reproducibility
```

For detailed usage instructions, see [docs/USAGE.md](docs/USAGE.md).

## Model Architecture

### Bio-CBAM Components

1. **fMRI Prior Module**: Generates spatial attention maps from fMRI-derived priors
2. **Channel Attention**: Learns channel-wise feature importance
3. **Spatial Attention**: Learns spatial feature importance
4. **Classification Head**: Emotion classification from learned features

### Backbone Options

- **ResNet-18**: Lightweight, faster training
- **ResNet-50**: Better accuracy, larger model

## Datasets

### FER-2013
- 35,887 images
- 7 emotion classes
- 48×48 grayscale images
- Download: https://www.kaggle.com/datasets/msambare/fer2013

### CK+
- 593 sequences from 123 subjects
- 7 emotion classes
- High-quality, controlled environment
- Download: https://www.jeffcohn.com/databases/

### JAFFE
- 213 images from 10 female subjects
- 7 emotion classes
- Diverse expression variations
- Download: https://zenodo.org/record/3451524

## Key Features

### Data Preprocessing
- SSIM-based filtering to remove near-duplicate images
- Prevents data leakage between train/val/test splits
- Quantitative analysis of dataset entropy

### Attention Mechanisms
- Channel Attention Module (CAM)
- Spatial Attention Module (SAM)
- fMRI-guided spatial priors

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Per-class metrics
- Confusion matrix
- Cross-validation analysis

### Reproducibility
- Fixed random seeds (42, 123, 456, 789, 999)
- Multiple independent runs
- Statistical analysis of results
- Detailed logging

## Citation

If you use Bio-CBAM in your research, please cite:

```bibtex
@article{laziri2026biocbam,
  title={Bio-CBAM: A Neuro-Guided Attention Mechanism for Robust Facial Expression Recognition in the Wild},
  author={Laziri, Hind and Riffi, Mohammed Essaid},
  journal={Multimedia Tools and Applications},
  year={2026},
  publisher={Springer}
}
```

## Code Availability

Complete code and pre-trained weights are available at:
- **GitHub**: https://github.com/hindlaziri/Bio-CBAM-FER
- **Zenodo**: https://doi.org/10.5281/zenodo.18818259

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- FER-2013 dataset: Goodfellow et al.
- CK+ dataset: Kanade et al.
- JAFFE dataset: Lyons et al.
- ResNet backbone: He et al.
- CBAM: Woo et al.

## Contact

For questions or issues, please contact:
- **Author**: Hind Laziri
- **Email**: (as provided to MTAP)
- **Affiliation**: Department of Computer Science, Chouaib Doukali Faculty, El Jadida, Morocco

## References

1. Goodfellow, I. J., et al. (2013). Challenges in representation learning: A report on three machine learning contests. ICONIP.
2. Kanade, T., et al. (2000). The CMU pose, illumination, and expression database. IEEE TPAMI.
3. Lyons, M. J., et al. (1998). Automatic classification of single facial images. IEEE TPAMI.
4. He, K., et al. (2016). Deep residual learning for image recognition. CVPR.
5. Woo, S., et al. (2018). CBAM: Convolutional block attention module. ECCV.

---

**Status**: ✅ Ready for Publication
**Last Updated**: February 28, 2026
**Version**: 1.0 (Final)
