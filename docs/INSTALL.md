# Installation Guide for Bio-CBAM

## System Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (for GPU support)
- **cuDNN**: 8.0 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/hindlaziri/Bio-CBAM-FER.git
cd Bio-CBAM-FER
```

### 2. Create Virtual Environment

#### Using Conda (Recommended)

```bash
conda create -n bio-cbam python=3.9
conda activate bio-cbam
```

#### Using venv

```bash
python3 -m venv bio-cbam
source bio-cbam/bin/activate  # On Windows: bio-cbam\Scripts\activate
```

### 3. Install PyTorch

#### For GPU (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU

```bash
pip install torch torchvision torchaudio
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### Manual Installation (if requirements.txt not available)

```bash
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install opencv-python==4.8.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install tqdm==4.66.1
pip install Pillow==10.0.0
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Setup

### FER-2013 Dataset

1. Download from: https://www.kaggle.com/datasets/msambare/fer2013
2. Extract to `data/fer2013/`
3. Ensure the CSV file is at `data/fer2013/fer2013.csv`

### CK+ Dataset

1. Download from: https://www.jeffcohn.com/databases/
2. Extract to `data/ckplus/`
3. Organize as:
   ```
   data/ckplus/
   ├── cohn-kanade-images/
   ├── Emotion/
   └── ...
   ```

### JAFFE Dataset

1. Download from: https://zenodo.org/record/3451524
2. Extract to `data/jaffe/`
3. Organize as:
   ```
   data/jaffe/
   ├── tiff/
   └── ...
   ```

## Directory Structure

After installation, your project should look like:

```
Bio-CBAM-FER/
├── models/
│   ├── __init__.py
│   └── bio_cbam.py
├── dataset_scripts/
│   ├── __init__.py
│   └── dataset_loader.py
├── weights/
│   └── (pre-trained model weights)
├── data/
│   ├── fer2013/
│   ├── ckplus/
│   └── jaffe/
├── checkpoints/
├── results/
├── docs/
│   ├── INSTALL.md
│   └── USAGE.md
├── train.py
├── eval.py
├── reproducibility_test.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Reduce batch size: `--batch-size 16`
2. Use CPU instead: `--device cpu`
3. Clear GPU cache: `python -c "import torch; torch.cuda.empty_cache()"`

### Import Errors

If you encounter import errors:

1. Ensure virtual environment is activated
2. Reinstall PyTorch: `pip install --upgrade torch`
3. Check Python version: `python --version`

### Dataset Loading Issues

1. Verify dataset paths are correct
2. Check file permissions: `chmod 755 data/`
3. Ensure CSV files are in correct format

## GPU Acceleration

To verify GPU acceleration:

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## Next Steps

After installation, proceed to [USAGE.md](USAGE.md) for training and evaluation instructions.
