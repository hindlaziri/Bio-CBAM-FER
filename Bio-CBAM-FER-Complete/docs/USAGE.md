# Usage Guide for Bio-CBAM

## Quick Start

### 1. Training

Train Bio-CBAM on FER-2013 dataset:

```bash
python train.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 0.001 \
    --backbone resnet50 \
    --device cuda \
    --checkpoint-dir ./checkpoints
```

### 2. Evaluation

Evaluate trained model:

```bash
python eval.py \
    --model-path ./checkpoints/best_model.pth \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --batch-size 32 \
    --device cuda \
    --output-dir ./results
```

### 3. Reproducibility Testing

Test model reproducibility across multiple runs:

```bash
python reproducibility_test.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --num-runs 5 \
    --device cuda \
    --output-dir ./reproducibility
```

## Detailed Usage

### Training Arguments

```
--dataset {fer2013, ckplus, jaffe}
    Dataset to use (default: fer2013)

--data-dir PATH
    Path to dataset directory (default: ./data)

--num-classes {4, 7}
    Number of emotion classes (default: 7)
    - 4-class: angry, happy, neutral, sad
    - 7-class: angry, disgust, fear, happy, neutral, sad, surprise

--batch-size SIZE
    Batch size for training (default: 32)
    - Reduce if GPU out of memory
    - Increase for faster training (if memory allows)

--num-epochs EPOCHS
    Number of training epochs (default: 100)

--learning-rate LR
    Initial learning rate (default: 0.001)
    - Higher LR: faster training but may diverge
    - Lower LR: slower training but more stable

--backbone {resnet18, resnet50}
    Backbone architecture (default: resnet50)
    - resnet18: smaller, faster
    - resnet50: larger, better accuracy

--use-fmri-prior
    Use fMRI-guided spatial priors (default: True)

--device {cuda, cpu}
    Device to use (default: cuda)

--checkpoint-dir PATH
    Directory to save checkpoints (default: ./checkpoints)

--seed SEED
    Random seed for reproducibility (default: 42)
```

### Evaluation Arguments

```
--model-path PATH
    Path to trained model checkpoint (required)

--dataset {fer2013, ckplus, jaffe}
    Dataset to evaluate on (default: fer2013)

--data-dir PATH
    Path to dataset directory (default: ./data)

--num-classes {4, 7}
    Number of emotion classes (default: 7)

--batch-size SIZE
    Batch size for evaluation (default: 32)

--backbone {resnet18, resnet50}
    Backbone architecture (default: resnet50)

--device {cuda, cpu}
    Device to use (default: cuda)

--output-dir PATH
    Directory to save results (default: ./results)
```

### Reproducibility Testing Arguments

```
--dataset {fer2013, ckplus, jaffe}
    Dataset to use (default: fer2013)

--data-dir PATH
    Path to dataset directory (default: ./data)

--num-classes {4, 7}
    Number of emotion classes (default: 7)

--batch-size SIZE
    Batch size for evaluation (default: 32)

--num-runs RUNS
    Number of reproducibility test runs (default: 5)

--backbone {resnet18, resnet50}
    Backbone architecture (default: resnet50)

--device {cuda, cpu}
    Device to use (default: cuda)

--output-dir PATH
    Directory to save results (default: ./reproducibility)
```

## Training Examples

### Example 1: Train on FER-2013 (4-class)

```bash
python train.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 4 \
    --batch-size 64 \
    --num-epochs 150 \
    --learning-rate 0.0005 \
    --backbone resnet50 \
    --seed 42
```

### Example 2: Train on CK+ Dataset

```bash
python train.py \
    --dataset ckplus \
    --data-dir ./data/ckplus \
    --num-classes 7 \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 0.001 \
    --backbone resnet50
```

### Example 3: Train with ResNet18 (Lightweight)

```bash
python train.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --batch-size 64 \
    --num-epochs 100 \
    --learning-rate 0.001 \
    --backbone resnet18
```

### Example 4: Train on CPU

```bash
python train.py \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --batch-size 16 \
    --num-epochs 50 \
    --device cpu
```

## Evaluation Examples

### Example 1: Evaluate Best Model

```bash
python eval.py \
    --model-path ./checkpoints/best_model.pth \
    --dataset fer2013 \
    --data-dir ./data/fer2013 \
    --num-classes 7 \
    --output-dir ./results/best_model
```

### Example 2: Evaluate on Different Dataset

```bash
python eval.py \
    --model-path ./checkpoints/best_model.pth \
    --dataset ckplus \
    --data-dir ./data/ckplus \
    --num-classes 7 \
    --output-dir ./results/ckplus_eval
```

## Output Files

### Training Outputs

- `checkpoints/best_model.pth`: Best model weights
- `checkpoints/final_model.pth`: Final model weights
- `checkpoints/training_history.json`: Training history (loss, accuracy)

### Evaluation Outputs

- `results/metrics.json`: Evaluation metrics (accuracy, precision, recall, F1)
- `results/confusion_matrix.png`: Confusion matrix visualization
- `results/metrics_plot.png`: Metrics visualization

### Reproducibility Outputs

- `reproducibility/reproducibility_results.json`: Results from multiple runs
- `reproducibility/reproducibility_plot.png`: Accuracy and loss across runs

## Performance Tips

### For Better Accuracy

1. Increase batch size (if GPU memory allows): `--batch-size 64`
2. Train longer: `--num-epochs 200`
3. Use larger backbone: `--backbone resnet50`
4. Lower learning rate: `--learning-rate 0.0001`
5. Use data augmentation (enabled by default)

### For Faster Training

1. Reduce batch size: `--batch-size 16`
2. Use smaller backbone: `--backbone resnet18`
3. Reduce epochs: `--num-epochs 50`
4. Use higher learning rate: `--learning-rate 0.01`

### For Memory Efficiency

1. Reduce batch size: `--batch-size 8`
2. Use CPU: `--device cpu`
3. Use smaller backbone: `--backbone resnet18`

## Monitoring Training

Training progress is displayed in real-time:

```
Epoch 1/100
Training: 100%|██████████| 1000/1000 [02:30<00:00, 6.67it/s]
Train Loss: 2.1234, Train Acc: 25.45%
Validation: 100%|██████████| 250/250 [00:30<00:00, 8.33it/s]
Val Loss: 1.9876, Val Acc: 35.67%
Saved best model to ./checkpoints/best_model.pth
```

## Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
python train.py --batch-size 16 ...

# Or use CPU
python train.py --device cpu ...
```

### Model Not Improving

```bash
# Try lower learning rate
python train.py --learning-rate 0.0001 ...

# Or train longer
python train.py --num-epochs 200 ...
```

### Dataset Not Found

```bash
# Check dataset path
ls -la ./data/fer2013/

# Verify CSV file exists
ls -la ./data/fer2013/fer2013.csv
```

## Next Steps

- See [README.md](../README.md) for project overview
- See [INSTALL.md](INSTALL.md) for installation instructions
- Check the paper for methodology details
