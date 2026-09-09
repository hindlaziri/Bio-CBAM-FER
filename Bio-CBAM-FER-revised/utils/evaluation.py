"""Evaluation utilities for trained Bio-CBAM checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .runtime import save_json


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal size")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set")
    labels = list(range(len(class_names))) if class_names is not None else sorted(set(y_true) | set(y_pred))
    target_names = list(class_names) if class_names is not None else [str(label) for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
        "support": int(y_true.size),
    }


def _unpack_batch(batch):
    if len(batch) == 2:
        images, labels = batch
        metadata = None
    elif len(batch) == 3:
        images, labels, metadata = batch
    else:
        raise ValueError("Expected batch=(images, labels[, metadata])")
    return images, labels, metadata


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[Sequence[str]] = None,
    criterion: Optional[nn.Module] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    losses: List[float] = []
    targets: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    identifiers: List[str] = []

    for batch in loader:
        images, labels, metadata = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = model(images)
        losses.append(float(criterion(logits, labels).item()))
        probs = torch.softmax(logits, dim=1)
        targets.append(labels.cpu().numpy())
        predictions.append(torch.argmax(probs, dim=1).cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        if metadata is not None and isinstance(metadata, dict) and "identifier" in metadata:
            identifiers.extend([str(item) for item in metadata["identifier"]])

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    y_prob = np.concatenate(probabilities)
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["loss"] = float(np.mean(losses))
    if hasattr(model, "lambda_values"):
        metrics["lambda_values"] = [float(value) for value in model.lambda_values().detach().cpu()]
    outputs: Dict[str, object] = {
        "targets": y_true.tolist(),
        "predictions": y_pred.tolist(),
        "probabilities": y_prob.tolist(),
    }
    if identifiers:
        outputs["identifiers"] = identifiers
    return metrics, outputs


def save_evaluation(
    metrics: Dict[str, object],
    outputs: Dict[str, object],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_json(metrics, destination / "metrics.json")
    save_json(outputs, destination / "predictions.json")
