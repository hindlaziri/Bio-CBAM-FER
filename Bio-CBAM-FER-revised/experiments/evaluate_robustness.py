"""Evaluate a trained checkpoint under controlled image corruptions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_scripts import create_dataloaders
from utils.checkpoint import load_checkpoint, model_from_checkpoint
from utils.evaluation import compute_metrics
from utils.runtime import resolve_device, save_json, set_global_seed


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images * STD.to(images) + MEAN.to(images)).clamp(0, 1)


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - MEAN.to(images)) / STD.to(images)


def gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, *kernel_2d.shape)


def corrupt(images: torch.Tensor, name: str, severity: float, generator: torch.Generator) -> torch.Tensor:
    pixels = denormalize(images)
    if name == "gaussian_noise":
        noise = torch.randn(pixels.shape, generator=generator, device=pixels.device, dtype=pixels.dtype)
        pixels = (pixels + severity * noise).clamp(0, 1)
    elif name == "brightness":
        pixels = (pixels * severity).clamp(0, 1)
    elif name == "blur":
        kernel = gaussian_kernel(severity, pixels.device, pixels.dtype).expand(3, 1, -1, -1)
        padding = kernel.shape[-1] // 2
        pixels = F.conv2d(pixels, kernel, padding=padding, groups=3)
    elif name == "center_occlusion":
        fraction = severity
        if not 0 < fraction < 1:
            raise ValueError("Occlusion severity is a side-length fraction in (0,1)")
        h, w = pixels.shape[-2:]
        oh, ow = max(1, int(h * fraction)), max(1, int(w * fraction))
        top, left = (h - oh) // 2, (w - ow) // 2
        pixels = pixels.clone()
        pixels[:, :, top : top + oh, left : left + ow] = 0.5
    elif name == "translation":
        fraction = severity
        theta = torch.tensor([[1.0, 0.0, fraction], [0.0, 1.0, fraction]], device=pixels.device, dtype=pixels.dtype)
        theta = theta.unsqueeze(0).expand(pixels.shape[0], -1, -1)
        grid = F.affine_grid(theta, pixels.shape, align_corners=False)
        pixels = F.grid_sample(pixels, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    elif name == "clean":
        pass
    else:
        raise ValueError(f"Unknown corruption '{name}'")
    return normalize(pixels)


@torch.inference_mode()
def evaluate_condition(model, loader, device, class_names: Sequence[str], name: str, severity: float, seed: int) -> Dict[str, object]:
    targets: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        perturbed = corrupt(images, name, severity, generator)
        logits, _ = model(perturbed)
        targets.append(labels.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return compute_metrics(np.concatenate(targets), np.concatenate(predictions), class_names)


def default_conditions() -> List[Dict[str, float | str]]:
    return [
        {"name": "clean", "severity": 0.0},
        {"name": "gaussian_noise", "severity": 0.03},
        {"name": "gaussian_noise", "severity": 0.08},
        {"name": "brightness", "severity": 0.6},
        {"name": "brightness", "severity": 1.4},
        {"name": "blur", "severity": 1.0},
        {"name": "blur", "severity": 2.0},
        {"name": "center_occlusion", "severity": 0.2},
        {"name": "center_occlusion", "severity": 0.35},
        {"name": "translation", "severity": 0.05},
        {"name": "translation", "severity": 0.12},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("fer2013", "ckplus", "jaffe"))
    parser.add_argument("--data-path")
    parser.add_argument("--conditions-json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_global_seed(args.seed)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model, config = model_from_checkpoint(checkpoint)
    model.to(device).eval()
    run = checkpoint.get("run_config", {})
    dataset, data_path = args.dataset or run.get("dataset"), args.data_path or run.get("data_path")
    if not dataset or not data_path:
        raise ValueError("Dataset and data path must come from CLI or checkpoint")
    loaders = create_dataloaders(
        dataset_name=dataset,
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=config.num_classes,
        four_class_names=str(run.get("four_classes", "angry,happy,sad,neutral")).split(","),
        image_size=int(run.get("image_size", 224)),
        seed=args.seed,
        return_metadata=True,
    )
    conditions = json.loads(Path(args.conditions_json).read_text(encoding="utf-8")) if args.conditions_json else default_conditions()
    rows = []
    clean_accuracy = None
    for condition in conditions:
        name, severity = str(condition["name"]), float(condition["severity"])
        metrics = evaluate_condition(model, loaders["test"], device, checkpoint["class_names"], name, severity, args.seed)
        if name == "clean":
            clean_accuracy = float(metrics["accuracy"])
        rows.append({"condition": name, "severity": severity, "metrics": metrics})
    if clean_accuracy is None:
        raise ValueError("conditions must include a clean baseline")
    for row in rows:
        row["absolute_accuracy_drop"] = clean_accuracy - float(row["metrics"]["accuracy"])
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "conditions": rows, "seed": args.seed}
    save_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
