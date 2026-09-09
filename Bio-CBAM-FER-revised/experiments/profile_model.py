"""Profile Bio-CBAM complexity and runtime under a documented environment."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import BioCBAMConfig, create_model
from utils.checkpoint import load_checkpoint, model_from_checkpoint
from utils.runtime import environment_report, resolve_device, save_json


def build_model(checkpoint_path: Optional[str], backbone: str, num_classes: int, architecture: str, device: torch.device) -> tuple[torch.nn.Module, Optional[dict]]:
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        model, _ = model_from_checkpoint(checkpoint)
        return model.to(device).eval(), checkpoint
    config = BioCBAMConfig(architecture=architecture, num_classes=num_classes, backbone=backbone, pretrained=False)
    return create_model(config).to(device).eval(), None


def try_flops(model: torch.nn.Module, example: torch.Tensor) -> Optional[float]:
    try:
        from thop import profile
    except ImportError:
        return None
    with torch.inference_mode():
        flops, _ = profile(model, inputs=(example,), verbose=False)
    return float(flops)


def profile_runtime(model: torch.nn.Module, example: torch.Tensor, warmup: int, repeats: int) -> Dict[str, float]:
    device = example.device
    with torch.inference_mode():
        for _ in range(warmup):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(example)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
    mean_seconds = statistics.fmean(durations)
    sorted_durations = sorted(durations)
    p50 = sorted_durations[int(0.50 * (len(sorted_durations) - 1))]
    p95 = sorted_durations[int(0.95 * (len(sorted_durations) - 1))]
    result = {
        "mean_batch_seconds": mean_seconds,
        "p50_batch_seconds": p50,
        "p95_batch_seconds": p95,
        "images_per_second": example.shape[0] / mean_seconds,
    }
    if device.type == "cuda":
        result["peak_cuda_memory_bytes"] = float(torch.cuda.max_memory_allocated(device))
    return result


def convergence_from_checkpoint(checkpoint: Optional[dict]) -> Optional[Dict[str, float | int]]:
    if checkpoint is None or not checkpoint.get("history"):
        return None
    history = checkpoint["history"]
    monitored_name = checkpoint.get("run_config", {}).get("monitor", "f1_macro")
    values = [float(item["validation"][monitored_name]) for item in history]
    best_index = int(max(range(len(values)), key=values.__getitem__))
    seconds = [float(item["train"]["seconds"]) for item in history]
    return {
        "epochs_recorded": len(history),
        "best_epoch_zero_based": best_index,
        "best_validation_metric": values[best_index],
        "monitor": monitored_name,
        "mean_training_seconds_per_epoch": statistics.fmean(seconds),
        "cumulative_training_seconds_to_best_epoch": float(sum(seconds[: best_index + 1])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--architecture", choices=("resnet", "cbam", "biocbam"), default="cbam")
    parser.add_argument("--backbone", choices=("resnet18", "resnet50"), default="resnet50")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = build_model(args.checkpoint, args.backbone, args.num_classes, args.architecture, device)
    example = torch.zeros((args.batch_size, 3, args.height, args.width), device=device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report: Dict[str, object] = {
        "model_config": model.config.__dict__,
        "input_shape": list(example.shape),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "flops_per_batch": try_flops(model, example),
        "runtime": profile_runtime(model, example, args.warmup, args.repeats),
        "convergence": convergence_from_checkpoint(checkpoint),
        "environment": environment_report(Path(__file__).resolve().parents[1]),
        "measurement_protocol": {"warmup": args.warmup, "repeats": args.repeats, "device": str(device)},
    }
    save_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
