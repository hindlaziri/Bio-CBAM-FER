"""Evaluate frozen Bio-CBAM checkpoints with deliberately misspecified priors.

The trained network weights and scalar gates are left unchanged. Only the fixed
prior buffer is replaced at inference, using deterministic transformations that
preserve either its value distribution or its gross spatial structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn

from dataset_scripts import create_dataloaders
from utils.checkpoint import load_checkpoint, model_from_checkpoint
from utils.evaluation import evaluate_model
from utils.runtime import environment_report, resolve_device, save_json, set_global_seed


def _discover_checkpoints(root: Path) -> List[Path]:
    checkpoints: List[Path] = []
    for variant in ("face_equal_weight", "fmri_weighted"):
        checkpoints.extend(sorted((root / "fer4" / variant).glob("seed_*/best_checkpoint.pt")))
    if len(checkpoints) != 6:
        raise ValueError(f"Expected six FER-2013 four-class prior checkpoints, found {len(checkpoints)}")
    return [path.resolve() for path in checkpoints]


def _variant_seed(path: Path) -> tuple[str, int]:
    return path.parent.parent.name, int(path.parent.name.removeprefix("seed_"))


def _zero_shift(prior: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    output = torch.zeros_like(prior)
    height, width = prior.shape[-2:]
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    target_y0, target_y1 = max(0, dy), min(height, height + dy)
    target_x0, target_x1 = max(0, dx), min(width, width + dx)
    output[..., target_y0:target_y1, target_x0:target_x1] = prior[..., source_y0:source_y1, source_x0:source_x1]
    return output


def _permute_blocks(prior: torch.Tensor, blocks: int, seed: int) -> torch.Tensor:
    height, width = prior.shape[-2:]
    if height % blocks or width % blocks:
        raise ValueError("Prior dimensions must be divisible by the requested block count")
    block_h, block_w = height // blocks, width // blocks
    tiles = []
    for row in range(blocks):
        for column in range(blocks):
            tiles.append(prior[..., row * block_h:(row + 1) * block_h, column * block_w:(column + 1) * block_w])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(tiles), generator=generator).tolist()
    rows = []
    for row in range(blocks):
        rows.append(torch.cat([tiles[order[row * blocks + column]] for column in range(blocks)], dim=-1))
    return torch.cat(rows, dim=-2)


def _conditions(original: torch.Tensor, shift_pixels: int, block_count: int, seed: int) -> Dict[str, torch.Tensor]:
    return {
        "original": original.clone(),
        "zero": torch.zeros_like(original),
        "horizontal_flip": torch.flip(original, dims=(-1,)),
        "rotation_180": torch.flip(original, dims=(-2, -1)),
        "intensity_complement": original.amax() + original.amin() - original,
        "shift_down_right": _zero_shift(original, shift_pixels, shift_pixels),
        "block_permutation": _permute_blocks(original.cpu(), block_count, seed).to(original.device),
    }


def _replace_prior(model: torch.nn.Module, prior: torch.Tensor) -> None:
    mixer = getattr(model, "prior_mixer", None)
    if mixer is None or not hasattr(mixer, "priors"):
        raise ValueError("Selected checkpoint does not contain a prior mixer")
    buffer = mixer.priors
    candidate = prior.to(device=buffer.device, dtype=buffer.dtype)
    if candidate.shape != buffer.shape:
        raise ValueError(f"Transformed prior shape {candidate.shape} differs from buffer shape {buffer.shape}")
    buffer.copy_(candidate)


def _aggregate(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    originals: Dict[tuple[str, int], float] = {}
    for run in runs:
        key = (str(run["variant"]), int(run["seed"]))
        if run["condition"] == "original":
            originals[key] = float(run["metrics"]["accuracy"])
    for run in runs:
        grouped[(str(run["variant"]), str(run["condition"]))].append(run)
    output: Dict[str, object] = defaultdict(dict)
    for (variant, condition), entries in sorted(grouped.items()):
        accuracy = [float(entry["metrics"]["accuracy"]) for entry in entries]
        balanced = [float(entry["metrics"]["balanced_accuracy"]) for entry in entries]
        macro_f1 = [float(entry["metrics"]["f1_macro"]) for entry in entries]
        changes = [
            100.0 * (float(entry["metrics"]["accuracy"]) - originals[(variant, int(entry["seed"]))])
            for entry in entries
        ]
        output[variant][condition] = {
            "n_seeds": len(entries),
            "accuracy_mean": float(statistics.mean(accuracy)),
            "accuracy_sd": float(statistics.stdev(accuracy)) if len(accuracy) > 1 else 0.0,
            "balanced_accuracy_mean": float(statistics.mean(balanced)),
            "macro_f1_mean": float(statistics.mean(macro_f1)),
            "accuracy_change_percentage_points_mean": float(statistics.mean(changes)),
            "accuracy_change_percentage_points_sd": float(statistics.stdev(changes)) if len(changes) > 1 else 0.0,
            "accuracy_change_seed_values": changes,
        }
    return dict(output)


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_global_seed(args.evaluation_seed, deterministic=True)
    loaders = create_dataloaders(
        dataset_name="fer2013",
        data_path=args.fer_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=4,
        four_class_names=("angry", "happy", "sad", "neutral"),
        image_size=args.image_size,
        seed=args.evaluation_seed,
        use_ssim_filtering=False,
        return_metadata=True,
    )

    runs: List[Dict[str, object]] = []
    for checkpoint_path in _discover_checkpoints(checkpoint_root):
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        model, _ = model_from_checkpoint(checkpoint)
        model.to(device).eval()
        original = model.prior_mixer.priors.detach().clone()
        prior_hash = hashlib.sha256(original.cpu().numpy().tobytes()).hexdigest()
        variant, seed = _variant_seed(checkpoint_path)
        for condition, transformed in _conditions(original, args.shift_pixels, args.block_count, args.permutation_seed).items():
            _replace_prior(model, transformed)
            metrics, outputs = evaluate_model(
                model,
                loaders["test"],
                device,
                checkpoint["class_names"],
                nn.CrossEntropyLoss(),
            )
            run_dir = destination / variant / f"seed_{seed}" / condition
            save_json(metrics, run_dir / "metrics.json")
            save_json(outputs, run_dir / "predictions.json")
            save_json(
                {
                    "checkpoint": str(checkpoint_path),
                    "variant": variant,
                    "seed": seed,
                    "condition": condition,
                    "original_prior_sha256": prior_hash,
                    "weights_changed": False,
                    "adaptation": "none",
                },
                run_dir / "provenance.json",
            )
            runs.append({"variant": variant, "seed": seed, "condition": condition, "metrics": metrics})
        _replace_prior(model, original)

    payload = {
        "design": {
            "dataset": "FER-2013 strict four-class PrivateTest",
            "checkpoint_count": 6,
            "conditions": ["original", "zero", "horizontal_flip", "rotation_180", "intensity_complement", "shift_down_right", "block_permutation"],
            "shift_pixels": args.shift_pixels,
            "block_count_per_axis": args.block_count,
            "permutation_seed": args.permutation_seed,
            "weights_changed": False,
            "selection_on_test": False,
            "environment": environment_report(Path(__file__).resolve().parents[1]),
        },
        "aggregate": _aggregate(runs),
        "runs": runs,
    }
    save_json(payload, destination / "prior_sensitivity_summary.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--fer-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shift-pixels", type=int, default=16)
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--permutation-seed", type=int, default=20260904)
    parser.add_argument("--evaluation-seed", type=int, default=20260902)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    payload = evaluate(parse_args())
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
