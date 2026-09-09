"""Quantify spatial-attention faithfulness and stability on a frozen FER test subset.

Metrics are computed from stage-one spatial attention because the 64x64 input
produces only a 2x2 map at stage four. The fixed subset contains the same number
of official test examples per class for every checkpoint. No metric is used for
training or checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset_scripts.dataset_loader import DEFAULT_FOUR_CLASSES, FER2013Dataset, get_transforms
from utils.checkpoint import load_checkpoint, model_from_checkpoint
from utils.runtime import environment_report, resolve_device, save_json, set_global_seed

FRACTIONS = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def _discover_checkpoints(root: Path) -> List[Path]:
    checkpoints: List[Path] = []
    for variant in ("cbam", "face_equal_weight", "fmri_weighted"):
        checkpoints.extend(sorted((root / "fer4" / variant).glob("seed_*/best_checkpoint.pt")))
    checkpoints.extend(
        sorted((root / "fer4_random_control" / "random_matched").glob("seed_*/best_checkpoint.pt"))
    )
    if len(checkpoints) != 12:
        raise ValueError(f"Expected twelve FER-2013 four-class checkpoints, found {len(checkpoints)}")
    return [path.resolve() for path in checkpoints]


def _variant_seed(path: Path) -> tuple[str, int]:
    return path.parent.parent.name, int(path.parent.name.removeprefix("seed_"))


def _select_balanced_subset(dataset: FER2013Dataset, per_class: int, seed: int) -> List[int]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[int(record.label)].append(index)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for label in range(len(DEFAULT_FOUR_CLASSES)):
        candidates = np.asarray(grouped[label], dtype=np.int64)
        if candidates.size < per_class:
            raise ValueError(f"Class {label} has only {candidates.size} test images")
        chosen = rng.choice(candidates, size=per_class, replace=False)
        selected.extend(int(item) for item in chosen)
    return sorted(selected)


def _stage_attention(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits, diagnostics = model(images)
    stages = diagnostics.get("stages", [])
    if not stages or "spatial_attention" not in stages[0]:
        raise ValueError("The selected model does not expose stage-one spatial attention")
    attention = stages[0]["spatial_attention"]
    attention = F.interpolate(attention, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return logits, attention


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(1)
    minimum = flat.amin(dim=1, keepdim=True)
    maximum = flat.amax(dim=1, keepdim=True)
    normalized = (flat - minimum) / (maximum - minimum).clamp_min(1e-8)
    return normalized.reshape_as(values)


def _pearson_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = a.flatten(1)
    y = b.flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).sum(dim=1)
    denominator = torch.sqrt((x.square().sum(dim=1) * y.square().sum(dim=1)).clamp_min(1e-12))
    return numerator / denominator


def _ranking_masks(attention: torch.Tensor, grid_size: int) -> List[torch.Tensor]:
    coarse = F.adaptive_avg_pool2d(attention, (grid_size, grid_size)).flatten(1)
    order = torch.argsort(coarse, dim=1, descending=True)
    masks: List[torch.Tensor] = []
    cell_count = grid_size * grid_size
    for fraction in FRACTIONS:
        keep = int(round(float(fraction) * cell_count))
        mask = torch.zeros_like(coarse)
        if keep > 0:
            mask.scatter_(1, order[:, :keep], 1.0)
        mask = mask.reshape(-1, 1, grid_size, grid_size)
        masks.append(F.interpolate(mask, size=attention.shape[-2:], mode="nearest"))
    return masks


def _true_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)


def _curves(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    masks: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    insertion: List[torch.Tensor] = []
    deletion: List[torch.Tensor] = []
    baseline = torch.zeros_like(images)
    for mask in masks:
        insertion_logits, _ = model(baseline + images * mask)
        deletion_logits, _ = model(images * (1.0 - mask))
        insertion.append(_true_probability(insertion_logits, labels))
        deletion.append(_true_probability(deletion_logits, labels))
    return torch.stack(insertion, dim=1), torch.stack(deletion, dim=1)


def _auc(curves: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(FRACTIONS, device=curves.device, dtype=curves.dtype)
    return torch.trapz(curves, x=x, dim=1)


def _summary(values: Sequence[float]) -> Dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": float(statistics.mean(data)),
        "sd": float(statistics.stdev(data)) if len(data) > 1 else 0.0,
        "n": len(data),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    region_prior: torch.Tensor,
    device: torch.device,
    grid_size: int,
    noise_std: float,
    perturbation_seed: int,
) -> Dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model, config = model_from_checkpoint(checkpoint)
    model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(perturbation_seed)
    region_prior = region_prior.to(device=device, dtype=torch.float32)
    region_prior = _normalize_map(region_prior[None, None])
    threshold = torch.quantile(region_prior.flatten(), 0.75)
    region_mask = (region_prior >= threshold).float()

    collected: Dict[str, List[float]] = defaultdict(list)
    sample_records: List[Dict[str, object]] = []
    with torch.inference_mode():
        for images, labels, metadata in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, attention = _stage_attention(model, images)
            attention = _normalize_map(attention)
            masks = _ranking_masks(attention, grid_size)
            insertion, deletion = _curves(model, images, labels, masks)
            insertion_auc = _auc(insertion)
            deletion_auc = _auc(deletion)
            faithfulness_gap = insertion_auc - deletion_auc

            expanded_region = region_prior.expand(images.shape[0], -1, -1, -1)
            expanded_mask = region_mask.expand(images.shape[0], -1, -1, -1)
            region_correlation = _pearson_batch(attention, expanded_region)
            region_mass = (attention * expanded_mask).flatten(1).sum(dim=1) / attention.flatten(1).sum(dim=1).clamp_min(1e-8)

            noise = torch.randn(images.shape, generator=generator, device=device, dtype=images.dtype) * noise_std
            _, noisy_attention = _stage_attention(model, images + noise)
            noisy_attention = _normalize_map(noisy_attention)
            noise_stability = _pearson_batch(attention, noisy_attention)

            _, flipped_attention = _stage_attention(model, torch.flip(images, dims=(-1,)))
            flipped_attention = torch.flip(_normalize_map(flipped_attention), dims=(-1,))
            flip_stability = _pearson_batch(attention, flipped_attention)

            confidence = torch.softmax(logits, dim=1).amax(dim=1)
            correct = torch.argmax(logits, dim=1).eq(labels)
            batch_values = {
                "insertion_auc": insertion_auc,
                "deletion_auc": deletion_auc,
                "faithfulness_gap": faithfulness_gap,
                "region_correlation": region_correlation,
                "top_quartile_region_mass": region_mass,
                "noise_stability": noise_stability,
                "flip_stability": flip_stability,
                "confidence": confidence,
                "correct": correct.float(),
            }
            for key, tensor in batch_values.items():
                collected[key].extend(float(item) for item in tensor.detach().cpu())
            identifiers = [str(item) for item in metadata["identifier"]]
            for index, identifier in enumerate(identifiers):
                sample_records.append(
                    {
                        "identifier": identifier,
                        "label": int(labels[index].item()),
                        **{key: float(tensor[index].item()) for key, tensor in batch_values.items()},
                    }
                )

    variant, seed = _variant_seed(checkpoint_path)
    return {
        "variant": variant,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "model_config": config.__dict__,
        "metrics": {key: _summary(values) for key, values in sorted(collected.items())},
        "samples": sample_records,
    }


def _aggregate(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["variant"])].append(run)
    output: Dict[str, object] = {}
    for variant, entries in sorted(grouped.items()):
        variant_summary: Dict[str, object] = {"n_seeds": len(entries)}
        metric_names = entries[0]["metrics"].keys()
        for metric in metric_names:
            seed_means = [float(entry["metrics"][metric]["mean"]) for entry in entries]
            variant_summary[metric] = {
                "mean_across_seeds": float(statistics.mean(seed_means)),
                "sd_across_seeds": float(statistics.stdev(seed_means)) if len(seed_means) > 1 else 0.0,
                "seed_values": seed_means,
            }
        output[variant] = variant_summary
    return output


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    root = Path(args.checkpoint_root).expanduser().resolve()
    data_path = Path(args.fer_data).expanduser().resolve()
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_global_seed(args.subset_seed, deterministic=True)

    dataset = FER2013Dataset(
        str(data_path),
        split="test",
        class_names=DEFAULT_FOUR_CLASSES,
        transform=get_transforms(args.image_size)["eval"],
        return_metadata=True,
    )
    indices = _select_balanced_subset(dataset, args.per_class, args.subset_seed)
    identifiers = [dataset.records[index].identifier for index in indices]
    selection_hash = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
    save_json(
        {
            "subset_seed": args.subset_seed,
            "per_class": args.per_class,
            "count": len(indices),
            "identifiers": identifiers,
            "sha256": selection_hash,
        },
        destination / "selected_subset.json",
    )
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    region_prior = torch.from_numpy(np.load(Path(args.region_prior).expanduser().resolve())).float().squeeze()
    if region_prior.ndim != 2:
        raise ValueError("region prior must be a single two-dimensional map")

    runs: List[Dict[str, object]] = []
    for checkpoint_path in _discover_checkpoints(root):
        variant, seed = _variant_seed(checkpoint_path)
        run = evaluate_checkpoint(
            checkpoint_path,
            loader,
            region_prior,
            device,
            args.grid_size,
            args.noise_std,
            args.perturbation_seed,
        )
        save_json(run, destination / variant / f"seed_{seed}" / "attention_faithfulness.json")
        runs.append(run)
    payload = {
        "design": {
            "dataset": "FER-2013 strict four-class PrivateTest",
            "classes": list(DEFAULT_FOUR_CLASSES),
            "stage": 1,
            "subset_count": len(indices),
            "subset_sha256": selection_hash,
            "fractions": FRACTIONS.tolist(),
            "grid_size": args.grid_size,
            "baseline": "zero after ImageNet normalization (channel means)",
            "noise_std_normalized_units": args.noise_std,
            "region_reference": str(Path(args.region_prior).expanduser().resolve()),
            "environment": environment_report(Path(__file__).resolve().parents[1]),
        },
        "aggregate": _aggregate(runs),
        "runs": runs,
    }
    save_json(payload, destination / "attention_faithfulness_summary.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--fer-data", required=True)
    parser.add_argument("--region-prior", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--subset-seed", type=int, default=20260902)
    parser.add_argument("--perturbation-seed", type=int, default=20260903)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    payload = evaluate(parse_args())
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
