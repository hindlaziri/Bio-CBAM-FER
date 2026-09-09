"""Export Bio-CBAM attention maps and optionally compare them with references."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_scripts import create_dataloaders
from utils.checkpoint import load_checkpoint, model_from_checkpoint
from utils.runtime import resolve_device, save_json, set_global_seed


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)


def normalize_distribution(array: np.ndarray) -> np.ndarray:
    positive = np.clip(array.astype(np.float64), 0, None)
    return positive / max(float(positive.sum()), 1e-12)


def attention_metrics(prediction: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    prediction = cv2.resize(prediction.astype(np.float32), (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    prediction = (prediction - prediction.min()) / max(float(prediction.max() - prediction.min()), 1e-12)
    reference = reference.astype(np.float32)
    reference = (reference - reference.min()) / max(float(reference.max() - reference.min()), 1e-12)
    pred_dist, ref_dist = normalize_distribution(prediction), normalize_distribution(reference)
    centered = (prediction - prediction.mean()) / max(float(prediction.std()), 1e-12)
    fixation = reference > reference.mean()
    nss = float(centered[fixation].mean()) if fixation.any() else float("nan")
    correlation = float(np.corrcoef(prediction.ravel(), reference.ravel())[0, 1]) if prediction.std() > 0 and reference.std() > 0 else float("nan")
    kl = float(np.sum(ref_dist * np.log((ref_dist + 1e-12) / (pred_dist + 1e-12))))
    similarity = float(np.minimum(pred_dist, ref_dist).sum())
    binary = fixation.astype(np.uint8).ravel()
    auc = float(roc_auc_score(binary, prediction.ravel())) if np.unique(binary).size == 2 else float("nan")
    return {"nss": nss, "pearson": correlation, "kl_divergence": kl, "similarity": similarity, "auc": auc}


def load_reference_map(path: str) -> np.ndarray:
    resolved = Path(path)
    if resolved.suffix.lower() == ".npy":
        result = np.load(resolved, allow_pickle=False)
    else:
        result = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE)
        if result is None:
            raise FileNotFoundError(resolved)
    if result.ndim != 2:
        raise ValueError(f"Reference map must be 2D: {resolved}")
    return result.astype(np.float32)


def save_overlay(image_tensor: torch.Tensor, attention: np.ndarray, path: Path, title: str) -> None:
    image = image_tensor.detach().cpu().numpy() * STD + MEAN
    image = np.clip(np.transpose(image, (1, 2, 0)), 0, 1)
    heatmap = cv2.resize(attention.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(heatmap, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Attention")
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap="magma", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("fer2013", "ckplus", "jaffe"))
    parser.add_argument("--data-path")
    parser.add_argument("--reference-manifest", help="CSV with identifier and map_path")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.output_dir).expanduser().resolve()
    maps_dir, overlays_dir = destination / "maps", destination / "overlays"
    maps_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
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

    references: Dict[str, str] = {}
    if args.reference_manifest:
        frame = pd.read_csv(args.reference_manifest)
        if not {"identifier", "map_path"}.issubset(frame.columns):
            raise ValueError("Reference manifest requires identifier and map_path")
        base = Path(args.reference_manifest).resolve().parent
        for _, row in frame.iterrows():
            path = Path(str(row["map_path"]))
            references[str(row["identifier"])] = str(path if path.is_absolute() else base / path)

    rows: List[Dict[str, object]] = []
    count = 0
    with torch.inference_mode():
        for images, labels, metadata in loaders["test"]:
            images_device = images.to(device)
            _, diagnostics = model(images_device)
            if len(diagnostics.get("stages", [])) < args.stage:
                raise ValueError(
                    f"Checkpoint architecture '{config.architecture}' does not expose attention stage {args.stage}"
                )
            attention = diagnostics["stages"][args.stage - 1]["spatial_attention"]
            attention = F.interpolate(attention, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
            for index in range(images.shape[0]):
                identifier = str(metadata["identifier"][index])
                safe_name = identifier.replace(":", "_").replace("/", "_")
                array = attention[index].astype(np.float32)
                np.save(maps_dir / f"{safe_name}.npy", array, allow_pickle=False)
                save_overlay(images[index], array, overlays_dir / f"{safe_name}.png", f"{identifier}; true={metadata['label_name'][index]}")
                row: Dict[str, object] = {"identifier": identifier, "true_label": int(labels[index]), "map_path": str(maps_dir / f"{safe_name}.npy")}
                if identifier in references:
                    row["reference_metrics"] = attention_metrics(array, load_reference_map(references[identifier]))
                rows.append(row)
                count += 1
                if count >= args.max_images:
                    break
            if count >= args.max_images:
                break

    aggregate: Dict[str, float] = {}
    evaluated = [row["reference_metrics"] for row in rows if "reference_metrics" in row]
    if evaluated:
        for key in evaluated[0]:
            values = np.asarray([item[key] for item in evaluated], dtype=float)
            aggregate[key] = float(np.nanmean(values))
    report = {"stage": args.stage, "images_exported": len(rows), "references_evaluated": len(evaluated), "aggregate_reference_metrics": aggregate, "records": rows}
    save_json(report, destination / "attention_report.json")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
