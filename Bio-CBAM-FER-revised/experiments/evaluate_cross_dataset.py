"""Evaluate FER-2013 checkpoints on all JAFFE images without adaptation.

The script uses a fixed official seven-class label map, does not inspect JAFFE
labels for model selection, and reports both image-level and subject-level
metrics. The same 213 JAFFE images are evaluated by every FER-trained checkpoint.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_scripts.dataset_loader import FER2013_LABELS, ImageRecord, RecordDataset, get_transforms
from utils.checkpoint import load_checkpoint, model_from_checkpoint
from utils.evaluation import compute_metrics, evaluate_model
from utils.runtime import environment_report, resolve_device, save_json, set_global_seed

FER_CLASS_NAMES = tuple(FER2013_LABELS[index] for index in sorted(FER2013_LABELS))


def _build_external_loader(
    manifest_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, Dict[str, str]]:
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"id", "path", "label", "subject_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"JAFFE manifest is missing columns: {sorted(missing)}")
    frame = frame.drop_duplicates(subset=["id"]).sort_values("id").reset_index(drop=True)
    labels = set(frame["label"].astype(str).str.lower())
    if labels != set(FER_CLASS_NAMES):
        raise ValueError(
            "Cross-dataset evaluation requires the same seven official labels; "
            f"observed={sorted(labels)}, expected={sorted(FER_CLASS_NAMES)}"
        )
    label_map = {name: index for index, name in enumerate(FER_CLASS_NAMES)}
    records: List[ImageRecord] = []
    subjects: Dict[str, str] = {}
    for row in frame.itertuples(index=False):
        identifier = str(row.id)
        label_name = str(row.label).lower()
        image_path = Path(str(row.path)).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        subject = str(row.subject_id)
        records.append(
            ImageRecord(
                identifier=identifier,
                label=label_map[label_name],
                label_name=label_name,
                split="external_test",
                subject_id=subject,
                path=str(image_path),
            )
        )
        subjects[identifier] = subject
    dataset = RecordDataset(records, transform=get_transforms(image_size)["eval"], return_metadata=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, subjects


def _subject_metrics(outputs: Mapping[str, object], subjects: Mapping[str, str]) -> Dict[str, object]:
    identifiers = [str(item) for item in outputs["identifiers"]]
    targets = np.asarray(outputs["targets"], dtype=np.int64)
    predictions = np.asarray(outputs["predictions"], dtype=np.int64)
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, identifier in enumerate(identifiers):
        grouped[subjects[identifier]].append(index)
    per_subject: Dict[str, Dict[str, float]] = {}
    for subject, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        metrics = compute_metrics(targets[idx], predictions[idx], FER_CLASS_NAMES)
        per_subject[subject] = {
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "f1_macro": float(metrics["f1_macro"]),
            "support": int(metrics["support"]),
        }
    summary: Dict[str, object] = {"per_subject": per_subject, "subject_count": len(per_subject)}
    for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
        values = [entry[metric] for entry in per_subject.values()]
        summary[f"mean_subject_{metric}"] = float(statistics.mean(values))
        summary[f"sd_subject_{metric}"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return summary


def _discover_checkpoints(root: Path) -> List[Path]:
    patterns = (
        "fer7/*/seed_*/best_checkpoint.pt",
        "fer7_crossmodal/*/seed_*/best_checkpoint.pt",
    )
    checkpoints = sorted({path.resolve() for pattern in patterns for path in root.glob(pattern)})
    if len(checkpoints) != 15:
        raise ValueError(f"Expected 15 FER-2013 seven-class checkpoints, found {len(checkpoints)}")
    return checkpoints


def _variant_seed(checkpoint_path: Path) -> tuple[str, int]:
    seed = int(checkpoint_path.parent.name.removeprefix("seed_"))
    return checkpoint_path.parent.parent.name, seed


def _aggregate(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["variant"])].append(record)
    result: Dict[str, object] = {}
    for variant, entries in sorted(grouped.items()):
        values: Dict[str, object] = {"n_seeds": len(entries), "seeds": [int(item["seed"]) for item in entries]}
        for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
            observations = [float(item["metrics"][metric]) for item in entries]
            values[f"mean_{metric}"] = float(statistics.mean(observations))
            values[f"sd_{metric}"] = float(statistics.stdev(observations)) if len(observations) > 1 else 0.0
        for metric in ("mean_subject_accuracy", "mean_subject_balanced_accuracy", "mean_subject_f1_macro"):
            observations = [float(item["subject_metrics"][metric]) for item in entries]
            values[f"mean_{metric}"] = float(statistics.mean(observations))
            values[f"sd_{metric}"] = float(statistics.stdev(observations)) if len(observations) > 1 else 0.0
        result[variant] = values
    return result


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    root = Path(args.checkpoint_root).expanduser().resolve()
    manifest = Path(args.jaffe_manifest).expanduser().resolve()
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_global_seed(args.evaluation_seed, deterministic=True)
    checkpoints = _discover_checkpoints(root)
    records: List[Dict[str, object]] = []
    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        class_names = tuple(str(item).lower() for item in checkpoint["class_names"])
        if class_names != FER_CLASS_NAMES:
            raise ValueError(f"Unexpected checkpoint class order in {checkpoint_path}: {class_names}")
        model, _ = model_from_checkpoint(checkpoint)
        model.to(device)
        run = checkpoint.get("run_config", {})
        image_size = int(run.get("image_size", 64))
        loader, subjects = _build_external_loader(manifest, image_size, args.batch_size, args.num_workers)
        metrics, outputs = evaluate_model(model, loader, device, FER_CLASS_NAMES, nn.CrossEntropyLoss())
        subject_metrics = _subject_metrics(outputs, subjects)
        variant, seed = _variant_seed(checkpoint_path)
        run_dir = destination / variant / f"seed_{seed}"
        save_json(metrics, run_dir / "metrics.json")
        save_json(outputs, run_dir / "predictions.json")
        save_json(subject_metrics, run_dir / "subject_metrics.json")
        save_json(
            {
                "training_dataset": "FER-2013 official seven-class Training/PublicTest",
                "external_test_dataset": "JAFFE, all 213 images",
                "adaptation": "none",
                "checkpoint": str(checkpoint_path),
                "variant": variant,
                "seed": seed,
                "class_names": list(FER_CLASS_NAMES),
                "manifest": str(manifest),
                "image_size": image_size,
            },
            run_dir / "provenance.json",
        )
        records.append(
            {
                "variant": variant,
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "metrics": metrics,
                "subject_metrics": subject_metrics,
            }
        )
    payload = {
        "design": {
            "training_dataset": "FER-2013 official seven-class",
            "external_test_dataset": "JAFFE, all unique images",
            "adaptation": "none",
            "selection_on_external_test": False,
            "class_names": list(FER_CLASS_NAMES),
            "checkpoint_count": len(checkpoints),
            "environment": environment_report(Path(__file__).resolve().parents[1]),
        },
        "aggregate": _aggregate(records),
        "runs": records,
    }
    save_json(payload, destination / "cross_dataset_summary.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--jaffe-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=20260902)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
