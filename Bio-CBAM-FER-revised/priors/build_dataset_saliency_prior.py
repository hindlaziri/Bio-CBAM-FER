"""Build an average spectral-saliency prior from training images only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from dataset_scripts.dataset_loader import DEFAULT_FOUR_CLASSES, FER2013Dataset, ManifestDataset, RecordDataset
from priors.variants import normalize_map, spectral_residual_saliency


def source_digest(records: Sequence) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.identifier.encode("utf-8"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fer2013", "jaffe", "ckplus"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-classes", type=int, choices=(4, 7), default=7)
    parser.add_argument("--four-classes", default=",".join(DEFAULT_FOUR_CLASSES))
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = None
    if args.dataset == "fer2013" and args.num_classes == 4:
        class_names = tuple(item.strip().lower() for item in args.four_classes.split(",") if item.strip())
    if args.dataset == "fer2013":
        dataset = FER2013Dataset(args.data_path, split="train", class_names=class_names)
    else:
        dataset = ManifestDataset(args.data_path, split="train")

    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    rng = np.random.default_rng(args.seed)
    count = min(args.sample_count, len(dataset.records))
    indices = np.sort(rng.choice(len(dataset.records), size=count, replace=False))
    selected = [dataset.records[int(index)] for index in indices]

    accumulator = np.zeros((args.height, args.width), dtype=np.float64)
    for record in selected:
        image = RecordDataset.read_record(record)
        accumulator += spectral_residual_saliency(image, (args.height, args.width))
    prior = normalize_map(accumulator / float(count)).astype(np.float32)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, prior, allow_pickle=False)
    metadata = {
        "method": "mean_spectral_residual_saliency",
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "split": "train",
        "num_classes": args.num_classes,
        "four_classes": list(class_names) if class_names is not None else None,
        "seed": args.seed,
        "requested_sample_count": args.sample_count,
        "actual_sample_count": count,
        "selected_identifiers_sha256": source_digest(selected),
        "output_shape": [args.height, args.width],
        "output_file": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "uses_labels_for_selection": False,
        "uses_validation_or_test": False,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
