"""Audit dataset splits and class distributions without training a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

from .dataset_loader import DEFAULT_FOUR_CLASSES, FER2013Dataset, validate_subject_disjointness


def path_inventory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def audit_fer(path: Path, num_classes: int, four_classes: tuple[str, ...]) -> Dict[str, object]:
    classes = None if num_classes == 7 else four_classes
    report: Dict[str, object] = {
        "dataset": "fer2013",
        "source": str(path),
        "source_type": "directory" if path.is_dir() else "csv",
        "subject_independent": False,
        "note": "FER-2013 provides no subject identifiers; official Usage splits are used.",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = FER2013Dataset(str(path), split=split, class_names=classes)
        counts = Counter(record.label_name for record in dataset.records)
        report["splits"][split] = {"count": len(dataset), "class_counts": dict(sorted(counts.items()))}
    if path.is_dir():
        report["inventory_sha256"] = path_inventory_digest(path)
    else:
        report["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


def audit_manifest(path: Path) -> Dict[str, object]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"path", "label", "subject_id", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(frame.columns))}")
    validate_subject_disjointness(frame)
    splits: Dict[str, object] = {}
    for split, subset in frame.groupby(frame["split"].astype(str).str.lower()):
        splits[str(split)] = {
            "count": int(len(subset)),
            "subjects": int(subset["subject_id"].astype(str).nunique()),
            "class_counts": {str(key): int(value) for key, value in subset["label"].astype(str).str.lower().value_counts().sort_index().items()},
        }
    return {
        "dataset": "subject_manifest",
        "source": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "subject_independent": True,
        "splits": splits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fer2013", "manifest"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-classes", type=int, choices=(4, 7), default=7)
    parser.add_argument("--four-classes", default=",".join(DEFAULT_FOUR_CLASSES))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if args.dataset == "fer2013":
        four_classes = tuple(item.strip().lower() for item in args.four_classes.split(",") if item.strip())
        report = audit_fer(path, args.num_classes, four_classes)
    else:
        report = audit_manifest(path)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
