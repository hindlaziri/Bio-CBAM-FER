"""Build subject-aware manifests for JAFFE and CK+.

The generated CSV files contain path, label, subject_id and split columns and
can be consumed by ManifestDataset. Subject-disjoint folds are produced by the
shared create_subject_folds utility.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .dataset_loader import create_subject_folds


JAFFE_CODES: Dict[str, str] = {
    "AN": "angry",
    "DI": "disgust",
    "FE": "fear",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise",
}

CKPLUS_LABELS: Dict[int, str] = {
    0: "neutral",
    1: "angry",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happy",
    6: "sad",
    7: "surprise",
}


def build_jaffe_manifest(image_dir: Path, output_path: Path) -> int:
    records: List[dict] = []
    pattern = re.compile(r"^(?P<subject>[A-Za-z]+)\.(?P<code>AN|DI|FE|HA|NE|SA|SU)", re.I)
    for path in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}):
        match = pattern.match(path.name)
        if match is None:
            continue
        code = match.group("code").upper()
        records.append(
            {
                "id": path.stem,
                "path": str(path.resolve()),
                "label": JAFFE_CODES[code],
                "subject_id": match.group("subject").upper(),
            }
        )
    if not records:
        raise ValueError("No JAFFE filename matching SUBJECT.EMOTION... was found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return len(records)


def _read_ckplus_emotion(label_dir: Path) -> int:
    files = sorted(label_dir.glob("*_emotion.txt"))
    if not files:
        raise FileNotFoundError(f"No CK+ emotion file in {label_dir}")
    value = float(files[-1].read_text(encoding="utf-8").strip())
    label = int(round(value))
    if label not in CKPLUS_LABELS or label == 0:
        raise ValueError(f"Unsupported CK+ peak label {value} in {files[-1]}")
    return label


def build_ckplus_manifest(
    image_root: Path,
    emotion_root: Path,
    output_path: Path,
    include_neutral: bool = True,
) -> int:
    records: List[dict] = []
    for subject_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        for sequence_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            frames = sorted(
                p for p in sequence_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if not frames:
                continue
            relative = sequence_dir.relative_to(image_root)
            label_dir = emotion_root / relative
            try:
                peak_label = _read_ckplus_emotion(label_dir)
            except FileNotFoundError:
                continue
            subject_id = subject_dir.name
            peak = frames[-1]
            records.append(
                {
                    "id": f"{subject_id}_{sequence_dir.name}_peak",
                    "path": str(peak.resolve()),
                    "label": CKPLUS_LABELS[peak_label],
                    "subject_id": subject_id,
                }
            )
            if include_neutral:
                neutral = frames[0]
                records.append(
                    {
                        "id": f"{subject_id}_{sequence_dir.name}_neutral",
                        "path": str(neutral.resolve()),
                        "label": "neutral",
                        "subject_id": subject_id,
                    }
                )
    if not records:
        raise ValueError("No labeled CK+ sequence was found; verify both roots")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    jaffe = sub.add_parser("jaffe")
    jaffe.add_argument("--images", required=True)
    jaffe.add_argument("--output", required=True)

    ckplus = sub.add_parser("ckplus")
    ckplus.add_argument("--images", required=True)
    ckplus.add_argument("--emotion-labels", required=True)
    ckplus.add_argument("--output", required=True)
    ckplus.add_argument("--exclude-neutral", action="store_true")

    folds = sub.add_parser("folds")
    folds.add_argument("--manifest", required=True)
    folds.add_argument("--output-dir", required=True)
    folds.add_argument("--folds", type=int, default=5)
    folds.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "jaffe":
        count = build_jaffe_manifest(Path(args.images), Path(args.output))
        output = {"manifest": args.output, "records": count}
    elif args.command == "ckplus":
        count = build_ckplus_manifest(
            Path(args.images), Path(args.emotion_labels), Path(args.output),
            include_neutral=not args.exclude_neutral,
        )
        output = {"manifest": args.output, "records": count}
    else:
        paths = create_subject_folds(args.manifest, args.output_dir, args.folds, args.seed)
        output = {"fold_manifests": paths}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
