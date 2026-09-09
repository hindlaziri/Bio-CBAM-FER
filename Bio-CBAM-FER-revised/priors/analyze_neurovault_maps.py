"""Quantify public group-level NeuroVault maps without geometric face mapping."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="JSON mapping labels to NIfTI group maps")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def world_center_of_mass(mask: np.ndarray, weights: np.ndarray, affine: np.ndarray) -> list[float] | None:
    if not np.any(mask):
        return None
    coordinates = np.argwhere(mask)
    values = weights[mask].astype(np.float64)
    voxel_center = np.average(coordinates, axis=0, weights=values)
    world = nib.affines.apply_affine(affine, voxel_center)
    return [float(value) for value in world]


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec).expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    threshold = float(spec["threshold"])
    maps: dict[str, dict[str, object]] = {}
    binary_masks: dict[str, np.ndarray] = {}
    reference_shape = None
    reference_affine = None

    for label, definition in spec["maps"].items():
        path = Path(definition["path"]).expanduser().resolve()
        image = nib.load(str(path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite values in {path}")
        if data.min() < 0.0 or data.max() > 1.0:
            raise ValueError(f"Expected a 1-P map in [0,1], got {data.min()}..{data.max()}")
        if reference_shape is None:
            reference_shape = image.shape
            reference_affine = image.affine
        if tuple(image.shape) != tuple(reference_shape) or not np.allclose(image.affine, reference_affine):
            raise ValueError(f"Map {path} does not share the selected-map grid")

        mask = data >= threshold
        excess = np.where(mask, data - threshold, 0.0)
        binary_masks[label] = mask
        maps[label] = {
            "id": int(definition["id"]),
            "contrast": definition["contrast"],
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(image.shape),
            "voxel_volume_mm3": float(np.prod(image.header.get_zooms()[:3])),
            "threshold_1_minus_p": threshold,
            "suprathreshold_voxels": int(mask.sum()),
            "suprathreshold_volume_mm3": float(mask.sum() * np.prod(image.header.get_zooms()[:3])),
            "significance_excess_mass": float(excess.sum()),
            "mean_1_minus_p_suprathreshold": float(data[mask].mean()) if np.any(mask) else None,
            "peak_1_minus_p": float(data.max()),
            "weighted_center_of_mass_mni_mm": world_center_of_mass(mask, data, image.affine),
        }

    pairwise = []
    for first, second in combinations(binary_masks, 2):
        a = binary_masks[first]
        b = binary_masks[second]
        intersection = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        denominator = int(a.sum() + b.sum())
        pairwise.append({
            "first": first,
            "second": second,
            "intersection_voxels": intersection,
            "union_voxels": union,
            "dice": float(2 * intersection / denominator) if denominator else None,
            "jaccard": float(intersection / union) if union else None,
        })

    emotion_labels = [label for label in ("happy", "angry", "sad") if label in maps]
    masses = np.asarray([maps[label]["significance_excess_mass"] for label in emotion_labels], dtype=np.float64)
    if np.allclose(masses.sum(), 0.0):
        weights = np.full(len(emotion_labels), 1.0 / len(emotion_labels))
    else:
        weights = masses / masses.sum()

    report = {
        "analysis": "secondary analysis of public group-level 1-P maps",
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "source_dataset": spec["source_dataset"],
        "source_dataset_doi": spec["source_dataset_doi"],
        "source_collection": spec["source_collection"],
        "map_semantics": spec["map_semantics"],
        "threshold": threshold,
        "maps": maps,
        "pairwise_overlap": pairwise,
        "emotion_evidence_mass_weights": {label: float(weight) for label, weight in zip(emotion_labels, weights)},
        "warning": "Weights summarize normalized suprathreshold 1-P excess mass in brain space. They are descriptive, not psychometric reliability estimates or anatomical coordinates in face space.",
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "emotion_evidence_mass_weights": report["emotion_evidence_mass_weights"]}, indent=2))


if __name__ == "__main__":
    main()
