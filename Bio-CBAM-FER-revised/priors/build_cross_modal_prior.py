"""Build a cross-modal face prior without geometric brain-to-face warping.

Face-space geometry comes from published diagnostic-tile weights. Public fMRI
maps contribute only normalized condition-level evidence-mass weights. The two
coordinate systems are never treated as anatomically corresponding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EMOTIONS = ("happy", "angry", "sad")
CSV_LABELS = {"happy": "hap", "angry": "ang", "sad": "sad"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum <= 1e-8:
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def load_diagnostic_maps(csv_path: Path, output_size: int) -> dict[str, np.ndarray]:
    table = pd.read_csv(csv_path, index_col=[0, 1])
    table.index.names = ["identity", "emotion"]
    maps: dict[str, np.ndarray] = {}
    for emotion in EMOTIONS:
        code = CSV_LABELS[emotion]
        rows = table.xs(code, level="emotion")
        if set(rows.index.astype(str)) != {"f", "m"}:
            raise ValueError(f"Expected female and male rows for {code}")
        vector = rows.astype(float).mean(axis=0).to_numpy(dtype=np.float32)
        if vector.size != 48:
            raise ValueError(f"Expected 48 published tiles, found {vector.size}")
        # The published notebook creates coordinates with x as the outer loop and y
        # as the inner loop for a 6-column by 8-row grid. Therefore the flat vector
        # must be reshaped as [x,y] and transposed to image [y,x]. The authors use
        # standard min-max scaling over all 48 percent-change values.
        tile_map = vector.reshape(6, 8).T
        tile_map = normalize(tile_map)
        resized = cv2.resize(tile_map, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        maps[emotion] = normalize(resized)
    return maps


def save_preview(components: dict[str, np.ndarray], combined: np.ndarray, path: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for axis, emotion in zip(axes[:3], EMOTIONS):
        axis.imshow(components[emotion], cmap="magma", vmin=0, vmax=1)
        axis.set_title(emotion.capitalize())
        axis.axis("off")
    axes[3].imshow(combined, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("fMRI-weighted")
    axes[3].axis("off")
    figure.suptitle("Functional face priors; no anatomical brain-to-face mapping")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-weights-csv", required=True)
    parser.add_argument("--fmri-analysis-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 8:
        raise ValueError("size must be at least 8")
    csv_path = Path(args.face_weights_csv).expanduser().resolve()
    fmri_path = Path(args.fmri_analysis_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = json.loads(fmri_path.read_text(encoding="utf-8"))
    weights = analysis["emotion_evidence_mass_weights"]
    if set(weights) != set(EMOTIONS):
        raise ValueError(f"Expected fMRI weights for {EMOTIONS}, found {tuple(weights)}")
    weight_values = np.asarray([float(weights[emotion]) for emotion in EMOTIONS], dtype=np.float64)
    if np.any(weight_values < 0) or not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("fMRI evidence-mass weights must be non-negative and sum to one")

    components = load_diagnostic_maps(csv_path, args.size)
    component_bank = np.stack([components[emotion] for emotion in EMOTIONS]).astype(np.float32)
    combined = normalize(np.tensordot(weight_values, component_bank, axes=(0, 0))).astype(np.float32)
    equal_weight = normalize(component_bank.mean(axis=0)).astype(np.float32)

    component_path = output_dir / "published_face_component_bank.npy"
    combined_path = output_dir / "fmri_weighted_functional_face_prior.npy"
    equal_path = output_dir / "equal_weight_functional_face_prior.npy"
    np.save(component_path, component_bank, allow_pickle=False)
    np.save(combined_path, combined, allow_pickle=False)
    np.save(equal_path, equal_weight, allow_pickle=False)
    save_preview(components, combined, output_dir / "functional_face_prior_preview.png")

    metadata = {
        "method": "cross-modal condition weighting without geometric brain-to-face warping",
        "face_space_source": {
            "citation": "Wegrzyn et al. (2017), PLOS ONE 12:e0177239",
            "doi": "10.1371/journal.pone.0177239",
            "table": str(csv_path),
            "table_sha256": sha256(csv_path),
            "rows": {emotion: ["f", "m"] for emotion in EMOTIONS},
            "tile_layout": [8, 6],
            "transformation": "average female/male rows, reproduce the published x-major 6-column by 8-row tile order, min-max normalize all 48 values, bilinear resize",
        },
        "fmri_source": {
            "analysis": str(fmri_path),
            "analysis_sha256": sha256(fmri_path),
            "dataset": analysis["source_dataset"],
            "dataset_doi": analysis["source_dataset_doi"],
            "collection": analysis["source_collection"],
            "weights": {emotion: float(weights[emotion]) for emotion in EMOTIONS},
            "weight_definition": "normalized suprathreshold 1-P excess mass at the preregistered 0.95 threshold",
        },
        "outputs": {
            "component_bank": {"path": str(component_path), "sha256": sha256(component_path), "shape": list(component_bank.shape)},
            "fmri_weighted_prior": {"path": str(combined_path), "sha256": sha256(combined_path), "shape": list(combined.shape)},
            "equal_weight_control": {"path": str(equal_path), "sha256": sha256(equal_path), "shape": list(equal_weight.shape)},
        },
        "interpretation_limit": "The fMRI maps supply condition-level weights only. No voxel, brain coordinate, or anatomical region is mapped to a facial pixel or muscle.",
    }
    metadata_path = output_dir / "functional_face_prior_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": str(metadata_path), "weights": metadata["fmri_source"]["weights"]}, indent=2))


if __name__ == "__main__":
    main()
