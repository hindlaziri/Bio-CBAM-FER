"""Assess fMRI-weight and face-prior sensitivity to a prespecified threshold grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import nibabel as nib
import numpy as np

EMOTIONS = ("happy", "angry", "sad")


def _minmax(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    return (values - minimum) / max(maximum - minimum, 1e-12)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.square(x).sum() * np.square(y).sum())
    return float(np.dot(x, y) / max(float(denominator), 1e-12))


def _top_quartile_iou(a: np.ndarray, b: np.ndarray) -> float:
    mask_a = a >= np.quantile(a, 0.75)
    mask_b = b >= np.quantile(b, 0.75)
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / max(int(union), 1))


def analyze(spec_path: Path, component_path: Path, thresholds: Sequence[float]) -> Dict[str, object]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    components = np.load(component_path).astype(np.float64)
    if components.shape[0] != len(EMOTIONS):
        raise ValueError(f"Expected three behavioral component maps, observed {components.shape}")
    maps = {
        emotion: np.nan_to_num(
            nib.load(spec["maps"][emotion]["path"]).get_fdata(dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float64)
        for emotion in EMOTIONS
    }
    records: Dict[str, object] = {}
    priors: Dict[float, np.ndarray] = {}
    for threshold in thresholds:
        masses = np.asarray(
            [np.maximum(maps[emotion] - threshold, 0.0).sum() for emotion in EMOTIONS],
            dtype=np.float64,
        )
        if masses.sum() <= 0:
            raise ValueError(f"Threshold {threshold} yields zero total suprathreshold mass")
        weights = masses / masses.sum()
        prior = _minmax(np.tensordot(weights, components, axes=(0, 0)))
        priors[float(threshold)] = prior
        records[f"{threshold:.3f}"] = {
            "masses": {emotion: float(value) for emotion, value in zip(EMOTIONS, masses)},
            "weights": {emotion: float(value) for emotion, value in zip(EMOTIONS, weights)},
        }
    reference_threshold = 0.95
    if reference_threshold not in priors:
        raise ValueError("Threshold grid must include the preregistered 0.95 reference")
    reference = priors[reference_threshold]
    for threshold, prior in priors.items():
        record = records[f"{threshold:.3f}"]
        record["prior_pearson_vs_0.95"] = _pearson(prior, reference)
        record["prior_mean_absolute_difference_vs_0.95"] = float(np.abs(prior - reference).mean())
        record["prior_maximum_absolute_difference_vs_0.95"] = float(np.abs(prior - reference).max())
        record["prior_top_quartile_iou_vs_0.95"] = _top_quartile_iou(prior, reference)
    return {
        "design": {
            "thresholds": [float(value) for value in thresholds],
            "reference_threshold": reference_threshold,
            "map_semantics": spec["map_semantics"],
            "emotion_order": list(EMOTIONS),
            "selection_changed": False,
            "training_or_test_data_used": False,
        },
        "results": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-spec", required=True)
    parser.add_argument("--component-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.90,0.925,0.95,0.975,0.99")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = tuple(float(item.strip()) for item in args.thresholds.split(",") if item.strip())
    payload = analyze(
        Path(args.map_spec).expanduser().resolve(),
        Path(args.component_bank).expanduser().resolve(),
        thresholds,
    )
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()
