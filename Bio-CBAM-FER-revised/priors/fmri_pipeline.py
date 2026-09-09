"""Reproducible pipeline for constructing 2D priors from fMRI statistics.

The pipeline deliberately separates neuroimaging preprocessing from the
brain-to-face correspondence hypothesis. It accepts either a 2D cortical
projection exported by established neuroimaging software or a 3D statistical
volume that can be reduced to a documented 2D projection for sensitivity tests.
The TPS correspondences are supplied explicitly in a CSV file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .tps import warp_image_tps


@dataclass(frozen=True)
class PriorBuildConfig:
    source_path: str
    correspondence_path: str
    output_height: int = 224
    output_width: int = 224
    volume_index: Optional[int] = None
    projection_axis: int = 2
    projection_method: str = "max_abs"
    slice_index: Optional[int] = None
    activation_mode: str = "positive"
    threshold_percentile: Optional[float] = None
    smooth_sigma: float = 2.0
    tps_regularization: float = 1e-3
    interpolation_order: int = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_activation(path: Path, volume_index: Optional[int] = None) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.keys())
            if len(keys) != 1:
                raise ValueError("NPZ input must contain exactly one array")
            array = archive[keys[0]]
    elif suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("Install nibabel to read NIfTI files") from exc
        array = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        array = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported activation format: {path.name}")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        if volume_index is None:
            raise ValueError("A 4D input requires --volume-index")
        array = array[..., int(volume_index)]
    if array.ndim not in (2, 3):
        raise ValueError("Activation input must be 2D, 3D, or 4D with a selected volume")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def project_volume(
    volume: np.ndarray,
    axis: int = 2,
    method: str = "max_abs",
    slice_index: Optional[int] = None,
) -> np.ndarray:
    if volume.ndim == 2:
        return volume.astype(np.float32, copy=False)
    if axis not in (0, 1, 2):
        raise ValueError("projection_axis must be 0, 1, or 2")
    if method == "mean_abs":
        projected = np.mean(np.abs(volume), axis=axis)
    elif method == "max_abs":
        indices = np.argmax(np.abs(volume), axis=axis)
        projected = np.take_along_axis(volume, np.expand_dims(indices, axis), axis=axis).squeeze(axis)
    elif method == "maximum":
        projected = np.max(volume, axis=axis)
    elif method == "slice":
        if slice_index is None:
            raise ValueError("projection_method=slice requires --slice-index")
        projected = np.take(volume, int(slice_index), axis=axis)
    else:
        raise ValueError("projection_method must be max_abs, mean_abs, maximum, or slice")
    return np.asarray(projected, dtype=np.float32)


def select_activation(stat_map: np.ndarray, mode: str = "positive") -> np.ndarray:
    if mode == "positive":
        return np.clip(stat_map, 0, None)
    if mode == "negative":
        return np.clip(-stat_map, 0, None)
    if mode == "absolute":
        return np.abs(stat_map)
    raise ValueError("activation_mode must be positive, negative, or absolute")


def normalize_map(array: np.ndarray, threshold_percentile: Optional[float] = None) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if threshold_percentile is not None:
        if not 0 <= threshold_percentile <= 100:
            raise ValueError("threshold_percentile must lie in [0,100]")
        nonzero = result[result > 0]
        if nonzero.size:
            cutoff = np.percentile(nonzero, threshold_percentile)
            result = np.where(result >= cutoff, result, 0.0)
    low, high = float(result.min()), float(result.max())
    if high - low <= 1e-12:
        return np.zeros_like(result, dtype=np.float32)
    return ((result - low) / (high - low)).astype(np.float32)


def load_correspondences(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    required = ("source_x", "source_y", "target_x", "target_y")
    source, target = [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"Correspondence CSV must contain: {', '.join(required)}")
        for row in reader:
            source.append((float(row["source_x"]), float(row["source_y"])))
            target.append((float(row["target_x"]), float(row["target_y"])))
    if len(source) < 3:
        raise ValueError("At least three correspondences are required")
    return np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)


def save_prior(prior: np.ndarray, output_stem: Path, metadata: Dict[str, object]) -> Dict[str, str]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    npy_path = output_stem.with_suffix(".npy")
    png_path = output_stem.with_suffix(".png")
    json_path = output_stem.with_suffix(".json")
    np.save(npy_path, prior.astype(np.float32), allow_pickle=False)
    Image.fromarray(np.round(prior * 255).astype(np.uint8), mode="L").save(png_path)
    metadata = dict(metadata)
    metadata["prior_sha256"] = sha256_file(npy_path)
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"npy": str(npy_path), "png": str(png_path), "metadata": str(json_path)}


def build_prior(config: PriorBuildConfig, output_stem: Path) -> Dict[str, str]:
    source_path = Path(config.source_path).expanduser().resolve()
    correspondence_path = Path(config.correspondence_path).expanduser().resolve()
    activation = load_activation(source_path, config.volume_index)
    projected = project_volume(
        activation,
        axis=config.projection_axis,
        method=config.projection_method,
        slice_index=config.slice_index,
    )
    selected = select_activation(projected, config.activation_mode)
    normalized = normalize_map(selected, config.threshold_percentile)
    source_points, target_points = load_correspondences(correspondence_path)
    warped = warp_image_tps(
        normalized,
        source_points,
        target_points,
        output_shape=(config.output_height, config.output_width),
        regularization=config.tps_regularization,
        interpolation_order=config.interpolation_order,
    )
    if config.smooth_sigma > 0:
        warped = gaussian_filter(warped, sigma=config.smooth_sigma)
    prior = normalize_map(warped)
    metadata: Dict[str, object] = {
        "configuration": asdict(config),
        "source_sha256": sha256_file(source_path),
        "correspondence_sha256": sha256_file(correspondence_path),
        "source_shape": list(activation.shape),
        "projected_shape": list(projected.shape),
        "output_shape": list(prior.shape),
        "correspondence_count": int(source_points.shape[0]),
        "interpretation": (
            "Functional spatial prior created from an explicitly supplied correspondence; "
            "not an anatomical brain-to-muscle projection."
        ),
    }
    return save_prior(prior, output_stem, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--correspondences", required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--volume-index", type=int)
    parser.add_argument("--projection-axis", type=int, default=2)
    parser.add_argument("--projection-method", choices=("max_abs", "mean_abs", "maximum", "slice"), default="max_abs")
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--activation-mode", choices=("positive", "negative", "absolute"), default="positive")
    parser.add_argument("--threshold-percentile", type=float)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--tps-regularization", type=float, default=1e-3)
    parser.add_argument("--interpolation-order", type=int, choices=range(0, 6), default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PriorBuildConfig(
        source_path=args.source,
        correspondence_path=args.correspondences,
        output_height=args.height,
        output_width=args.width,
        volume_index=args.volume_index,
        projection_axis=args.projection_axis,
        projection_method=args.projection_method,
        slice_index=args.slice_index,
        activation_mode=args.activation_mode,
        threshold_percentile=args.threshold_percentile,
        smooth_sigma=args.smooth_sigma,
        tps_regularization=args.tps_regularization,
        interpolation_order=args.interpolation_order,
    )
    outputs = build_prior(config, Path(args.output_stem))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
