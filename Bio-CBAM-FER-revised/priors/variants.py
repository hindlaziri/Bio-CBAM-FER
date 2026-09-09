"""Control-prior construction for Bio-CBAM ablation studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .fmri_pipeline import normalize_map, save_prior


def random_matched_prior(reference: np.ndarray, seed: int) -> np.ndarray:
    """Preserve the exact value distribution while destroying spatial structure."""
    reference = normalize_map(reference)
    rng = np.random.default_rng(seed)
    flattened = reference.ravel().copy()
    rng.shuffle(flattened)
    return flattened.reshape(reference.shape).astype(np.float32)


def gaussian_center_prior(height: int, width: int, sigma_fraction: float = 0.2) -> np.ndarray:
    if not 0 < sigma_fraction <= 1:
        raise ValueError("sigma_fraction must lie in (0,1]")
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    sigma_y, sigma_x = sigma_fraction * height, sigma_fraction * width
    prior = np.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))
    return normalize_map(prior)


def spectral_residual_saliency(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Hou-Zhang spectral-residual saliency, implemented deterministically."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    small = cv2.resize(gray.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    average = cv2.blur(log_amplitude, (3, 3))
    residual = log_amplitude - average
    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency = np.abs(reconstructed) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    resized = cv2.resize(saliency, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
    return normalize_map(resized)


def _iter_manifest_images(manifest_path: Path, split: str = "train") -> Iterable[np.ndarray]:
    frame = pd.read_csv(manifest_path)
    required = {"path", "split"}
    if not required.issubset(frame.columns):
        raise ValueError("Manifest must contain path and split columns")
    subset = frame.loc[frame["split"].astype(str).str.lower() == split.lower()]
    for value in subset["path"]:
        path = Path(str(value))
        if not path.is_absolute():
            path = manifest_path.parent / path
        yield np.asarray(Image.open(path).convert("RGB"))


def _iter_fer_images(csv_path: Path, usage: str = "Training") -> Iterable[np.ndarray]:
    frame = pd.read_csv(csv_path)
    required = {"pixels", "Usage"}
    if not required.issubset(frame.columns):
        raise ValueError("FER CSV must contain pixels and Usage columns")
    for value in frame.loc[frame["Usage"] == usage, "pixels"]:
        pixels = np.fromstring(str(value), sep=" ", dtype=np.uint8)
        if pixels.size != 2304:
            raise ValueError("FER image does not contain 2304 pixels")
        gray = pixels.reshape(48, 48)
        yield cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def aggregate_generic_saliency(
    images: Iterable[np.ndarray],
    output_shape: tuple[int, int],
    max_images: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    total = np.zeros(output_shape, dtype=np.float64)
    count = 0
    for image in images:
        total += spectral_residual_saliency(image, output_shape)
        count += 1
        if max_images is not None and count >= max_images:
            break
    if count == 0:
        raise ValueError("No training images were available for generic saliency")
    return normalize_map(total / count), count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="variant", required=True)

    random_parser = sub.add_parser("random_matched")
    random_parser.add_argument("--reference", required=True)
    random_parser.add_argument("--seed", type=int, required=True)
    random_parser.add_argument("--output-stem", required=True)

    gaussian = sub.add_parser("gaussian_center")
    gaussian.add_argument("--height", type=int, default=224)
    gaussian.add_argument("--width", type=int, default=224)
    gaussian.add_argument("--sigma-fraction", type=float, default=0.2)
    gaussian.add_argument("--output-stem", required=True)

    saliency = sub.add_parser("generic_saliency")
    source = saliency.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fer-csv")
    saliency.add_argument("--height", type=int, default=224)
    saliency.add_argument("--width", type=int, default=224)
    saliency.add_argument("--max-images", type=int)
    saliency.add_argument("--output-stem", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = {"variant": args.variant}
    if args.variant == "random_matched":
        reference = np.load(args.reference, allow_pickle=False)
        prior = random_matched_prior(reference, args.seed)
        metadata.update({"reference": str(Path(args.reference).resolve()), "seed": args.seed})
    elif args.variant == "gaussian_center":
        prior = gaussian_center_prior(args.height, args.width, args.sigma_fraction)
        metadata.update({"height": args.height, "width": args.width, "sigma_fraction": args.sigma_fraction})
    else:
        shape = (args.height, args.width)
        if args.manifest:
            images = _iter_manifest_images(Path(args.manifest).resolve())
            source_description = {"manifest": str(Path(args.manifest).resolve()), "split": "train"}
        else:
            images = _iter_fer_images(Path(args.fer_csv).resolve())
            source_description = {"fer_csv": str(Path(args.fer_csv).resolve()), "usage": "Training"}
        prior, count = aggregate_generic_saliency(images, shape, args.max_images)
        metadata.update({**source_description, "algorithm": "Hou-Zhang spectral residual", "image_count": count})
    outputs = save_prior(prior, Path(args.output_stem), metadata)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
