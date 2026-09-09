"""Two-dimensional Thin-Plate Spline utilities.

The implementation maps target coordinates back to source coordinates before
sampling an image.  This avoids holes in the warped output.  Control-point
correspondences are experimental inputs and are never inferred anatomically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates


def _kernel(r2: np.ndarray) -> np.ndarray:
    """TPS radial kernel U(r) = r^2 log(r), expressed from squared radius."""
    safe = np.maximum(r2, np.finfo(np.float64).eps)
    return 0.5 * r2 * np.log(safe)


@dataclass
class ThinPlateSpline2D:
    """Fit and evaluate a 2D Thin-Plate Spline transform."""

    control_points: np.ndarray
    weights: np.ndarray
    affine: np.ndarray
    regularization: float = 0.0

    @classmethod
    def fit(
        cls,
        source_points: np.ndarray,
        target_points: np.ndarray,
        regularization: float = 0.0,
    ) -> "ThinPlateSpline2D":
        source = np.asarray(source_points, dtype=np.float64)
        target = np.asarray(target_points, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("source_points must have shape [N,2]")
        if target.shape != source.shape:
            raise ValueError("target_points must have the same [N,2] shape")
        if source.shape[0] < 3:
            raise ValueError("TPS requires at least three non-collinear points")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")

        delta = source[:, None, :] - source[None, :, :]
        kernel = _kernel(np.sum(delta * delta, axis=-1))
        if regularization:
            kernel = kernel + regularization * np.eye(source.shape[0])
        design = np.concatenate((np.ones((source.shape[0], 1)), source), axis=1)
        zeros = np.zeros((3, 3), dtype=np.float64)
        system = np.block([[kernel, design], [design.T, zeros]])
        rhs = np.concatenate((target, np.zeros((3, 2), dtype=np.float64)), axis=0)
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "TPS control points are singular or nearly collinear; "
                "check correspondences or increase regularization"
            ) from exc
        return cls(
            control_points=source,
            weights=solution[: source.shape[0]],
            affine=solution[source.shape[0] :],
            regularization=float(regularization),
        )

    def transform(self, points: np.ndarray) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        original_shape = query.shape
        if query.ndim < 2 or original_shape[-1] != 2:
            raise ValueError("points must have final dimension 2")
        flat = query.reshape(-1, 2)
        delta = flat[:, None, :] - self.control_points[None, :, :]
        radial = _kernel(np.sum(delta * delta, axis=-1))
        affine_design = np.concatenate((np.ones((flat.shape[0], 1)), flat), axis=1)
        mapped = radial @ self.weights + affine_design @ self.affine
        return mapped.reshape(original_shape)


def warp_image_tps(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    output_shape: Tuple[int, int],
    regularization: float = 0.0,
    interpolation_order: int = 1,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Warp a scalar 2D map from source space to target space.

    `source_points[i]` and `target_points[i]` must identify the same experimental
    correspondence.  The inverse TPS (target -> source) is fitted for resampling.
    Coordinates use (x, y) order; array sampling internally uses (row=y, col=x).
    """
    source_image = np.asarray(image, dtype=np.float64)
    if source_image.ndim != 2:
        raise ValueError("image must be a scalar 2D array")
    height, width = int(output_shape[0]), int(output_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("output_shape must contain positive integers")

    inverse = ThinPlateSpline2D.fit(
        np.asarray(target_points, dtype=np.float64),
        np.asarray(source_points, dtype=np.float64),
        regularization=regularization,
    )
    yy, xx = np.mgrid[0:height, 0:width]
    target_grid = np.stack((xx, yy), axis=-1)
    source_grid = inverse.transform(target_grid)
    # Numerical TPS solves can place exact border points a few ulps outside.
    # Snap only near-border coordinates; genuinely out-of-frame samples retain
    # the requested constant fill value.
    tolerance = 1e-7
    source_grid[..., 0] = np.where(
        np.isclose(source_grid[..., 0], 0.0, atol=tolerance), 0.0, source_grid[..., 0]
    )
    source_grid[..., 0] = np.where(
        np.isclose(source_grid[..., 0], source_image.shape[1] - 1, atol=tolerance),
        source_image.shape[1] - 1,
        source_grid[..., 0],
    )
    source_grid[..., 1] = np.where(
        np.isclose(source_grid[..., 1], 0.0, atol=tolerance), 0.0, source_grid[..., 1]
    )
    source_grid[..., 1] = np.where(
        np.isclose(source_grid[..., 1], source_image.shape[0] - 1, atol=tolerance),
        source_image.shape[0] - 1,
        source_grid[..., 1],
    )
    sampled = map_coordinates(
        source_image,
        [source_grid[..., 1], source_grid[..., 0]],
        order=interpolation_order,
        mode="constant",
        cval=float(fill_value),
        prefilter=interpolation_order > 1,
    )
    return sampled.astype(np.float32)
