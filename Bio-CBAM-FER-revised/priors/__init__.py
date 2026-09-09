"""Spatial-prior construction utilities for Bio-CBAM."""

from .fmri_pipeline import (
    PriorBuildConfig,
    build_prior,
    load_activation,
    load_correspondences,
    normalize_map,
    project_volume,
    select_activation,
)
from .tps import ThinPlateSpline2D, warp_image_tps

__all__ = [
    "PriorBuildConfig",
    "build_prior",
    "load_activation",
    "load_correspondences",
    "normalize_map",
    "project_volume",
    "select_activation",
    "ThinPlateSpline2D",
    "warp_image_tps",
]
