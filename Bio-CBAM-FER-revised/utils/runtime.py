"""Runtime, provenance, and reproducibility helpers."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_prior_bank(paths: Sequence[str]) -> Optional[Tensor]:
    if not paths:
        return None
    maps = []
    expected_shape = None
    for item in paths:
        path = Path(item).expanduser().resolve()
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2:
            raise ValueError(f"Prior {path} must be a 2D array")
        if not np.isfinite(array).all():
            raise ValueError(f"Prior {path} contains NaN or infinite values")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError("All priors in a bank must have the same dimensions")
        low, high = float(array.min()), float(array.max())
        normalized = np.zeros_like(array, dtype=np.float32) if high - low <= 1e-12 else ((array - low) / (high - low)).astype(np.float32)
        maps.append(normalized)
    return torch.from_numpy(np.stack(maps, axis=0))


def git_revision(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_report(project_root: Optional[Path] = None) -> Dict[str, object]:
    report: Dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_count"] = torch.cuda.device_count()
    if project_root is not None:
        report["git_revision"] = git_revision(project_root)
    return report


def save_json(payload: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
