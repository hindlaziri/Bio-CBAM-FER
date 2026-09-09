"""Checkpoint validation and model reconstruction helpers."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from models import BioCBAMConfig, create_model


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a trusted local Bio-CBAM checkpoint and validate its structure.

    PyTorch checkpoints use pickle internally. Do not load files from untrusted
    sources. The package only loads checkpoints explicitly selected by the user.
    """
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint; expected a Bio-CBAM format_version=2 dictionary")
    required = {"model_config", "model_state", "class_names"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    return checkpoint


def model_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[torch.nn.Module, BioCBAMConfig]:
    saved_config = BioCBAMConfig(**checkpoint["model_config"])
    # State dict restoration supersedes ImageNet initialization and avoids any
    # network access when a checkpoint records pretrained=True.
    initialization_config = replace(saved_config, pretrained=False)
    model = create_model(initialization_config, prior_bank=checkpoint.get("prior_bank"))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.config = saved_config
    return model, saved_config
