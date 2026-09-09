"""Bio-CBAM multi-scale architecture.

A ResNet backbone exposes four residual stages and a Bio-CBAM block follows
all stages. Spatial-attention logits are fused additively with an externally
supplied spatial prior. No experimental result is hard-coded in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BioCBAMConfig:
    architecture: str = "biocbam"
    num_classes: int = 7
    backbone: str = "resnet50"
    pretrained: bool = False
    reduction_ratio: int = 16
    spatial_kernel_size: int = 7
    lambda_init: float = 0.0
    lambda_nonnegative: bool = False
    shared_lambda: bool = False
    lambda_regularization: float = 0.0
    classifier_dropout: float = 0.5


class ChannelAttention(nn.Module):
    """Channel attention from CBAM."""

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttentionLogits(nn.Module):
    """Return CBAM spatial-attention logits before the sigmoid."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("spatial_kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        return self.conv(torch.cat((average, maximum), dim=1))


class ScalarGate(nn.Module):
    """Learnable scalar gate with an optional non-negative constraint."""

    def __init__(self, initial_value: float = 0.0, nonnegative: bool = False) -> None:
        super().__init__()
        self.nonnegative = bool(nonnegative)
        if self.nonnegative:
            if initial_value < 0:
                raise ValueError("A non-negative gate cannot start below zero")
            raw = -20.0 if initial_value == 0 else math.log(math.expm1(initial_value))
        else:
            raw = float(initial_value)
        self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def value(self) -> Tensor:
        return F.softplus(self.raw) if self.nonnegative else self.raw

    def forward(self, x: Tensor) -> Tensor:
        return self.value() * x


class BioCBAMBlock(nn.Module):
    """CBAM block with additive spatial-prior fusion.

    M_b = sigmoid(Z_b + lambda_b H_b), and F'_b = F_b elementwise-multiplied by M_b.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
        gate: Optional[ScalarGate] = None,
        lambda_init: float = 0.0,
        lambda_nonnegative: bool = False,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_logits = SpatialAttentionLogits(spatial_kernel_size)
        self.gate = gate or ScalarGate(lambda_init, lambda_nonnegative)

    @staticmethod
    def _prepare_prior(prior: Optional[Tensor], x: Tensor) -> Tensor:
        if prior is None:
            return torch.zeros((x.shape[0], 1, *x.shape[-2:]), dtype=x.dtype, device=x.device)
        if prior.ndim == 2:
            prior = prior[None, None]
        elif prior.ndim == 3:
            prior = prior[:, None]
        if prior.ndim != 4 or prior.shape[1] != 1:
            raise ValueError("prior must be [H,W], [B,H,W], or [B,1,H,W]")
        prior = prior.to(device=x.device, dtype=x.dtype)
        if prior.shape[0] == 1 and x.shape[0] > 1:
            prior = prior.expand(x.shape[0], -1, -1, -1)
        if prior.shape[0] != x.shape[0]:
            raise ValueError("prior batch size must be one or match the feature batch")
        if prior.shape[-2:] != x.shape[-2:]:
            prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
        flat = prior.flatten(2)
        minimum = flat.amin(dim=-1, keepdim=True).unsqueeze(-1)
        maximum = flat.amax(dim=-1, keepdim=True).unsqueeze(-1)
        return (prior - minimum) / (maximum - minimum).clamp_min(1e-8)

    def forward(self, x: Tensor, prior: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        channel_map = self.channel_attention(x)
        channel_refined = x * channel_map
        spatial_logits = self.spatial_logits(channel_refined)
        resized_prior = self._prepare_prior(prior, channel_refined)
        spatial_map = torch.sigmoid(spatial_logits + self.gate(resized_prior))
        return channel_refined * spatial_map, {
            "channel_attention": channel_map,
            "spatial_logits": spatial_logits,
            "resized_prior": resized_prior,
            "spatial_attention": spatial_map,
            "lambda": self.gate.value(),
        }


class CBAMBlock(nn.Module):
    """Standard CBAM block without an external prior or lambda gate."""

    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_logits = SpatialAttentionLogits(spatial_kernel_size)

    def forward(self, x: Tensor, prior: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        if prior is not None:
            raise ValueError("The CBAM baseline does not accept an external prior")
        channel_map = self.channel_attention(x)
        channel_refined = x * channel_map
        spatial_logits = self.spatial_logits(channel_refined)
        spatial_map = torch.sigmoid(spatial_logits)
        return channel_refined * spatial_map, {
            "channel_attention": channel_map,
            "spatial_logits": spatial_logits,
            "spatial_attention": spatial_map,
        }


class PriorBankMixer(nn.Module):
    """Mix K priors from visual features, never from the ground-truth label."""

    def __init__(self, priors: Tensor, selector_channels: int) -> None:
        super().__init__()
        if priors.ndim == 3:
            priors = priors[:, None]
        if priors.ndim != 4 or priors.shape[1] != 1:
            raise ValueError("priors must be [K,H,W] or [K,1,H,W]")
        self.register_buffer("priors", priors.float())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.selector = nn.Linear(selector_channels, priors.shape[0]) if priors.shape[0] > 1 else None

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        batch = features.shape[0]
        if self.selector is None:
            weights = torch.ones((batch, 1), device=features.device, dtype=features.dtype)
        else:
            weights = torch.softmax(self.selector(self.pool(features).flatten(1)), dim=1)
        bank = self.priors.to(device=features.device, dtype=features.dtype)
        return torch.einsum("bk,kchw->bchw", weights, bank), weights


class BioCBAM(nn.Module):
    """ResNet with Bio-CBAM blocks after all four residual stages."""

    def __init__(self, config: Optional[BioCBAMConfig] = None, prior_bank: Optional[Tensor] = None) -> None:
        super().__init__()
        self.config = config or BioCBAMConfig()
        if self.config.architecture not in {"cbam", "biocbam"}:
            raise ValueError("BioCBAM requires architecture='cbam' or 'biocbam'")
        if self.config.backbone not in {"resnet18", "resnet50"}:
            raise ValueError("backbone must be resnet18 or resnet50")

        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
        if self.config.backbone == "resnet50":
            base = resnet50(weights=ResNet50_Weights.DEFAULT if self.config.pretrained else None)
            stage_channels = (256, 512, 1024, 2048)
        else:
            base = resnet18(weights=ResNet18_Weights.DEFAULT if self.config.pretrained else None)
            stage_channels = (64, 128, 256, 512)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stages = nn.ModuleList((base.layer1, base.layer2, base.layer3, base.layer4))
        if self.config.architecture == "biocbam":
            shared_gate = ScalarGate(self.config.lambda_init, self.config.lambda_nonnegative) if self.config.shared_lambda else None
            blocks = [
                BioCBAMBlock(
                    channels,
                    self.config.reduction_ratio,
                    self.config.spatial_kernel_size,
                    gate=shared_gate,
                    lambda_init=self.config.lambda_init,
                    lambda_nonnegative=self.config.lambda_nonnegative,
                )
                for channels in stage_channels
            ]
            self.prior_mixer = PriorBankMixer(prior_bank, stage_channels[0]) if prior_bank is not None else None
        else:
            if prior_bank is not None:
                raise ValueError("CBAM baseline must not receive a prior bank")
            blocks = [
                CBAMBlock(channels, self.config.reduction_ratio, self.config.spatial_kernel_size)
                for channels in stage_channels
            ]
            self.prior_mixer = None
        self.bio_blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(stage_channels[-1], self.config.num_classes),
        )

    def _unique_gate_parameters(self) -> List[Tensor]:
        output: List[Tensor] = []
        seen = set()
        for block in self.bio_blocks:
            if not hasattr(block, "gate"):
                continue
            parameter = block.gate.raw
            if id(parameter) not in seen:
                seen.add(id(parameter))
                output.append(parameter)
        return output

    def lambda_values(self) -> Tensor:
        values = [block.gate.value() for block in self.bio_blocks if hasattr(block, "gate")]
        return torch.stack(values) if values else next(self.parameters()).new_empty((0,))

    def lambda_regularization_loss(self) -> Tensor:
        parameters = self._unique_gate_parameters()
        if not parameters:
            return next(self.parameters()).new_zeros(())
        penalty = torch.stack([p.square() for p in parameters]).sum()
        return penalty * float(self.config.lambda_regularization)

    def forward(self, x: Tensor, prior: Optional[Tensor] = None, return_features: bool = False) -> Tuple[Tensor, Dict[str, object]]:
        diagnostics: Dict[str, object] = {"stages": []}
        x = self.stem(x)
        if self.config.architecture == "cbam" and prior is not None:
            raise ValueError("CBAM baseline does not accept an external prior")
        mixed_prior = prior
        selector_weights: Optional[Tensor] = None
        for index, (stage, block) in enumerate(zip(self.stages, self.bio_blocks)):
            x = stage(x)
            if index == 0 and mixed_prior is None and self.prior_mixer is not None:
                mixed_prior, selector_weights = self.prior_mixer(x)
            x, stage_info = block(x, mixed_prior)
            diagnostics["stages"].append(stage_info)
        features = self.pool(x).flatten(1)
        logits = self.classifier(features)
        diagnostics["lambda_values"] = self.lambda_values()
        if selector_weights is not None:
            diagnostics["prior_selector_weights"] = selector_weights
        if return_features:
            diagnostics["features"] = features
        return logits, diagnostics



class ResNetBaseline(nn.Module):
    """ResNet baseline with the same dropout-linear classification head."""

    def __init__(self, config: BioCBAMConfig) -> None:
        super().__init__()
        if config.architecture != "resnet":
            raise ValueError("ResNetBaseline requires architecture='resnet'")
        self.config = config
        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
        if config.backbone == "resnet50":
            base = resnet50(weights=ResNet50_Weights.DEFAULT if config.pretrained else None)
            feature_dim = 2048
        elif config.backbone == "resnet18":
            base = resnet18(weights=ResNet18_Weights.DEFAULT if config.pretrained else None)
            feature_dim = 512
        else:
            raise ValueError("backbone must be resnet18 or resnet50")
        base.fc = nn.Sequential(nn.Dropout(config.classifier_dropout), nn.Linear(feature_dim, config.num_classes))
        self.network = base
        self.prior_mixer = None

    def lambda_values(self) -> Tensor:
        return next(self.parameters()).new_empty((0,))

    def lambda_regularization_loss(self) -> Tensor:
        return next(self.parameters()).new_zeros(())

    def forward(self, x: Tensor, prior: Optional[Tensor] = None, return_features: bool = False) -> Tuple[Tensor, Dict[str, object]]:
        if prior is not None:
            raise ValueError("The ResNet baseline does not accept a spatial prior")
        logits = self.network(x)
        return logits, {"stages": [], "lambda_values": self.lambda_values()}


def create_model(config: BioCBAMConfig, prior_bank: Optional[Tensor] = None) -> nn.Module:
    """Create the exact architecture named in a serializable configuration."""
    if config.architecture == "resnet":
        if prior_bank is not None:
            raise ValueError("A ResNet baseline cannot receive a prior bank")
        return ResNetBaseline(config)
    if config.architecture == "cbam":
        if prior_bank is not None:
            raise ValueError("CBAM baseline must not receive an external prior")
        return BioCBAM(config, prior_bank=None)
    if config.architecture == "biocbam":
        return BioCBAM(config, prior_bank=prior_bank)
    raise ValueError("architecture must be resnet, cbam, or biocbam")


def create_bio_cbam(
    num_classes: int = 7,
    backbone: str = "resnet50",
    use_fmri_prior: bool = True,
    device: Optional[str] = None,
    pretrained: bool = False,
    prior_bank: Optional[Tensor] = None,
    lambda_init: float = 0.0,
    lambda_nonnegative: bool = False,
    shared_lambda: bool = False,
    lambda_regularization: float = 0.0,
) -> BioCBAM:
    if not use_fmri_prior:
        prior_bank = None
    config = BioCBAMConfig(
        architecture="biocbam" if use_fmri_prior else "cbam",
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        lambda_init=lambda_init,
        lambda_nonnegative=lambda_nonnegative,
        shared_lambda=shared_lambda,
        lambda_regularization=lambda_regularization,
    )
    model = create_model(config=config, prior_bank=prior_bank)
    return model.to(device) if device is not None else model
