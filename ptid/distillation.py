"""Core spatial attention-transfer objective for PTID."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def freeze_teacher(teacher: nn.Module) -> nn.Module:
    """Freeze Teacher parameters and batch-normalization statistics."""

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def spatial_attention_map(features: Tensor) -> Tensor:
    """Channel-mean squared activation, spatially L2-normalized per sample."""

    if features.ndim != 4:
        raise ValueError("features must have shape [B, C, H, W]")
    attention = features.square().mean(dim=1, keepdim=True)
    flattened = attention.flatten(start_dim=1)
    epsilon = torch.finfo(flattened.dtype).eps
    normalized = F.normalize(flattened, p=2, dim=1, eps=epsilon)
    return normalized.reshape_as(attention)


def _normalized_layer_weights(
    layers: Sequence[str],
    layer_weights: Sequence[float],
    reference: Tensor,
) -> Tensor:
    if len(layers) == 0 or len(layers) != len(layer_weights):
        raise ValueError("layers and layer_weights must have the same non-zero length")
    weights = reference.new_tensor(tuple(float(value) for value in layer_weights))
    if not torch.isfinite(weights).all() or torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("layer_weights must be finite, non-negative, and sum to a positive value")
    return weights / weights.sum()


def attention_transfer_loss(
    student_auxiliary: Mapping[str, Tensor],
    teacher_features: Mapping[str, Tensor],
    layers: Sequence[str],
    layer_weights: Sequence[float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted squared-L2 attention distance over selected encoder layers."""

    selected_layers = tuple(layers)
    if not selected_layers:
        raise ValueError("at least one distillation layer is required")
    reference = student_auxiliary[selected_layers[0]]
    weights = _normalized_layer_weights(selected_layers, layer_weights, reference)
    per_layer: dict[str, Tensor] = {}
    for layer in selected_layers:
        student_attention = spatial_attention_map(student_auxiliary[layer])
        teacher_attention = spatial_attention_map(teacher_features[layer].detach())
        if student_attention.shape != teacher_attention.shape:
            raise ValueError(
                f"attention shape mismatch at {layer}: "
                f"student={tuple(student_attention.shape)}, "
                f"teacher={tuple(teacher_attention.shape)}"
            )
        difference = (student_attention - teacher_attention).flatten(start_dim=1)
        per_layer[layer] = difference.square().sum(dim=1).mean()
    stacked = torch.stack(tuple(per_layer[layer] for layer in selected_layers))
    return torch.sum(weights * stacked), per_layer


def ptid_loss(
    predicted_bm: Tensor,
    predicted_ubm: Tensor,
    target_bm: Tensor,
    target_ubm: Tensor,
    student_auxiliary: Mapping[str, Tensor],
    teacher_features: Mapping[str, Tensor],
    layers: Sequence[str],
    layer_weights: Sequence[float],
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """BM MSE + UBM MSE + beta times spatial attention transfer."""

    if not torch.isfinite(torch.tensor(beta)) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    bm_loss = F.mse_loss(predicted_bm, target_bm)
    ubm_loss = F.mse_loss(predicted_ubm, target_ubm)
    distillation, per_layer = attention_transfer_loss(
        student_auxiliary,
        teacher_features,
        layers,
        layer_weights,
    )
    total = bm_loss + ubm_loss + float(beta) * distillation
    components = {
        "total": total,
        "bm": bm_loss,
        "ubm": ubm_loss,
        "distillation": distillation,
        **{f"distillation_{name}": value for name, value in per_layer.items()},
    }
    return total, components

