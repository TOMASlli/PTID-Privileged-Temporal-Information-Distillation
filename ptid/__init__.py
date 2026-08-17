"""Core PTID model and attention-transfer API."""

from .distillation import (
    attention_transfer_loss,
    freeze_teacher,
    ptid_loss,
    spatial_attention_map,
)
from .model import PTIDStudent, StudentBackbone, TemporalTeacher

__all__ = [
    "PTIDStudent",
    "StudentBackbone",
    "TemporalTeacher",
    "attention_transfer_loss",
    "freeze_teacher",
    "ptid_loss",
    "spatial_attention_map",
]

