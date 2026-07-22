"""Distillation: datasets, losses, and the training loop."""

from .dataset import DistillDataset, LabeledBatch, collate_with_labels
from .losses import distill_loss, ms2_cosine_loss, spectral_angle
from .streaming import build_val_set, curriculum_batches, estimate_norm
from .trainer import TrainConfig, evaluate, train, train_streaming

__all__ = [
    "DistillDataset",
    "LabeledBatch",
    "collate_with_labels",
    "ms2_cosine_loss",
    "spectral_angle",
    "distill_loss",
    "TrainConfig",
    "train",
    "train_streaming",
    "evaluate",
    "curriculum_batches",
    "estimate_norm",
    "build_val_set",
]
