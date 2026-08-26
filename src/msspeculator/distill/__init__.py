"""Distillation: datasets, losses, and the Lightning training pipeline."""

from .dataset import DistillDataset, LabeledBatch, collate_with_labels
from .lightning import DistillModule, fit_distill
from .losses import distill_loss, ms2_cosine_loss, spectral_angle
from .pipeline import RunConfig, run_pipeline

__all__ = [
    "DistillDataset",
    "LabeledBatch",
    "collate_with_labels",
    "ms2_cosine_loss",
    "spectral_angle",
    "distill_loss",
    "DistillModule",
    "fit_distill",
    "RunConfig",
    "run_pipeline",
]
