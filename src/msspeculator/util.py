"""Small shared helpers."""

from __future__ import annotations


def resolve_device(device: str) -> str:
    """Map 'auto' to mps (Apple Silicon) or cpu; pass explicit choices through."""
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
