"""Named student presets and checkpoint (de)serialization."""

from __future__ import annotations

from pathlib import Path

import torch

from .student import StudentConfig, StudentModel

# Size presets. Swap freely; benchmark decides the winner.
# "flash" is the throughput pick from forward-only benchmarks: a single-head, single-layer
# d32 transformer hits ~165k precursors/s on CPU and ~266k on MPS (no GPU needed).
PRESETS: dict[str, StudentConfig] = {
    "flash": StudentConfig(backbone="transformer", d_model=32, n_layers=1, n_heads=1),
    "tiny": StudentConfig(backbone="cnn", d_model=48, n_layers=2),
    "small": StudentConfig(backbone="transformer", d_model=64, n_layers=2, n_heads=4),
    "base": StudentConfig(backbone="transformer", d_model=128, n_layers=4, n_heads=8),
}


def build_student(preset_or_cfg: str | StudentConfig) -> StudentModel:
    if isinstance(preset_or_cfg, str):
        try:
            cfg = PRESETS[preset_or_cfg]
        except KeyError as exc:
            raise ValueError(f"unknown preset {preset_or_cfg!r}; known: {sorted(PRESETS)}") from exc
    else:
        cfg = preset_or_cfg
    return StudentModel(cfg)


def save_checkpoint(model: StudentModel, path: str | Path) -> None:
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> StudentModel:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = StudentModel(StudentConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
