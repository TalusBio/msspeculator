"""Named student presets and checkpoint (de)serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .context import (
    DEFAULT_DETECTORS,
    DEFAULT_FRAGMENTATIONS,
    DEFAULT_INSTRUMENTS,
    ChromRunbook,
    MSContextEncoder,
)
from .student import StudentConfig, StudentModel

# Size presets. Benchmark the complete library-generation workload on the deployment hardware;
# old forward-only rates are not representative of digestion, chemistry, and serialization.
PRESETS: dict[str, StudentConfig] = {
    "flash": StudentConfig(backbone="transformer", d_model=32, n_layers=1, n_heads=1),
    "small-2h": StudentConfig(backbone="transformer", d_model=64, n_layers=2, n_heads=2),
    "small": StudentConfig(backbone="transformer", d_model=64, n_layers=2, n_heads=4),
    "base-4h": StudentConfig(backbone="transformer", d_model=128, n_layers=4, n_heads=4),
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


@dataclass
class ContextBundle:
    """Trained acquisition context that rides alongside a checkpoint. Either half may be
    ``None`` (e.g. stream pretrain saves an encoder but no per-run book)."""

    encoder: MSContextEncoder | None = None
    runbook: ChromRunbook | None = None
    dataset_index: dict | None = None  # dataset name -> ChromRunbook row (0 = iRT/neutral)


def _encoder_blob(enc: MSContextEncoder) -> dict:
    return {
        "context_dim": enc.context_dim,
        "instruments": enc.instruments,
        "detectors": enc.detectors,
        "fragmentations": enc.fragmentations,
        "state_dict": enc.state_dict(),
    }


def _runbook_blob(book: ChromRunbook) -> dict:
    return {
        "n_datasets": book.n_datasets,
        "context_dim": book.context_dim,
        "state_dict": book.state_dict(),
    }


def save_checkpoint(
    model: StudentModel,
    path: str | Path,
    *,
    encoder: MSContextEncoder | None = None,
    runbook: ChromRunbook | None = None,
    dataset_index: dict | None = None,
) -> None:
    """Save the student, plus any trained acquisition context so the artifact is complete.

    The context projections live inside StudentModel, but the ``MSContextEncoder`` that
    *produces* the context vectors are separate modules — persist them here or a loaded model
    can only make base (context-free) predictions.
    """
    blob: dict = {"config": model.cfg.to_dict(), "state_dict": model.state_dict()}
    if encoder is not None or runbook is not None or dataset_index is not None:
        blob["context"] = {
            "encoder": _encoder_blob(encoder) if encoder is not None else None,
            "runbook": _runbook_blob(runbook) if runbook is not None else None,
            "dataset_index": dataset_index,
        }
    torch.save(blob, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> StudentModel:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = StudentModel(StudentConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_context(path: str | Path, map_location: str = "cpu") -> ContextBundle | None:
    """Load the acquisition context saved with a checkpoint, or ``None`` if it had none."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    ctx = ckpt.get("context")
    if not ctx:
        return None
    encoder = runbook = None
    if ctx.get("encoder"):
        e = ctx["encoder"]
        encoder = MSContextEncoder(
            e["context_dim"],
            instruments=tuple(e.get("instruments", DEFAULT_INSTRUMENTS)),
            detectors=tuple(e.get("detectors", DEFAULT_DETECTORS)),
            fragmentations=tuple(e.get("fragmentations", DEFAULT_FRAGMENTATIONS)),
        )
        encoder.load_state_dict(e["state_dict"])
        encoder.eval()
    if ctx.get("runbook"):
        r = ctx["runbook"]
        runbook = ChromRunbook(r["n_datasets"], r["context_dim"])
        runbook.load_state_dict(r["state_dict"])
        runbook.eval()
    return ContextBundle(encoder=encoder, runbook=runbook, dataset_index=ctx.get("dataset_index"))
