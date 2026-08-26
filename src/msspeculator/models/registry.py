"""Named student presets and checkpoint (de)serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fsspec
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
    # `setups` rides with `state_dict` for the same reason `names` does below: a fitted
    # acquisition row means nothing without the setup it was fitted for.
    return {
        "context_dim": enc.context_dim,
        "instruments": enc.instruments,
        "detectors": enc.detectors,
        "fragmentations": enc.fragmentations,
        "setups": enc.setups,
        "state_dict": enc.state_dict(),
    }


def _runbook_blob(book: ChromRunbook) -> dict:
    # `names` rides with `state_dict` deliberately: a row is meaningless without the dataset it
    # was trained for, and storing the two apart is what let a growing corpus renumber the names
    # while the weights stayed put.
    return {
        "n_datasets": book.n_datasets,
        "context_dim": book.context_dim,
        "names": book.names,
        "state_dict": book.state_dict(),
    }


def save_checkpoint(
    model: StudentModel,
    path: str | Path,
    *,
    encoder: MSContextEncoder | None = None,
    runbook: ChromRunbook | None = None,
    dataset_index: dict | None = None,
    training_metadata: dict | None = None,
) -> None:
    """Save the student, plus any trained acquisition context so the artifact is complete.

    The context projections live inside StudentModel, but the ``MSContextEncoder`` that
    *produces* the context vectors are separate modules, persist them here or a loaded model
    can only make base (context-free) predictions.
    """
    blob: dict = {"config": model.cfg.to_dict(), "state_dict": model.state_dict()}
    if encoder is not None or runbook is not None or dataset_index is not None:
        if runbook is not None and dataset_index and not runbook.names:
            # A caller that still threads the index separately: adopt it into the book so the
            # rows and their names are stored together from here on.
            runbook.adopt_names(dataset_index)
        blob["context"] = {
            "encoder": _encoder_blob(encoder) if encoder is not None else None,
            "runbook": _runbook_blob(runbook) if runbook is not None else None,
            "dataset_index": runbook.names
            if runbook is not None and runbook.names
            else dataset_index,
        }
    if training_metadata is not None:
        # Plain scalar/container metadata remains inspectable with torch.load and is ignored by
        # inference loaders. It intentionally does not contain optimizer state.
        blob["training"] = training_metadata
    torch.save(blob, path)


def _load_checkpoint_blob(path: str | Path, map_location: str = "cpu") -> dict:
    if "://" in str(path):
        with fsspec.open(str(path), "rb") as stream:
            return torch.load(stream, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> StudentModel:
    ckpt = _load_checkpoint_blob(path, map_location)
    model = StudentModel(StudentConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_context(path: str | Path, map_location: str = "cpu") -> ContextBundle | None:
    """Load the acquisition context saved with a checkpoint, or ``None`` if it had none."""
    ckpt = _load_checkpoint_blob(path, map_location)
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
            setups=e.get("setups"),
        )
        # A checkpoint written before named setups existed has no `setup_emb`. Its absence is
        # not ambiguous; there were none to save; so the zero-init table stands, which is
        # exactly the neutral term an unnamed source gets. Filled in by name rather than with a
        # non-strict load, which would swallow a genuinely mismatched checkpoint too.
        state = dict(e["state_dict"])
        state.setdefault("setup_emb.weight", encoder.setup_emb.weight.detach().clone())
        encoder.load_state_dict(state)
        encoder.eval()
    if ctx.get("runbook"):
        r = ctx["runbook"]
        # Checkpoints written before the book owned its index keep the map beside it; read either,
        # so an older checkpoint still resumes with its rows correctly named.
        runbook = ChromRunbook(
            r["n_datasets"],
            r["context_dim"],
            names=r.get("names") or ctx.get("dataset_index"),
        )
        runbook.load_state_dict(r["state_dict"])
        runbook.eval()
    index = runbook.names if runbook is not None and runbook.names else ctx.get("dataset_index")
    return ContextBundle(encoder=encoder, runbook=runbook, dataset_index=index)
