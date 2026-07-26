"""Named student presets and checkpoint (de)serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .context import DEFAULT_ANALYZERS, DEFAULT_FRAGMENTATIONS, ContextBook, ContextEncoder
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


@dataclass
class ContextBundle:
    """Trained acquisition context that rides alongside a checkpoint. Either half may be
    ``None`` (e.g. stream pretrain saves an encoder but no per-run book)."""

    encoder: ContextEncoder | None = None
    book: ContextBook | None = None
    source_index: dict | None = None  # raw_file -> ctx_lc row (matches book.lc)


def _encoder_blob(enc: ContextEncoder) -> dict:
    return {
        "context_dim": enc.proj.out_features,
        "ce_center": enc.ce_center,
        "ce_scale": enc.ce_scale,
        "analyzers": enc.analyzers,
        "fragmentations": enc.fragmentations,
        "state_dict": enc.state_dict(),
    }


def _book_blob(book: ContextBook) -> dict:
    return {
        "n_acq": book.acq.num_embeddings,
        "n_lc": book.lc.num_embeddings,
        "context_dim": book.lc.embedding_dim,
        "state_dict": book.state_dict(),
    }


def save_checkpoint(
    model: StudentModel,
    path: str | Path,
    *,
    encoder: ContextEncoder | None = None,
    book: ContextBook | None = None,
    source_index: dict | None = None,
) -> None:
    """Save the student, plus any trained acquisition context so the artifact is complete.

    The context projections live inside StudentModel, but the CE ``ContextEncoder`` and the
    per-run ``ContextBook`` that *produce* the context vectors are separate modules — persist
    them here or a loaded model can only make base (context-free) predictions.
    """
    blob: dict = {"config": model.cfg.to_dict(), "state_dict": model.state_dict()}
    if encoder is not None or book is not None or source_index is not None:
        blob["context"] = {
            "encoder": _encoder_blob(encoder) if encoder is not None else None,
            "book": _book_blob(book) if book is not None else None,
            "source_index": source_index,
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
    encoder = book = None
    if ctx.get("encoder"):
        e = ctx["encoder"]
        encoder = ContextEncoder(
            e["context_dim"],
            e["ce_center"],
            e["ce_scale"],
            analyzers=tuple(e.get("analyzers", DEFAULT_ANALYZERS)),
            fragmentations=tuple(e.get("fragmentations", DEFAULT_FRAGMENTATIONS)),
        )
        encoder.load_state_dict(e["state_dict"])
        encoder.eval()
    if ctx.get("book"):
        b = ctx["book"]
        book = ContextBook(b["n_acq"], b["n_lc"], b["context_dim"])
        book.load_state_dict(b["state_dict"])
        book.eval()
    return ContextBundle(encoder=encoder, book=book, source_index=ctx.get("source_index"))
