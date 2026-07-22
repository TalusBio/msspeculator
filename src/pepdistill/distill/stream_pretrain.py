"""Online (streaming) teacher-distill pretrain with a collision-energy sweep.

Aggressive warmup: instead of caching labels, the teacher scores freshly-sampled peptides
live in the training loop, and each batch draws a random NCE from a range. The batch's NCE is
fed to the shared :class:`ContextEncoder` as ``ctx_acq``, so the student sees a broad CE axis
(not a single teacher condition) and the CE encoder can learn a real response instead of
memorizing a few points.

Peptides come from a weighted mix of streams — e.g. immunopeptidome-like unspecific windows
and tryptic digests of a proteome — so the student learns fragmentation over a wide,
effectively infinite peptide space. Stays on the Lightning engine: an ``IterableDataset``
yields ready-made labeled batches (teacher called inside ``__iter__``) into ``DistillModule``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..data.config import DigestConfig
from ..data.sources import (
    fasta_peptide_stream,
    precursors_from_sequences,
    unspecific_window_stream,
)
from ..models.context import ContextEncoder
from ..models.student import StudentModel
from .dataset import collate_with_labels
from .lightning import DistillModule


@dataclass
class StreamMix:
    """One peptide stream in the pretrain mix: its FASTA, how to sample it, how to build precursors."""

    name: str
    kind: str  # "unspecific" | "tryptic"
    fasta: str
    cfg: DigestConfig  # charges + mods used by precursors_from_sequences (and enzyme for tryptic)
    weight: float = 1.0
    min_len: int = 8
    max_len: int = 11


@dataclass
class StreamPretrainCfg:
    mixes: list[StreamMix] = field(default_factory=list)
    nce_range: tuple[float, float] = (20.0, 40.0)
    total_batches: int = 5000
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0


def default_mixes(fasta: str) -> list[StreamMix]:
    """Immunopeptidome-like unspecific windows (charge 1-2, no var mods) + tryptic (charge 2-4)."""
    immuno = DigestConfig(enzyme="unspecific", min_charge=1, max_charge=2, max_variable_mods=0)
    tryptic = DigestConfig()  # trypsin, fixed CAM, variable Met-ox, charge 2-4 (project defaults)
    return [
        StreamMix("immuno", "unspecific", fasta, immuno, weight=1.0, min_len=8, max_len=11),
        StreamMix("tryptic", "tryptic", fasta, tryptic, weight=1.0),
    ]


class _StreamingDataset(IterableDataset):
    def __init__(self, teacher, cfg: StreamPretrainCfg) -> None:
        self.teacher = teacher
        self.cfg = cfg

    def _samplers(self, rng):
        out = []
        for m in self.cfg.mixes:
            if m.kind == "unspecific":
                it = unspecific_window_stream(m.fasta, rng, m.min_len, m.max_len)
            else:
                it = fasta_peptide_stream(m.fasta, m.cfg, rng)
            out.append(it)
        return out

    def _batch(self, samplers, weights, rng):
        nce = float(rng.uniform(*self.cfg.nce_range))
        which = rng.choice(len(samplers), p=weights)
        mix = self.cfg.mixes[which]
        seqs = [next(samplers[which]) for _ in range(self.cfg.batch_size)]
        precs = precursors_from_sequences(seqs, mix.cfg, rng)
        self.teacher.nce = nce
        labels = self.teacher.predict(precs)
        pairs = [(p, lab) for p, lab in zip(precs, labels) if lab is not None]
        if not pairs:
            return None
        lb = collate_with_labels([p for p, _ in pairs], [lab for _, lab in pairs])
        lb.ce = torch.full((len(pairs),), nce, dtype=torch.float32)
        return lb

    def __iter__(self):
        rng = np.random.default_rng(self.cfg.seed)
        samplers = self._samplers(rng)
        w = np.array([m.weight for m in self.cfg.mixes], dtype=np.float64)
        w /= w.sum()
        made = 0
        while made < self.cfg.total_batches:
            lb = self._batch(samplers, w, rng)
            if lb is not None:
                made += 1
                yield lb


def _estimate_norm(teacher, cfg: StreamPretrainCfg, n: int = 512):
    """Label a mid-NCE sample to standardize rt/ccs (teacher frame); real train resets it."""
    rng = np.random.default_rng(cfg.seed + 1)
    ds = _StreamingDataset(teacher, cfg)
    samplers = ds._samplers(rng)
    w = np.array([m.weight for m in cfg.mixes], dtype=np.float64)
    w /= w.sum()
    teacher.nce = float(np.mean(cfg.nce_range))
    seqs, mixidx = [], int(np.argmax(w))
    for _ in range(n):
        seqs.append(next(samplers[mixidx]))
    precs = precursors_from_sequences(seqs, cfg.mixes[mixidx].cfg, rng)
    labels = [lab for lab in teacher.predict(precs) if lab is not None]
    rt = np.array([lab.rt for lab in labels], dtype=np.float64)
    ccs = np.array([lab.ccs for lab in labels], dtype=np.float64)
    return float(rt.mean()), float(rt.std() or 1.0), float(ccs.mean()), float(ccs.std() or 1.0)


class _StepLogger(L.Callback):
    # NB: store the emit fn as `_emit`, NOT `log` — Lightning treats a callback's `.log`
    # as its own logging hook, which would shadow our callable.
    def __init__(self, every: int, emit) -> None:
        self.every, self._emit = every, emit

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.every == 0:
            m = trainer.callback_metrics
            self._emit(
                f"  step {batch_idx + 1}: ms2={float(m.get('train_ms2', float('nan'))):.3f} "
                f"total={float(m.get('train_total', float('nan'))):.3f}"
            )


def fit_stream_pretrain(
    model: StudentModel,
    encoder: ContextEncoder,
    teacher,
    cfg: StreamPretrainCfg,
    *,
    accelerator: str = "cpu",
    log=print,
    log_every: int = 250,
) -> DistillModule:
    """Online NCE-sweep teacher-distill warmup on the shared backbone + CE encoder."""
    L.seed_everything(cfg.seed, verbose=False)
    log("[stream] estimating rt/ccs norm from a teacher sample...")
    model.set_norm(*_estimate_norm(teacher, cfg))
    module = DistillModule(model, lr=cfg.lr, context_encoder=encoder)
    loader = DataLoader(_StreamingDataset(teacher, cfg), batch_size=None)
    trainer = L.Trainer(
        max_steps=cfg.total_batches,
        max_epochs=1,
        accelerator=accelerator,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        callbacks=[_StepLogger(log_every, log)],
    )
    trainer.fit(module, loader)
    return module
