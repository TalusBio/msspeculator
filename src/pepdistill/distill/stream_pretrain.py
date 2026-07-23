"""Online teacher-distill pretrain: enumerate the digests, chunk-label, per-peptide NCE sweep.

Aggressive but not wasteful. Instead of randomly sampling peptides (coupon-collector — needs
~16x the draws to cover a space), it *enumerates* the digests lazily: walk the FASTA, yield
every tryptic peptide and every unspecific (immunopeptidome-like) window, no dedup (~2% repeat
rate isn't worth a proteome-scale seen-set). One pass = full coverage.

Throughput is teacher-bound, and peptdeep amortizes per-call overhead over big batches
(~4k pep/s at 10k vs ~340 at 256). So peptides are labeled in large CHUNKS (prefetcher-style)
and fed to the student as small mini-batches. Each peptide in a chunk gets its OWN collision
energy drawn from a range, so ``ctx_acq`` is conditioned on a genuine per-peptide NCE sweep
(the shared ContextEncoder learns a real CE response). Stays on the Lightning engine: a finite
``IterableDataset`` over ``passes`` enumerations feeds ``DistillModule``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..data.config import DigestConfig
from ..data.sources import (
    enumerate_tryptic_stream,
    enumerate_unspecific_stream,
    precursors_from_sequences,
)
from ..models.context import ContextEncoder
from ..models.student import StudentModel
from .dataset import collate_with_labels
from .lightning import DistillModule


@dataclass
class StreamMix:
    """One digest stream: its FASTA, kind (enumerator), and precursor settings (charge/mods)."""

    name: str
    kind: str  # "unspecific" | "tryptic"
    fasta: str
    cfg: DigestConfig
    min_len: int = 8
    max_len: int = 11


@dataclass
class StreamPretrainCfg:
    mixes: list[StreamMix] = field(default_factory=list)
    nce_range: tuple[float, float] = (20.0, 40.0)
    chunk_size: int = 10000  # peptides per teacher call (throughput knob)
    batch_size: int = 256  # student mini-batch
    passes: int = 1  # full enumerations of the digests
    lr: float = 1e-3
    seed: int = 0
    # Early stop when the student saturates the teacher (MS2 loss plateaus) — avoids burning
    # teacher throughput on a converged model. patience=0 disables it. Patience counts
    # consecutive `check_every`-step windows with < min_delta mean-loss improvement.
    patience: int = 0
    min_delta: float = 1e-3
    check_every: int = 200
    warmup_steps: int = 500


def default_mixes(fasta: str) -> list[StreamMix]:
    """Immunopeptidome-like unspecific windows (charge 1-2, no var mods) + tryptic (charge 2-4)."""
    immuno = DigestConfig(enzyme="unspecific", min_charge=1, max_charge=2, max_variable_mods=0)
    tryptic = DigestConfig()
    return [
        StreamMix("immuno", "unspecific", fasta, immuno, min_len=8, max_len=11),
        StreamMix("tryptic", "tryptic", fasta, tryptic),
    ]


def _peptides(mix: StreamMix, loop: bool):
    if mix.kind == "unspecific":
        return enumerate_unspecific_stream(mix.fasta, mix.min_len, mix.max_len, loop)
    return enumerate_tryptic_stream(mix.fasta, mix.cfg, loop)


class _StreamingDataset(IterableDataset):
    def __init__(self, teacher, cfg: StreamPretrainCfg) -> None:
        self.teacher = teacher
        self.cfg = cfg

    def _round_robin(self, iters):
        """Interleave the mix streams -> (mix_idx, seq) until all are exhausted."""
        done = [False] * len(iters)
        while not all(done):
            for mi, it in enumerate(iters):
                if done[mi]:
                    continue
                try:
                    yield mi, next(it)
                except StopIteration:
                    done[mi] = True

    def _build_precs(self, items, rng):
        """items: [(mix_idx, seq)] -> precursor list aligned to items (per-mix charge/mods)."""
        precs = [None] * len(items)
        by_mix: dict[int, list[int]] = {}
        for idx, (mi, _seq) in enumerate(items):
            by_mix.setdefault(mi, []).append(idx)
        for mi, idxs in by_mix.items():
            built = precursors_from_sequences(
                [items[i][1] for i in idxs], self.cfg.mixes[mi].cfg, rng
            )
            for i, p in zip(idxs, built):
                precs[i] = p
        return precs

    def _label_chunk(self, items, rng):
        precs = self._build_precs(items, rng)
        nces = rng.uniform(*self.cfg.nce_range, size=len(precs))
        labels = self.teacher.predict(precs, nces=nces)
        triples = [(p, lab, float(n)) for p, lab, n in zip(precs, labels, nces) if lab is not None]
        for start in range(0, len(triples), self.cfg.batch_size):
            sub = triples[start : start + self.cfg.batch_size]
            lb = collate_with_labels([p for p, _, _ in sub], [lab for _, lab, _ in sub])
            lb.ce = torch.tensor([n for _, _, n in sub], dtype=torch.float32)
            yield lb

    def __iter__(self):
        rng = np.random.default_rng(self.cfg.seed)
        for _ in range(self.cfg.passes):
            iters = [_peptides(m, loop=False) for m in self.cfg.mixes]
            buf: list = []
            for item in self._round_robin(iters):
                buf.append(item)
                if len(buf) >= self.cfg.chunk_size:
                    yield from self._label_chunk(buf, rng)
                    buf = []
            if buf:
                yield from self._label_chunk(buf, rng)


def _estimate_norm(teacher, cfg: StreamPretrainCfg, n: int = 512):
    """Label a mid-NCE sample to standardize rt/ccs (teacher frame); real train resets it."""
    rng = np.random.default_rng(cfg.seed + 1)
    ds = _StreamingDataset(teacher, cfg)
    iters = [_peptides(m, loop=False) for m in cfg.mixes]
    items = []
    for item in ds._round_robin(iters):
        items.append(item)
        if len(items) >= n:
            break
    precs = ds._build_precs(items, rng)
    teacher.nce = float(np.mean(cfg.nce_range))
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


class _LossPlateauStop(L.Callback):
    """Stop the (single-epoch) stream when the mean MS2 loss over a window stops improving."""

    def __init__(self, patience, min_delta, check_every, warmup, emit) -> None:
        self.patience, self.min_delta, self.check_every, self.warmup, self._emit = (
            patience,
            min_delta,
            check_every,
            warmup,
            emit,
        )
        self.best = float("inf")
        self.bad = 0
        self.buf: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        m = trainer.callback_metrics.get("train_ms2")
        if m is None:
            return
        self.buf.append(float(m))
        step = batch_idx + 1
        if step < self.warmup or step % self.check_every:
            return
        cur = sum(self.buf) / len(self.buf)
        self.buf.clear()
        if self.best - cur > self.min_delta:
            self.best, self.bad = cur, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self._emit(
                    f"[early-stop] ms2 plateaued at {cur:.3f} (best {self.best:.3f}) "
                    f"-> stopping at step {step}"
                )
                trainer.should_stop = True


def fit_stream_pretrain(
    model: StudentModel,
    encoder: ContextEncoder,
    teacher,
    cfg: StreamPretrainCfg,
    *,
    accelerator: str = "cpu",
    log=print,
    log_every: int = 100,
) -> DistillModule:
    """Enumerate-and-chunk online teacher-distill warmup on the shared backbone + CE encoder."""
    L.seed_everything(cfg.seed, verbose=False)
    log("[stream] estimating rt/ccs norm from a teacher sample...")
    model.set_norm(*_estimate_norm(teacher, cfg))
    module = DistillModule(model, lr=cfg.lr, context_encoder=encoder)
    loader = DataLoader(_StreamingDataset(teacher, cfg), batch_size=None)
    callbacks: list[L.Callback] = [_StepLogger(log_every, log)]
    if cfg.patience > 0:
        callbacks.append(
            _LossPlateauStop(cfg.patience, cfg.min_delta, cfg.check_every, cfg.warmup_steps, log)
        )
    trainer = L.Trainer(
        max_epochs=1,  # the dataset is finite (passes enumerations); it drives the length
        accelerator=accelerator,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        callbacks=callbacks,
    )
    trainer.fit(module, loader)
    return module
