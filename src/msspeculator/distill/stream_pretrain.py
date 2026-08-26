"""Online teacher-distill pretrain: enumerate the digests, chunk-label, per-peptide NCE sweep.

Aggressive but not wasteful. Instead of randomly sampling peptides (coupon-collector, needs
~16x the draws to cover a space), it *enumerates* the digests lazily: walk the FASTA, yield
every tryptic peptide and every unspecific (immunopeptidome-like) window, no dedup (~2% repeat
rate isn't worth a proteome-scale seen-set). One pass = full coverage.

Throughput is teacher-bound, and peptdeep amortizes per-call overhead over big batches
(~4k pep/s at 10k vs ~340 at 256). So peptides are labeled in large CHUNKS (prefetcher-style)
and fed to the student as small mini-batches. Each peptide in a chunk gets its OWN collision
energy drawn from a range, so ``ms_context`` is conditioned on a genuine per-peptide NCE sweep
(the shared MSContextEncoder learns a real energy response). Stays on the Lightning engine: a
finite ``IterableDataset`` over ``passes`` enumerations feeds ``DistillModule``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, IterableDataset

from ..data.config import DigestConfig
from ..data.precursors import Precursor
from ..data.sources import (
    enumerate_tryptic_stream,
    enumerate_unspecific_stream,
    precursors_from_sequences,
)
from ..models.context import MSContextEncoder
from ..models.registry import save_checkpoint
from ..models.student import StudentModel
from .dataset import DistillDataset, MSFactors, collate_with_labels
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
    # Emit EVERY charge in each mix's range per peptide, consecutively, rather than sampling
    # one. Charge is factored out of the trunk and re-enters only at the MS2/CCS heads, so those
    # heads learn it from the contrast between charges of the SAME peptide, a contrast sampling
    # never produces, since each peptide appears at one charge per pass. Costs len(charges)x the
    # precursors, and therefore that much teacher time, per peptide.
    all_charge_states: bool = True
    lr: float = 1e-3
    seed: int = 0
    # The teacher's fixed acquisition (peptdeep Orbitrap Lumos, FTMS detector, HCD) -> ms_context
    # factors; only the energy varies (per-peptide NCE sweep).
    instrument: str = "Lumos"
    detector: str = "FTMS"
    fragmentation: str = "HCD"
    # Early stop when the student saturates the teacher (MS2 loss plateaus), avoids burning
    # teacher throughput on a converged model. patience=0 disables it. Patience counts
    # consecutive `check_every`-step windows with < min_delta mean-loss improvement.
    patience: int = 0
    min_delta: float = 1e-3
    check_every: int = 200
    warmup_steps: int = 500
    # OneCycle is enabled by default for the streaming warmup. The stream is an iterable with no
    # cheap, reliable length (teacher filtering changes the yielded batch count), so the cycle
    # length is explicit and should be set to the expected optimizer-step count for a run.
    onecycle_max_lr: float | None = 1e-3
    onecycle_total_steps: int | None = 2500
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    # Ties mass_enc onto a stop-gradiented comp_enc (see losses.mod_align_loss); this regime
    # also drives a DistillModule, so it gets the same knob as fit_distill/fit_realspeclib.
    mod_align_weight: float = 1.0
    residue_substitution_probability: float = 0.0

    def __post_init__(self) -> None:
        if (self.onecycle_max_lr is None) != (self.onecycle_total_steps is None):
            raise ValueError("onecycle_max_lr and onecycle_total_steps must be provided together")
        if self.onecycle_max_lr is not None:
            if self.onecycle_max_lr <= 0:
                raise ValueError("onecycle_max_lr must be positive")
            if self.onecycle_total_steps < 1:
                raise ValueError("onecycle_total_steps must be positive")
            if not 0.0 <= self.onecycle_pct_start <= 1.0:
                raise ValueError("onecycle_pct_start must be between 0 and 1")
            if self.onecycle_div_factor <= 0 or self.onecycle_final_div_factor <= 0:
                raise ValueError(
                    "onecycle_div_factor and onecycle_final_div_factor must be positive"
                )
        if not 0.0 <= self.residue_substitution_probability <= 1.0:
            raise ValueError("residue_substitution_probability must be between 0 and 1")


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
    def __init__(self, teacher, encoder: MSContextEncoder, cfg: StreamPretrainCfg) -> None:
        self.teacher = teacher
        self.encoder = encoder
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
        """``items: [(mix_idx, seq)]`` -> flat precursor list, grouped by mix.

        Not aligned 1:1 to ``items``: with ``all_charge_states`` a sequence expands to one
        precursor per charge. Nothing downstream needs the item correspondence, the teacher
        takes a flat list and labels are zipped back positionally, and keeping a peptide's
        charge states consecutive is what puts them in the same mini-batch.
        """
        by_mix: dict[int, list[str]] = {}
        for mi, seq in items:
            by_mix.setdefault(mi, []).append(seq)
        precs: list[Precursor] = []
        for mi, seqs in by_mix.items():
            precs.extend(
                precursors_from_sequences(
                    seqs,
                    self.cfg.mixes[mi].cfg,
                    rng,
                    all_charge_states=self.cfg.all_charge_states,
                )
            )
        return precs

    def _label_chunk(self, items, rng):
        precs = self._build_precs(items, rng)
        nces = rng.uniform(*self.cfg.nce_range, size=len(precs))
        labels = self.teacher.predict(precs, nces=nces)
        triples = [(p, lab, float(n)) for p, lab, n in zip(precs, labels, nces) if lab is not None]
        inst_id = self.encoder.instrument_id(self.cfg.instrument)
        det_id = self.encoder.detector_id(self.cfg.detector)
        frag_id = self.encoder.fragmentation_id(self.cfg.fragmentation)
        for start in range(0, len(triples), self.cfg.batch_size):
            sub = triples[start : start + self.cfg.batch_size]
            lb = collate_with_labels([p for p, _, _ in sub], [lab for _, lab, _ in sub])
            n = len(sub)
            lb.ms_factors = MSFactors(
                instrument_id=torch.full((n,), inst_id, dtype=torch.long),
                detector_id=torch.full((n,), det_id, dtype=torch.long),
                fragmentation_id=torch.full((n,), frag_id, dtype=torch.long),
                energy=torch.tensor([nce for _, _, nce in sub], dtype=torch.float32),
            )
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


def _estimate_norm(teacher, encoder: MSContextEncoder, cfg: StreamPretrainCfg, n: int = 512):
    """Label a mid-NCE sample to standardize rt/ccs (teacher frame); real train resets it."""
    rng = np.random.default_rng(cfg.seed + 1)
    ds = _StreamingDataset(teacher, encoder, cfg)
    iters = [_peptides(m, loop=False) for m in cfg.mixes]
    items = []
    for item in ds._round_robin(iters):
        items.append(item)
        if len(items) >= n:
            break
    precs = ds._build_precs(items, rng)
    teacher.nce = float(np.mean(cfg.nce_range))
    pairs = [(p, lab) for p, lab in zip(precs, teacher.predict(precs)) if lab is not None]
    return DistillDataset([p for p, _ in pairs], [lab for _, lab in pairs]).rt_ccs_stats()


class _StepLogger(L.Callback):
    # NB: store the emit fn as `_emit`, NOT `log`, Lightning treats a callback's `.log`
    # as its own logging hook, which would shadow our callable.
    def __init__(self, every: int, emit) -> None:
        self.every, self._emit = every, emit

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.every == 0:
            m = trainer.callback_metrics
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            self._emit(
                f"  step {batch_idx + 1}: ms2={float(m.get('train_ms2', float('nan'))):.3f} "
                f"total={float(m.get('train_total', float('nan'))):.3f} lr={lr:.6g}"
            )


class _LossPlateauStop(L.Callback):
    """Stop the (single-epoch) stream when the mean MS2 loss over a window stops improving."""

    def __init__(self, patience, min_delta, check_every, warmup, emit) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.check_every = check_every
        self.warmup = warmup
        self._emit = emit
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


class _StreamCheckpoint(L.Callback):
    """Persist an inference-ready warm-start snapshot during the one-epoch stream."""

    def __init__(
        self,
        every: int,
        path: str | Path,
        mirror: Callable[[Path], str] | None,
        emit: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.every = every
        self.path = Path(path)
        self.mirror = mirror
        self.emit = emit

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = batch_idx + 1
        if step % self.every:
            return
        save_checkpoint(
            pl_module.model,
            self.path,
            encoder=pl_module.context_encoder,
        )
        if self.mirror is not None:
            self.mirror(self.path)
        self.emit(f"[pretrain] checkpointed step {step} -> {self.path}")


def fit_stream_pretrain(
    model: StudentModel,
    encoder: MSContextEncoder,
    teacher,
    cfg: StreamPretrainCfg,
    *,
    accelerator: str = "cpu",
    log=print,
    log_every: int = 100,
    checkpoint_every: int = 0,
    checkpoint_path=None,
    artifact_mirror=None,
    logger=False,
    callbacks: list[L.Callback] | None = None,
) -> DistillModule:
    """Enumerate-and-chunk online teacher-distill warmup on the shared backbone + MS context
    encoder."""
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if checkpoint_every > 0 and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when checkpoint_every is positive")
    L.seed_everything(cfg.seed, verbose=False)
    # CCS only. The teacher is the ONLY source of CCS, so its scale must be established here
    # (teacher CCS is ~543+-122; under an identity norm that target would dominate the loss).
    #
    # RT is deliberately NOT established from the teacher. iRT is the canonical RT frame, and
    # the teacher predicts RT in its own normalized 0-1 space (~0.49+-0.27), a different
    # quantity. Establishing the global affine from it forces every later real dataset to be
    # standardized through the wrong units: measured, that drove train_irt from 0.031 to 379.8
    # and val_spectral_angle from 0.617 to 0.476. Teacher RT under the identity norm is already
    # well-conditioned, so pretrain trains fine without it and the first real dataset
    # establishes the frame in iRT units.
    log("[stream] estimating ccs norm from a teacher sample (RT frame comes from real iRT)...")
    _, _, ccs_mean, ccs_std = _estimate_norm(teacher, encoder, cfg)
    model.set_norm(ccs_mean=ccs_mean, ccs_std=ccs_std)
    module = DistillModule(
        model,
        lr=cfg.lr,
        context_encoder=encoder,
        mod_align_weight=cfg.mod_align_weight,
        onecycle_max_lr=cfg.onecycle_max_lr,
        onecycle_total_steps=cfg.onecycle_total_steps,
        onecycle_pct_start=cfg.onecycle_pct_start,
        onecycle_div_factor=cfg.onecycle_div_factor,
        onecycle_final_div_factor=cfg.onecycle_final_div_factor,
        residue_substitution_probability=cfg.residue_substitution_probability,
    )
    loader = DataLoader(_StreamingDataset(teacher, encoder, cfg), batch_size=None)
    callbacks = [*(callbacks or ()), _StepLogger(log_every, log)]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if checkpoint_every > 0:
        callbacks.append(_StreamCheckpoint(checkpoint_every, checkpoint_path, artifact_mirror, log))
    if cfg.patience > 0:
        callbacks.append(
            _LossPlateauStop(cfg.patience, cfg.min_delta, cfg.check_every, cfg.warmup_steps, log)
        )
    trainer = L.Trainer(
        max_epochs=1,  # the dataset is finite (passes enumerations); it drives the length
        accelerator=accelerator,
        enable_checkpointing=False,
        logger=logger,
        enable_progress_bar=False,
        limit_val_batches=0,
        callbacks=callbacks,
    )
    trainer.fit(module, loader)
    return module
