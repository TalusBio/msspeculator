"""Real-speclib training regime with per-source acquisition context.

A second Lightning regime over the SAME StudentModel backbone (see lightning.py). It trains
on real experimental spectra (e.g. PROSPECT) and gradient-descends a per-source context via
a :class:`ContextBook`, keyed by raw_file:

- MS2 uses ``ctx_acq`` (acquisition: analyzer/fragmentation/collision energy — per raw_file).
- RT is dual-target: the context-free base head is pinned to iRT, and the ctx_lc-conditioned
  head is fit to the raw retention_time. So ctx_lc learns ONLY each run's LC deviation, and
  base RT stays the run-independent peptide property. (iRT is context-free.)
- CCS is unsupervised here (real DDA has none) — left to teacher distillation.

Backbone + context vectors are optimized together. To fine-tune only the context (adapt to a
new run), freeze the model and step only the book.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.precursors import Precursor
from ..eval import best_per_key, ms2_intensity, precursor_key
from ..models.context import ContextBook, ContextEncoder
from ..models.student import StudentModel
from ..teacher.base import PrecursorLabels
from .dataset import BatchIterable, LabeledBatch, collate_with_labels, iter_batch_indices
from .lightning import build_trainer
from .losses import ms2_cosine_loss, spectral_angle

if TYPE_CHECKING:
    from ..data.prospect import RealLabels

_T = TypeVar("_T")


@dataclass(slots=True)
class RealExample:
    """One real observation: the peptide, its teacher/experimental labels, and the acquisition
    factors that condition the two context vectors. ``ce``/``analyzer_id``/``frag_id`` drive
    ``ctx_acq`` (MS2/CCS); ``source_id`` drives ``ctx_lc`` (RT / chromatography)."""

    precursor: Precursor
    label: PrecursorLabels  # ms2 + rt(=iRT) + ccs(NaN for real DDA)
    raw_rt: float  # run-dependent retention time (the ctx_lc target)
    source_id: int  # raw_file index -> ctx_lc
    ce: float  # collision energy (absolute NCE) -> ctx_acq
    analyzer_id: int  # mass-analyzer factor -> ctx_acq
    frag_id: int  # fragmentation factor -> ctx_acq


@dataclass(slots=True)
class RealBatch:
    base: LabeledBatch  # inputs + ms2_target + rt_target(=iRT) + ccs_target(NaN)
    raw_rt: torch.Tensor  # (B,) run-dependent retention time
    source_id: torch.Tensor  # (B,) long, raw_file index (for ctx_lc / chromatography)
    ce: torch.Tensor  # (B,) collision energy (absolute NCE) -> ctx_acq
    analyzer_id: torch.Tensor  # (B,) long, mass-analyzer factor -> ctx_acq
    frag_id: torch.Tensor  # (B,) long, fragmentation factor -> ctx_acq

    def to(self, device: torch.device | str) -> RealBatch:
        return RealBatch(
            base=self.base.to(device),
            raw_rt=self.raw_rt.to(device),
            source_id=self.source_id.to(device),
            ce=self.ce.to(device),
            analyzer_id=self.analyzer_id.to(device),
            frag_id=self.frag_id.to(device),
        )


class RealSpeclibDataset:
    """A list of :class:`RealExample`; columns the collate/context path needs are cached as
    arrays so :meth:`batches` only pays per-batch indexing, not per-batch attribute walks."""

    def __init__(self, examples: list[RealExample]) -> None:
        self.examples = examples
        self.raw_rt = np.array([e.raw_rt for e in examples], dtype=np.float32)
        self.source_id = np.array([e.source_id for e in examples], dtype=np.int64)
        self.ce = np.array([e.ce for e in examples], dtype=np.float32)
        self.analyzer_id = np.array([e.analyzer_id for e in examples], dtype=np.int64)
        self.frag_id = np.array([e.frag_id for e in examples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.examples)

    def batches(
        self, batch_size: int, shuffle: bool, generator: torch.Generator
    ) -> Iterator[RealBatch]:
        for idx in iter_batch_indices(len(self), batch_size, shuffle, generator):
            base = collate_with_labels(
                [self.examples[i].precursor for i in idx],
                [self.examples[i].label for i in idx],
            )
            yield RealBatch(
                base=base,
                raw_rt=torch.from_numpy(self.raw_rt[idx]),
                source_id=torch.from_numpy(self.source_id[idx]),
                ce=torch.from_numpy(self.ce[idx]),
                analyzer_id=torch.from_numpy(self.analyzer_id[idx]),
                frag_id=torch.from_numpy(self.frag_id[idx]),
            )


class RealSpeclibModule(L.LightningModule):
    """Fit the shared backbone on real spectra while learning per-run context: MS2 on
    ``ctx_acq`` (from acquisition factors), the dual RT targets on the base head (iRT) and the
    ``ctx_lc``-conditioned head (raw RT). Set ``freeze_backbone`` to adapt to a new run by
    training only the context (ctx_lc book + ctx_acq encoder) — the PEFT path from the module
    docstring — leaving the backbone fixed."""

    def __init__(
        self,
        model: StudentModel,
        book: ContextBook,
        encoder: ContextEncoder,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),  # ms2, irt, raw_rt
        source_index: dict[str, int] | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.book = book  # ctx_lc (chromatography, per raw_file)
        self.encoder = encoder  # ctx_acq (MS2/CCS, from acquisition factors)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        # raw_file -> ctx_lc row; needed to address a trained run's context at inference.
        self.source_index = source_index
        if freeze_backbone:  # PEFT: fit only the context vectors, backbone stays fixed.
            self.model.requires_grad_(False)

    def transfer_batch_to_device(
        self, batch: RealBatch, device: torch.device | str, dataloader_idx: int
    ) -> RealBatch:
        return batch.to(device)

    def _forward(self, rb: RealBatch) -> dict[str, torch.Tensor]:
        ctx_acq = self.encoder(rb.ce, rb.analyzer_id, rb.frag_id)  # MS2/CCS acquisition factors
        ctx_lc = self.book.lc(rb.source_id)  # RT context per run (chromatography)
        return self.model.forward_context(rb.base.inputs, ctx_acq=ctx_acq, ctx_lc=ctx_lc)

    def training_step(self, rb: RealBatch, batch_idx: int) -> torch.Tensor:
        w_ms2, w_irt, w_raw = self.loss_weights
        out = self._forward(rb)
        lb = rb.base
        loss_ms2 = ms2_cosine_loss(out["ms2"], lb.ms2_target, lb.inputs.frag_mask)
        # rt_base (context-free) -> iRT; the ctx_lc-conditioned rt -> this run's raw RT.
        loss_irt = torch.nn.functional.mse_loss(
            out["rt_base"], self.model.standardize_rt(lb.rt_target)
        )
        loss_raw = torch.nn.functional.mse_loss(out["rt"], self.model.standardize_rt(rb.raw_rt))
        self.log_dict(
            {"train_ms2": loss_ms2, "train_irt": loss_irt, "train_rawrt": loss_raw},
            prog_bar=False,
            batch_size=rb.raw_rt.shape[0],
        )
        return w_ms2 * loss_ms2 + w_irt * loss_irt + w_raw * loss_raw

    def validation_step(self, rb: RealBatch, batch_idx: int) -> None:
        out = self._forward(rb)
        lb = rb.base
        sa = spectral_angle(out["ms2"], lb.ms2_target, lb.inputs.frag_mask).mean()
        irt_mae = (self.model.unstandardize_rt(out["rt_base"]) - lb.rt_target).abs().mean()
        rawrt_mae = (self.model.unstandardize_rt(out["rt"]) - rb.raw_rt).abs().mean()
        self.log_dict(
            {"val_spectral_angle": sa, "val_irt_mae": irt_mae, "val_rawrt_mae": rawrt_mae},
            prog_bar=True,
            batch_size=rb.raw_rt.shape[0],
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Optimize every trainable parameter across the registered submodules (model + book +
        # encoder); with freeze_backbone the model's are already requires_grad=False, so this
        # narrows to the context vectors without listing them by hand.
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


def _build_examples(
    real: RealLabels, encoder: ContextEncoder, source_index: dict[str, int]
) -> list[RealExample]:
    """Turn ``RealLabels`` (parallel columns) into per-example records, resolving each run's
    acquisition factors ONCE (few runs, many examples) and reusing them across its examples."""

    def acq(name: str, key: str, default: _T) -> _T:
        return real.acquisition.get(name, {}).get(key, default)

    def ce(name: str) -> float:  # missing CE -> encoder center (its zero-point) -> base ctx_acq
        v = float(acq(name, "collision_energy", encoder.ce_center))
        return encoder.ce_center if math.isnan(v) else v

    ce_of = {n: ce(n) for n in source_index}
    ana_of = {n: encoder.analyzer_id(acq(n, "mass_analyzer", "")) for n in source_index}
    frag_of = {n: encoder.frag_id(acq(n, "fragmentation", "")) for n in source_index}
    return [
        RealExample(
            precursor=p,
            label=lab,
            raw_rt=float(rrt),
            source_id=source_index[s],
            ce=ce_of[s],
            analyzer_id=ana_of[s],
            frag_id=frag_of[s],
        )
        for p, lab, rrt, s in zip(real.precursors, real.labels, real.raw_rt, real.source_ids)
    ]


def _dedupe_val(examples: list[RealExample], dataset_name: str | None) -> list[RealExample]:
    """Keep one best-quality example per (dataset, modified_sequence, charge) so abundant
    peptides don't dominate the val metric. Train keeps every observation."""
    keys = [precursor_key(e.precursor, dataset_name) for e in examples]
    quality = [ms2_intensity(e.label) for e in examples]
    return [examples[i] for i in best_per_key(quality, keys)]


def fit_realspeclib(
    model: StudentModel,
    real: RealLabels,
    *,
    context_dim: int | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    grad_clip: float = 1.0,
    seed: int = 0,
    accelerator: str = "auto",
    dataset_name: str | None = None,
    encoder: ContextEncoder | None = None,
    book: ContextBook | None = None,
    freeze_backbone: bool = False,
    **trainer_kwargs,
) -> RealSpeclibModule:
    """Train on real spectra with per-run context. ``ctx_acq`` comes from the acquisition
    factors via a :class:`ContextEncoder`; ``ctx_lc`` from a per-raw_file :class:`ContextBook`.
    Pass an existing ``encoder``/``book`` to continue a curriculum (share them with the warmup);
    their ``context_dim`` must match the model's. Set ``freeze_backbone`` to adapt to a run by
    fitting only the context (PEFT). ``dataset_name`` is the val dedup key's dataset label.
    Returns the module (``.model``/``.encoder``/``.book`` trained; ``.source_index`` maps
    raw_file -> ctx_lc row)."""
    L.seed_everything(seed, verbose=False)
    cdim = context_dim or model.cfg.context_dim
    if cdim <= 0:
        raise ValueError(f"context_dim must be positive to condition on acquisition, got {cdim}")
    if encoder is not None and encoder.proj.out_features != cdim:
        raise ValueError(f"encoder context_dim {encoder.proj.out_features} != model's {cdim}")
    if book is not None and book.lc.embedding_dim != cdim:
        raise ValueError(f"book context_dim {book.lc.embedding_dim} != model's {cdim}")
    encoder = encoder or ContextEncoder(context_dim=cdim)

    # raw_file -> ctx_lc row. ctx_acq comes from the encoder, so the book only holds ctx_lc
    # (one row per run); its acq table is unused here -> size it to 1, not per-raw_file.
    source_index = {name: i for i, name in enumerate(sorted(set(real.source_ids)))}
    book = book or ContextBook(n_acq=1, n_lc=len(source_index), context_dim=cdim)

    examples = _build_examples(real, encoder, source_index)
    train = [e for e in examples if e.precursor.split != "val"]
    val = [e for e in examples if e.precursor.split == "val"]

    # Normalize the RT heads on TRAIN iRT (CCS is unsupervised here -> identity norm).
    irt_train = np.array([e.label.rt for e in train], dtype=np.float64)
    model.set_norm(float(irt_train.mean()), float(irt_train.std() or 1.0), 0.0, 1.0)

    module = RealSpeclibModule(
        model,
        book,
        encoder,
        lr=lr,
        weight_decay=weight_decay,
        loss_weights=loss_weights,
        source_index=source_index,
        freeze_backbone=freeze_backbone,
    )
    train_ds = RealSpeclibDataset(train)
    val_ds = RealSpeclibDataset(_dedupe_val(val, dataset_name)) if val else None

    def loader(ds: RealSpeclibDataset | None, shuffle: bool) -> DataLoader | None:
        if ds is None:
            return None
        return DataLoader(BatchIterable(ds, batch_size, shuffle, seed), batch_size=None)

    trainer = build_trainer(epochs, accelerator, grad_clip, **trainer_kwargs)
    trainer.fit(module, loader(train_ds, True), loader(val_ds, False))
    return module
