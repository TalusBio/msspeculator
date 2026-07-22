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

from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from ..eval import best_examples
from ..models.context import ContextBook
from ..models.student import StudentModel
from .dataset import LabeledBatch, collate_with_labels
from .losses import ms2_cosine_loss


@dataclass(slots=True)
class RealBatch:
    base: LabeledBatch  # inputs + ms2_target + rt_target(=iRT) + ccs_target(NaN)
    raw_rt: torch.Tensor  # (B,) run-dependent retention time
    source_id: torch.Tensor  # (B,) long, raw_file index

    def to(self, device) -> "RealBatch":
        return RealBatch(self.base.to(device), self.raw_rt.to(device), self.source_id.to(device))


class RealSpeclibDataset:
    """Real examples + per-example raw RT and integer source id (raw_file)."""

    def __init__(self, precursors, labels, raw_rt, source_id) -> None:
        self.precursors = precursors
        self.labels = labels
        self.raw_rt = np.asarray(raw_rt, dtype=np.float32)
        self.source_id = np.asarray(source_id, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.precursors)

    def batches(self, batch_size: int, shuffle: bool, generator: torch.Generator):
        n = len(self)
        order = torch.randperm(n, generator=generator).tolist() if shuffle else list(range(n))
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            base = collate_with_labels(
                [self.precursors[i] for i in idx], [self.labels[i] for i in idx]
            )
            yield RealBatch(
                base,
                torch.from_numpy(self.raw_rt[idx]),
                torch.from_numpy(self.source_id[idx]),
            )


class RealSpeclibModule(L.LightningModule):
    def __init__(
        self,
        model: StudentModel,
        book: ContextBook,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),  # ms2, irt, raw_rt
    ) -> None:
        super().__init__()
        self.model = model
        self.book = book
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights

    def transfer_batch_to_device(self, batch: RealBatch, device, dataloader_idx: int):
        return batch.to(device)

    def _forward(self, rb: RealBatch):
        ctx_acq, ctx_lc = self.book(rb.source_id, rb.source_id)
        return self.model.forward_context(rb.base.inputs, ctx_acq=ctx_acq, ctx_lc=ctx_lc)

    def _std(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.model.rt_mean) / self.model.rt_std

    def training_step(self, rb: RealBatch, batch_idx: int) -> torch.Tensor:
        w_ms2, w_irt, w_raw = self.loss_weights
        out = self._forward(rb)
        lb = rb.base
        loss_ms2 = ms2_cosine_loss(out["ms2"], lb.ms2_target, lb.inputs.frag_mask)
        loss_irt = torch.nn.functional.mse_loss(out["rt_base"], self._std(lb.rt_target))
        loss_raw = torch.nn.functional.mse_loss(out["rt"], self._std(rb.raw_rt))
        total = w_ms2 * loss_ms2 + w_irt * loss_irt + w_raw * loss_raw
        self.log_dict(
            {"train_ms2": loss_ms2, "train_irt": loss_irt, "train_rawrt": loss_raw},
            prog_bar=False,
        )
        return total

    @torch.no_grad()
    def validation_step(self, rb: RealBatch, batch_idx: int) -> None:
        from .losses import spectral_angle

        out = self._forward(rb)
        lb = rb.base
        rt_mean, rt_std = self.model.rt_mean, self.model.rt_std
        sa = spectral_angle(out["ms2"], lb.ms2_target, lb.inputs.frag_mask).mean()
        irt_mae = ((out["rt_base"] * rt_std + rt_mean) - lb.rt_target).abs().mean()
        rawrt_mae = ((out["rt"] * rt_std + rt_mean) - rb.raw_rt).abs().mean()
        self.log_dict(
            {"val_spectral_angle": sa, "val_irt_mae": irt_mae, "val_rawrt_mae": rawrt_mae},
            prog_bar=True,
            batch_size=rb.raw_rt.shape[0],
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.parameters()) + list(self.book.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


class _RealIterable(IterableDataset):
    def __init__(self, ds: RealSpeclibDataset, batch_size: int, shuffle: bool, seed: int) -> None:
        self.ds, self.batch_size, self.shuffle, self.seed, self._epoch = ds, batch_size, shuffle, seed, 0

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        yield from self.ds.batches(self.batch_size, self.shuffle, gen)


def fit_realspeclib(
    model: StudentModel,
    real,  # prospect.RealLabels
    *,
    context_dim: int | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 0,
    accelerator: str = "auto",
    dataset: str | None = None,
    **trainer_kwargs,
) -> RealSpeclibModule:
    """Train on real spectra with per-raw_file context. Returns the module (``.model`` and
    ``.book`` are the trained backbone and context table; ``.source_index`` maps raw_file)."""
    L.seed_everything(seed, verbose=False)
    cdim = context_dim or model.cfg.context_dim

    # Map raw_file -> integer source id.
    uniq = sorted(set(real.source_ids))
    src_index = {name: i for i, name in enumerate(uniq)}
    src_ids = [src_index[s] for s in real.source_ids]

    # Split by the precursor's assigned split; normalize on TRAIN iRT.
    tr, va = ([], [], [], []), ([], [], [], [])
    for p, lab, rrt, sid in zip(real.precursors, real.labels, real.raw_rt, src_ids):
        bucket = va if p.split == "val" else tr
        for col, val in zip(bucket, (p, lab, rrt, sid)):
            col.append(val)
    irt_train = np.array([lab.rt for lab in tr[1]], dtype=np.float64)
    model.set_norm(float(irt_train.mean()), float(irt_train.std() or 1.0), 0.0, 1.0)

    book = ContextBook(n_acq=len(uniq), n_lc=len(uniq), context_dim=cdim)
    module = RealSpeclibModule(model, book, lr=lr, loss_weights=loss_weights)
    train_ds = RealSpeclibDataset(*tr)
    # Report val on one best-quality example per (dataset, modified_sequence, charge) so
    # abundant peptides don't dominate the metric; train keeps every observation.
    if va[0]:
        va = best_examples(va[0], va[1], va[2], va[3], dataset=dataset)
    val_ds = RealSpeclibDataset(*va) if va[0] else None

    def loader(ds, shuffle):
        return None if ds is None else DataLoader(
            _RealIterable(ds, batch_size, shuffle, seed), batch_size=None
        )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        gradient_clip_val=trainer_kwargs.pop("gradient_clip_val", 1.0),
        enable_checkpointing=trainer_kwargs.pop("enable_checkpointing", False),
        logger=trainer_kwargs.pop("logger", False),
        **trainer_kwargs,
    )
    trainer.fit(module, loader(train_ds, True), loader(val_ds, False))
    module.source_index = src_index
    return module
