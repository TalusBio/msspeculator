"""Real-speclib training regime with per-run context conditioning.

A second Lightning regime over the SAME StudentModel backbone (see lightning.py). It trains
on real experimental spectra (e.g. PROSPECT) and conditions the backbone on two context
vectors, each generated from metadata rather than gradient-descended per source id:

- ``ms_context`` (MS2 side) comes from a run's acquisition factors — instrument, detector,
  fragmentation (categorical) plus collision energy (continuous) — via :class:`MSContextEncoder`.
  Energy is never fabricated, and it is per EXAMPLE, not per run: a spectrum with no recorded
  collision energy carries NaN through the batch and is masked out of the energy term, so that
  term contributes zero for it rather than an invented center value. (The whole-call
  ``energy=None`` escape hatch still exists on :class:`MSContextEncoder`, but this path always
  passes a tensor.)
- ``chrom_context`` (RT side) comes from a per-DATASET :class:`ChromRunbook` row (dataset,
  not raw_file — coarser but good enough to start). Row 0 is reserved as the neutral/iRT row.
  RT is dual-target: the context-free base head (``rt_base``, no chrom_context) is pinned to
  iRT, and the runbook-conditioned head (``rt``) is fit to each example's raw retention_time.
  So the runbook learns ONLY the dataset's LC deviation, and base RT stays the run-independent
  peptide property.
- CCS is unsupervised here (real DDA has none) — left to teacher distillation.

Backbone + context modules are optimized together. To fine-tune only the context (adapt to a
new run), freeze the model and step only the encoder/runbook.
"""

from __future__ import annotations

import math
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.precursors import Precursor
from ..eval import best_per_key, ms2_intensity, precursor_key
from ..models.context import ChromRunbook, MSContextEncoder
from ..models.registry import save_checkpoint
from ..models.student import StudentModel
from ..teacher.base import PrecursorLabels
from .dataset import BatchIterable, LabeledBatch, MSFactors, collate_with_labels, iter_batch_indices
from .lightning import build_trainer
from .losses import mod_align_loss, ms2_cosine_loss, spectral_angle

if TYPE_CHECKING:
    from ..data.prospect import RealLabels


@dataclass(slots=True)
class RealExample:
    """One real observation: the peptide, its teacher/experimental labels, and the factors
    that condition the two context vectors. ``instrument_id``/``detector_id``/
    ``fragmentation_id``/``energy`` drive ``ms_context`` (MS2); ``dataset_id`` drives
    ``chrom_context`` (RT / chromatography)."""

    precursor: Precursor
    label: PrecursorLabels  # ms2 + rt(=iRT) + ccs(NaN for real DDA)
    raw_rt: float  # run-dependent retention time (the chrom_context target)
    instrument_id: int  # -> ms_context
    detector_id: int  # -> ms_context
    fragmentation_id: int  # -> ms_context
    energy: float  # collision energy (NaN if the run carries none -> ms_context omits it)
    dataset_id: int  # -> chrom_context (ChromRunbook row; 0 reserved for iRT/neutral)


@dataclass(slots=True)
class RealBatch:
    base: LabeledBatch  # inputs + ms2_target + rt_target(=iRT) + ccs_target(NaN)
    raw_rt: torch.Tensor  # (B,) run-dependent retention time
    dataset_id: torch.Tensor  # (B,) long, ChromRunbook row (for chrom_context)
    ms_factors: MSFactors  # instrument/detector/fragmentation/energy -> ms_context

    def to(self, device: torch.device | str) -> RealBatch:
        return RealBatch(
            base=self.base.to(device),
            raw_rt=self.raw_rt.to(device),
            dataset_id=self.dataset_id.to(device),
            ms_factors=self.ms_factors.to(device),
        )


class RealSpeclibDataset:
    """A list of :class:`RealExample`; columns the collate/context path needs are cached as
    arrays so :meth:`batches` only pays per-batch indexing, not per-batch attribute walks."""

    def __init__(self, examples: list[RealExample]) -> None:
        self.examples = examples
        self.raw_rt = np.array([e.raw_rt for e in examples], dtype=np.float32)
        self.dataset_id = np.array([e.dataset_id for e in examples], dtype=np.int64)
        self.instrument_id = np.array([e.instrument_id for e in examples], dtype=np.int64)
        self.detector_id = np.array([e.detector_id for e in examples], dtype=np.int64)
        self.fragmentation_id = np.array([e.fragmentation_id for e in examples], dtype=np.int64)
        self.energy = np.array([e.energy for e in examples], dtype=np.float32)

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
            energy_slice = self.energy[idx]
            # Missing energy is masked per example inside MSContextEncoder, so a batch mixing
            # present and absent values is ordinary. Passing None when EVERY value is missing
            # is equivalent, but passing the tensor keeps one code path.
            ms_factors = MSFactors(
                instrument_id=torch.from_numpy(self.instrument_id[idx]),
                detector_id=torch.from_numpy(self.detector_id[idx]),
                fragmentation_id=torch.from_numpy(self.fragmentation_id[idx]),
                energy=torch.from_numpy(energy_slice),
            )
            yield RealBatch(
                base=base,
                raw_rt=torch.from_numpy(self.raw_rt[idx]),
                dataset_id=torch.from_numpy(self.dataset_id[idx]),
                ms_factors=ms_factors,
            )


class RealSpeclibModule(L.LightningModule):
    """Fit the shared backbone on real spectra while learning per-run context: MS2 on
    ``ms_context`` (from acquisition factors via :class:`MSContextEncoder`), the dual RT
    targets on the base head (iRT) and the ``chrom_context``-conditioned head (raw RT, via
    :class:`ChromRunbook`). Set ``freeze_backbone`` to adapt to a new run by training only the
    context (runbook + encoder) — the PEFT path from the module docstring — leaving the
    backbone fixed."""

    def __init__(
        self,
        model: StudentModel,
        runbook: ChromRunbook,
        encoder: MSContextEncoder,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),  # ms2, irt, raw_rt
        dataset_index: dict[str, int] | None = None,
        freeze_backbone: bool = False,
        mod_align_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.runbook = runbook  # chrom_context (RT, per dataset)
        self.encoder = encoder  # ms_context (MS2, from acquisition factors)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        # Ties mass_enc onto a stop-gradiented comp_enc (see losses.mod_align_loss); a separate
        # scalar rather than a 4th slot in loss_weights, which already means (ms2, irt, raw_rt).
        self.mod_align_weight = mod_align_weight
        # dataset name -> chrom_context row; needed to address a trained dataset at inference.
        self.dataset_index = dataset_index
        self.dataset_names = (
            {row: name for name, row in dataset_index.items()} if dataset_index is not None else {}
        )
        if freeze_backbone:  # PEFT: fit only the context modules, backbone stays fixed.
            self.model.requires_grad_(False)

    def on_train_epoch_start(self) -> None:
        self._energy_masked = 0
        self._energy_present = 0

    def on_train_epoch_end(self) -> None:
        self.log_dict(
            {
                "train_energy_masked": float(self._energy_masked),
                "train_energy_present": float(self._energy_present),
            },
            prog_bar=False,
        )

    def transfer_batch_to_device(
        self, batch: RealBatch, device: torch.device | str, dataloader_idx: int
    ) -> RealBatch:
        return batch.to(device)

    def _forward(self, rb: RealBatch) -> dict[str, torch.Tensor]:
        ms_context = self.encoder(
            rb.ms_factors.instrument_id,
            rb.ms_factors.detector_id,
            rb.ms_factors.fragmentation_id,
            rb.ms_factors.energy,
        )
        # Two per-dataset chromatography terms, doing different jobs: the context vector is an
        # additive bias in feature space (peptide-dependent, can reorder), the affine is a
        # scale+shift on the head's output (global, absorbs gradient length / unit differences
        # that an additive bias cannot express).
        chrom_context = self.runbook(rb.dataset_id)
        chrom_affine = self.runbook.affine(rb.dataset_id)
        return self.model.forward_context(
            rb.base.inputs,
            ms_context=ms_context,
            chrom_context=chrom_context,
            chrom_affine=chrom_affine,
        )

    def training_step(self, rb: RealBatch, batch_idx: int) -> torch.Tensor:
        w_ms2, w_irt, w_raw = self.loss_weights
        out = self._forward(rb)
        lb = rb.base
        loss_ms2 = ms2_cosine_loss(out["ms2"], lb.ms2_target, lb.inputs.frag_mask)
        # rt_base (context-free) -> iRT; the chrom_context-conditioned rt -> this dataset's raw RT.
        loss_irt = torch.nn.functional.mse_loss(
            out["rt_base"], self.model.standardize_rt(lb.rt_target)
        )
        loss_raw = torch.nn.functional.mse_loss(out["rt"], self.model.standardize_rt(rb.raw_rt))
        loss = w_ms2 * loss_ms2 + w_irt * loss_irt + w_raw * loss_raw
        log = {"train_ms2": loss_ms2, "train_irt": loss_irt, "train_rawrt": loss_raw}
        if self.mod_align_weight:
            align = mod_align_loss(out["mod_g"], out["mod_m"], lb.inputs.mod_named)
            log["train_mod_align"] = align.detach()
            loss = loss + self.mod_align_weight * align
        e = rb.ms_factors.energy
        if e is not None:
            present = int(torch.isfinite(e).sum())
            self._energy_present += present
            self._energy_masked += int(e.numel()) - present
        self.log_dict(log, prog_bar=False, batch_size=rb.raw_rt.shape[0])
        return loss

    def validation_step(self, rb: RealBatch, batch_idx: int) -> None:
        out = self._forward(rb)
        lb = rb.base
        sa = spectral_angle(out["ms2"], lb.ms2_target, lb.inputs.frag_mask)
        irt_error = (self.model.unstandardize_rt(out["rt_base"]) - lb.rt_target).abs()
        rawrt_error = (self.model.unstandardize_rt(out["rt"]) - rb.raw_rt).abs()

        # A pooled score is dominated by whichever source contributes the most val winners and
        # hides regressions on smaller sources. Report each dataset as its own measurement.
        # Lightning combines repeated logs for a dataset across batches, weighted by the
        # dataset's row count in each batch.
        for dataset_id in rb.dataset_id.unique().tolist():
            name = self.dataset_names.get(dataset_id)
            if name is None:
                raise KeyError(
                    f"validation batch carries unnamed dataset_id={dataset_id!r}; known rows: "
                    f"{sorted(self.dataset_names)}"
                )
            mask = rb.dataset_id == dataset_id
            n = int(mask.sum())
            prefix = f"val/{name}"
            self.log_dict(
                {
                    f"{prefix}/spectral_angle": sa[mask].mean(),
                    f"{prefix}/irt_mae": irt_error[mask].mean(),
                    f"{prefix}/rawrt_mae": rawrt_error[mask].mean(),
                },
                prog_bar=False,
                batch_size=n,
            )
            self.log(f"{prefix}/n", float(n), reduce_fx="sum")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Optimize every trainable parameter across the registered submodules (model + runbook +
        # encoder); with freeze_backbone the model's are already requires_grad=False, so this
        # narrows to the context modules without listing them by hand.
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


class _RealTrainProgress(L.Callback):
    """Low-noise progress for the streaming real-data stage.

    The iterable dataset cannot know its epoch length without decoding every shard first, so
    Lightning's normal percentage bar is misleading here.  Report epoch/batch checkpoints and
    elapsed time instead; the batch index is still useful for spotting a stalled shard or loader.
    """

    def __init__(self, every: int = 100, metrics_path: str | Path | None = None) -> None:
        super().__init__()
        self.every = every
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self._started = 0.0

    def _write_epoch_metrics(self, trainer: L.Trainer) -> None:
        if self.metrics_path is None:
            return
        values = {}
        for key, value in trainer.callback_metrics.items():
            if "/n" in key or not torch.is_tensor(value) or value.numel() != 1:
                continue
            values[key] = float(value.detach().cpu())
        lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        record = {"epoch": trainer.current_epoch + 1, "lr": lr, **values}
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._started = time.perf_counter()
        trainer.print(f"[train] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        n = batch_idx + 1
        if self.every > 0 and n % self.every == 0:
            loss = trainer.callback_metrics.get("train_ms2")
            loss_text = "" if loss is None else f", ms2={float(loss):.4f}"
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            elapsed = time.perf_counter() - self._started
            trainer.print(
                f"[train] epoch {trainer.current_epoch + 1}/{trainer.max_epochs}, "
                f"batch {n}{loss_text}, lr={lr:.6g}, elapsed {elapsed:.0f}s"
            )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        elapsed = time.perf_counter() - self._started
        trainer.print(
            f"[train] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} finished "
            f"in {elapsed:.1f}s, lr={float(trainer.optimizers[0].param_groups[0]['lr']):.6g}"
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._write_epoch_metrics(trainer)
        names = sorted(k for k in trainer.callback_metrics if k.startswith("val/"))
        if names:
            # Keep the line compact; the full per-dataset values remain in callback_metrics and
            # are written to summary.json at the end of the run.
            preview = ", ".join(
                f"{k}={float(trainer.callback_metrics[k]):.4f}"
                for k in names
                if "/n" not in k
            )
            trainer.print(f"[val] epoch {trainer.current_epoch + 1}: {preview}")


class _RealValidationEarlyStop(L.Callback):
    """Stop real-data training when mean per-dataset spectral agreement plateaus."""

    def __init__(self, patience: int, min_delta: float, expected_keys: set[str]) -> None:
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.expected_keys = expected_keys
        # spectral_angle is normalized agreement: 1.0 = identical, so higher is better.
        self.best = float("-inf")
        self.bad = 0

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Lightning runs only a couple of validation batches before epoch 0 as a sanity check.
        # With per-dataset dynamic metric names that prefix is not guaranteed to contain every
        # dataset, so checking completeness here would reject a healthy run before training.
        if trainer.sanity_checking:
            return
        metrics = {
            key: value
            for key, value in trainer.callback_metrics.items()
            if key in self.expected_keys and torch.is_tensor(value) and value.numel() == 1
        }
        missing = sorted(self.expected_keys - metrics.keys())
        if missing:
            available = sorted(
                key
                for key in trainer.callback_metrics
                if key.startswith("val/") and key.endswith("/spectral_angle")
            )
            raise RuntimeError(
                "early stopping expected per-dataset validation metrics that were not logged: "
                f"missing={missing}, available={available}"
            )
        values = [float(value) for value in metrics.values()]
        current = sum(values) / len(values)
        improved = current - self.best > self.min_delta
        if improved:
            self.best, self.bad = current, 0
        else:
            self.bad += 1
        trainer_print = getattr(trainer, "print", print)
        trainer_print(
            f"[early-stop] epoch {trainer.current_epoch + 1}: mean spectral agreement "
            f"current={current:.4f}, best={self.best:.4f}, "
            f"bad={self.bad}/{self.patience}"
            f"{' (new best)' if improved else ''}"
        )
        if self.bad >= self.patience:
            trainer.should_stop = True
            trainer_print(
                f"[early-stop] validation spectral agreement plateaued at {current:.4f} "
                f"(best {self.best:.4f}) -> stopping after epoch {trainer.current_epoch + 1}"
            )


class _RealCheckpoint(L.Callback):
    """Persist inference-ready latest/best snapshots during real-data training."""

    def __init__(self, directory: str | Path, expected_keys: set[str]) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.expected_keys = expected_keys
        # spectral_angle is normalized agreement: 1.0 = identical, so higher is better.
        self.best = float("-inf")

    def _save(self, name: str, pl_module: RealSpeclibModule) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            pl_module.model,
            self.directory / name,
            encoder=pl_module.encoder,
            runbook=pl_module.runbook,
            dataset_index=pl_module.dataset_index,
        )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: RealSpeclibModule) -> None:
        # This snapshot is available even when validation is disabled or crashes afterwards.
        self._save("latest.ckpt", pl_module)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: RealSpeclibModule) -> None:
        if trainer.sanity_checking:
            return
        self._save("latest.ckpt", pl_module)
        if not self.expected_keys:
            return
        metrics = {
            key: value
            for key, value in trainer.callback_metrics.items()
            if key in self.expected_keys and torch.is_tensor(value) and value.numel() == 1
        }
        if len(metrics) != len(self.expected_keys):
            return  # the early-stop callback below will report the missing keys
        current = sum(float(value) for value in metrics.values()) / len(metrics)
        if current > self.best:
            self.best = current
            self._save("best.ckpt", pl_module)


def _build_examples(
    real: RealLabels, encoder: MSContextEncoder, dataset_id: int, instrument: str
) -> list[RealExample]:
    """Turn ``RealLabels`` (parallel columns) into per-example records, resolving each run's
    acquisition factors ONCE (few runs, many examples) and reusing them across its examples.
    Every example shares the same ``dataset_id`` (RT is keyed per dataset, not per raw_file) and
    the same ``instrument`` — a pool-level constant threaded from config, not per-run metadata
    (PROSPECT carries no instrument column)."""

    def acq(name: str, key: str, default):
        return real.acquisition.get(name, {}).get(key, default)

    names = set(real.source_ids)
    inst_id = encoder.instrument_id(instrument)
    det_of = {n: encoder.detector_id(acq(n, "mass_analyzer", "")) for n in names}
    frag_of = {n: encoder.fragmentation_id(acq(n, "fragmentation", "")) for n in names}
    energy_of = {n: float(acq(n, "collision_energy", float("nan"))) for n in names}

    return [
        RealExample(
            precursor=p,
            label=lab,
            raw_rt=float(rrt),
            instrument_id=inst_id,
            detector_id=det_of[s],
            fragmentation_id=frag_of[s],
            energy=energy_of[s],
            dataset_id=dataset_id,
        )
        for p, lab, rrt, s in zip(real.precursors, real.labels, real.raw_rt, real.source_ids)
    ]


def _dedupe_val(examples: list[RealExample], dataset_name: str | None) -> list[RealExample]:
    """Keep one best-quality example per (dataset, modified_sequence, charge) so abundant
    peptides don't dominate the val metric. Train keeps every observation."""
    keys = [precursor_key(e.precursor, dataset_name) for e in examples]
    quality = [ms2_intensity(e.label) for e in examples]
    return [examples[i] for i in best_per_key(quality, keys)]


def establish_rt_norm(model: StudentModel, stats: list[tuple[int, float, float]]) -> bool:
    """Set the global RT affine from combined ``(n, sum, sumsq)`` iRT statistics.

    Sufficient statistics rather than an array, so several sources combine by addition and
    nothing has to be held. The population must be exactly what training sees — for this
    pipeline the train split alone, with val and test both genuinely held out — and it is
    counted pre-decode, so it includes spectra that later drop out for having no surviving b/y
    fragments or fewer than two residues. Near-exact rather than exact, and deliberately
    preferred over the alternatives (Welford over the stream, or estimating from the first
    shard), both of which would be strictly worse for a value the set-once rule makes
    permanent for the run.

    Returns whether it set anything: a pretrain->train curriculum inherits the frame the
    pretrain stage established and must not recalibrate a trained head mid-stream.
    """
    if bool(model.norm_established):
        return False
    n = sum(c for c, _, _ in stats)
    if n == 0:
        raise ValueError(
            "no examples to establish the RT affine from; every sequence in the sources that "
            "feed training hashed to a held-out split"
        )
    total = sum(t for _, t, _ in stats)
    sumsq = sum(q for _, _, q in stats)
    mean = total / n
    var = max(sumsq / n - mean * mean, 0.0)
    model.set_norm(rt_mean=float(mean), rt_std=float(math.sqrt(var) or 1.0))
    return True


def fit_realspeclib_datasets(
    model: StudentModel,
    train_ds: RealSpeclibDataset,
    val_ds: RealSpeclibDataset | None,
    *,
    runbook: ChromRunbook,
    dataset_index: dict[str, int],
    encoder: MSContextEncoder,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    grad_clip: float = 1.0,
    seed: int = 0,
    accelerator: str = "auto",
    freeze_backbone: bool = False,
    mod_align_weight: float = 1.0,
    progress_log_every: int = 100,
    progress_metrics_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 1e-3,
    **trainer_kwargs,
) -> RealSpeclibModule:
    """Fit on datasets the caller already built.

    Split out of :func:`fit_realspeclib` so the streaming path can pass a
    ``StreamingRealDataset`` for train and a materialised one for val. Both only have to
    expose ``batches(batch_size, shuffle, generator)``. Normalisation is NOT touched here:
    the caller establishes the RT affine before training (see :func:`establish_rt_norm`).

    Global RNG seeding is the CALLER's responsibility (e.g. via ``L.seed_everything``) — this
    function does not call it. ``seed`` here only threads into ``BatchIterable`` to fix the
    per-epoch batch shuffle; it does not touch model/encoder/runbook init or dropout. Seeding
    globally here too would reset the RNG stream a second time after the caller already built
    encoder/runbook (which consume RNG draws), silently changing training's dropout stream
    out from under a caller who assumed one seed call meant one deterministic run.
    """
    module = RealSpeclibModule(
        model,
        runbook,
        encoder,
        lr=lr,
        weight_decay=weight_decay,
        loss_weights=loss_weights,
        dataset_index=dataset_index,
        freeze_backbone=freeze_backbone,
        mod_align_weight=mod_align_weight,
    )

    def loader(ds, shuffle: bool) -> DataLoader | None:
        if ds is None:
            return None
        return DataLoader(BatchIterable(ds, batch_size, shuffle, seed), batch_size=None)

    callbacks = list(trainer_kwargs.pop("callbacks", []))
    if progress_log_every > 0:
        callbacks.append(_RealTrainProgress(progress_log_every, progress_metrics_path))
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative")
    if early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be non-negative")
    if early_stop_patience > 0 and val_ds is not None and len(val_ds):
        expected_names = {
            module.dataset_names[int(dataset_id)] for dataset_id in np.unique(val_ds.dataset_id)
        }
        expected_keys = {f"val/{name}/spectral_angle" for name in expected_names}
        callbacks.append(
            _RealValidationEarlyStop(early_stop_patience, early_stop_min_delta, expected_keys)
        )
    if checkpoint_dir is not None:
        checkpoint_keys = (
            {
                f"val/{module.dataset_names[int(dataset_id)]}/spectral_angle"
                for dataset_id in np.unique(val_ds.dataset_id)
            }
            if val_ds is not None and len(val_ds)
            else set()
        )
        callbacks.insert(0, _RealCheckpoint(checkpoint_dir, checkpoint_keys))
    trainer = build_trainer(epochs, accelerator, grad_clip, callbacks=callbacks, **trainer_kwargs)
    trainer.fit(module, loader(train_ds, True), loader(val_ds, False))
    return module


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
    instrument: str = "Lumos",
    encoder: MSContextEncoder | None = None,
    runbook: ChromRunbook | None = None,
    freeze_backbone: bool = False,
    mod_align_weight: float = 1.0,
    **trainer_kwargs,
) -> RealSpeclibModule:
    """Train on real spectra with per-run context. ``ms_context`` comes from the acquisition
    factors via a :class:`MSContextEncoder`; ``chrom_context`` from a per-dataset
    :class:`ChromRunbook` (row 0 = iRT/neutral). Pass an existing ``encoder``/``runbook`` to
    continue a curriculum (share them with the warmup); their ``context_dim`` must match the
    model's. Set ``freeze_backbone`` to adapt to a run by fitting only the context (PEFT).
    ``dataset_name`` is both the val dedup key's dataset label and the runbook row key (one
    dataset -> one row for now; RT is keyed per dataset, not per raw_file — coarser but good
    enough to start). ``instrument`` is a pool-level constant (default "Lumos" for PROSPECT,
    which has no per-run instrument metadata) applied to every example, on equal footing with
    detector/fragmentation. Returns the module (``.model``/``.encoder``/``.runbook`` trained;
    ``.dataset_index`` maps dataset name -> chrom_context row)."""
    L.seed_everything(seed, verbose=False)
    cdim = context_dim or model.cfg.context_dim
    if cdim <= 0:
        raise ValueError(f"context_dim must be positive to condition on acquisition, got {cdim}")
    if encoder is not None and encoder.context_dim != cdim:
        raise ValueError(f"encoder context_dim {encoder.context_dim} != model's {cdim}")
    if runbook is not None and runbook.context_dim != cdim:
        raise ValueError(f"runbook context_dim {runbook.context_dim} != model's {cdim}")
    encoder = encoder or MSContextEncoder(context_dim=cdim)

    # One dataset -> one runbook row (row 0 is reserved for iRT/neutral).
    key = dataset_name or "default"
    dataset_index = {key: 1}
    runbook = runbook or ChromRunbook(n_datasets=len(dataset_index), context_dim=cdim)

    examples = _build_examples(real, encoder, dataset_index[key], instrument)
    train = [e for e in examples if e.precursor.split != "val"]
    val = [e for e in examples if e.precursor.split == "val"]

    # RT/CCS scale is ONE global affine, established once at cold start and never re-set when
    # a dataset is added — per-dataset RT variation is the ChromRunbook's job, not the norm's.
    # So establish it here only on a cold start (real-only, no pretrain); in a pretrain->real
    # curriculum, inherit the frame the pretrain stage set rather than recalibrating the
    # head mid-stream.
    #
    # CCS is never touched here at all: PROSPECT carries no CCS, so this regime has nothing
    # to estimate from, and writing an identity norm would overwrite the pretrain
    # calibration, leaving a trained head whose predictions denormalize to raw standardized
    # values.
    if not bool(model.norm_established):
        irt_train = np.array([e.label.rt for e in train], dtype=np.float64)
        model.set_norm(rt_mean=float(irt_train.mean()), rt_std=float(irt_train.std() or 1.0))

    train_ds = RealSpeclibDataset(train)
    val_ds = RealSpeclibDataset(_dedupe_val(val, dataset_name)) if val else None

    return fit_realspeclib_datasets(
        model,
        train_ds,
        val_ds,
        runbook=runbook,
        dataset_index=dataset_index,
        encoder=encoder,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        loss_weights=loss_weights,
        grad_clip=grad_clip,
        seed=seed,
        accelerator=accelerator,
        freeze_backbone=freeze_backbone,
        mod_align_weight=mod_align_weight,
        **trainer_kwargs,
    )
