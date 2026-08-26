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
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from ..data.augmentation import substitute_residues
from ..data.encode import Batch
from ..diagnostics import SA_HISTOGRAM_BINS, SA_HISTOGRAM_EDGES
from ..data.precursors import Precursor

from ..models.context import ChromRunbook, MSContextEncoder
from ..models.registry import save_checkpoint
from ..models.student import StudentModel
from ..teacher.base import PrecursorLabels
from .dataset import BatchIterable, LabeledBatch, MSFactors, collate_with_labels, iter_batch_indices
from .lightning import build_trainer
from .losses import labeled_mse, mod_align_loss, ms2_cosine_loss, spectral_angle


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


class BatchSource(Protocol):
    """What the trainer needs of a dataset: a row count, and this epoch's batches.

    Two implementations satisfy it — the in-memory :class:`RealSpeclibDataset` below and the
    streaming ``PreparedStreamingDataset``, which is what production passes. Annotating the
    concrete in-memory class named only one of the two and would have made a type checker
    reject the real call site.
    """

    def __len__(self) -> int: ...

    def batches(
        self, batch_size: int, shuffle: bool, generator: torch.Generator
    ) -> Iterator[RealBatch]: ...


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
        residue_substitution_probability: float = 0.0,
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
        if not 0.0 <= residue_substitution_probability <= 1.0:
            raise ValueError("residue_substitution_probability must be between 0 and 1")
        self.residue_substitution_probability = residue_substitution_probability
        self.last_validation_step: int | None = None
        # dataset name -> spectral-angle counts on the shared grid, rebuilt each validation check.
        # Kept as plain state rather than a logged scalar: the point is the shape of the
        # distribution, which is what makes it comparable to the teacher and the ceiling.
        self.val_sa_histograms: dict[str, np.ndarray] = {}
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

    def _forward(self, rb: RealBatch, inputs: Batch | None = None) -> dict[str, torch.Tensor]:
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
            rb.base.inputs if inputs is None else inputs,
            ms_context=ms_context,
            chrom_context=chrom_context,
            chrom_affine=chrom_affine,
        )

    def training_step(self, rb: RealBatch, batch_idx: int) -> torch.Tensor:
        w_ms2, w_irt, w_raw = self.loss_weights
        lb = rb.base
        inputs = substitute_residues(lb.inputs, self.residue_substitution_probability)
        out = self._forward(rb, inputs)
        loss_ms2 = ms2_cosine_loss(out["ms2"], lb.ms2_target, lb.inputs.frag_mask)
        # rt_base (context-free) -> iRT; the chrom_context-conditioned rt -> this dataset's raw RT.
        # Both masked per row: a source can carry one RT label and not the other.
        loss_irt = labeled_mse(out["rt_base"], self.model.standardize_rt(lb.rt_target))
        loss_raw = labeled_mse(out["rt"], self.model.standardize_rt(rb.raw_rt))
        # This is deliberately the single source of truth for the objective and its loss
        # telemetry: adding a term to optimization necessarily adds it to logging too.
        loss_terms = {
            "train_ms2": (w_ms2, loss_ms2),
            "train_irt": (w_irt, loss_irt),
            "train_rawrt": (w_raw, loss_raw),
        }
        if self.mod_align_weight:
            loss_terms["train_mod_align"] = (
                self.mod_align_weight,
                mod_align_loss(out["mod_g"], out["mod_m"], inputs.mod_has_composition),
            )
        loss = sum(weight * value for weight, value in loss_terms.values())
        # Keep the reporting metric beside its optimization surrogate. Spectral angle is only
        # used for telemetry here: differentiating through arccos is numerically ill-conditioned
        # near identical spectra, while cosine loss has the same per-spectrum optimum.
        with torch.no_grad():
            train_sa = spectral_angle(out["ms2"], lb.ms2_target, lb.inputs.frag_mask).mean()
        log = {name: value for name, (_, value) in loss_terms.items()}
        log["train_spectral_angle"] = train_sa
        # An unlabeled row still supervises MS2, so a low fraction is a corpus fact rather than
        # a fault -- but it has to be visible, or an iRT term training on a tenth of the batch
        # reads as a converged one.
        log["train_irt_labeled_fraction"] = torch.isfinite(lb.rt_target).float().mean()
        if self.residue_substitution_probability:
            log["train_residue_augmented_fraction"] = (
                (inputs.tokens != lb.inputs.tokens).any(dim=1).float().mean()
            )
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
        # Paired with the mask of rows that carry the label, keyed on the target rather than on
        # the error, so an unlabeled row is skipped while a NaN *prediction* still shows up as
        # a NaN metric instead of being masked away with it.
        rt_errors = {
            "irt_mae": (
                (self.model.unstandardize_rt(out["rt_base"]) - lb.rt_target).abs(),
                torch.isfinite(lb.rt_target),
            ),
            "rawrt_mae": (
                (self.model.unstandardize_rt(out["rt"]) - rb.raw_rt).abs(),
                torch.isfinite(rb.raw_rt),
            ),
        }

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
            # A mean cannot be drawn against the teacher yardstick or the replicate ceiling, both
            # of which ship a distribution. Accumulate counts on the shared grid so all three
            # overlay exactly; 50 ints per dataset per check costs nothing.
            # Lightning's sanity check runs this before training, so accumulating there would draw
            # a handful of untrained spectra as the student series in the first published panel.
            if self.trainer.sanity_checking:
                continue
            counts = self.val_sa_histograms.setdefault(
                name, np.zeros(SA_HISTOGRAM_BINS, dtype=np.int64)
            )
            counts += np.histogram(
                sa[mask].detach().float().cpu().numpy(),
                bins=SA_HISTOGRAM_BINS,
                range=(0.0, 1.0),
            )[0]
            self.log(f"{prefix}/spectral_angle", sa[mask].mean(), batch_size=n)
            # Each RT metric is weighted by its own labeled count, not by the batch: weighting a
            # half-labeled source's MAE by every row would let it outvote a fully labeled one.
            for metric, (error, labeled) in rt_errors.items():
                selected = mask & labeled
                rows = int(selected.sum())
                if rows:
                    self.log(f"{prefix}/{metric}", error[selected].mean(), batch_size=rows)
            self.log(f"{prefix}/n", float(n), reduce_fx="sum")

    def on_validation_epoch_start(self) -> None:
        # Each check reports its own distribution; accumulating across checks would smear the
        # student's progress into its own history.
        self.val_sa_histograms = {}

    def on_validation_epoch_end(self) -> None:
        # Used by the fit wrapper to guarantee one final validation when a short run finishes
        # before its first wall-clock check, without repeating a check that already ran after
        # the last optimizer step.
        self.last_validation_step = int(self.global_step)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Optimize every trainable parameter across the registered submodules (model + runbook +
        # encoder); with freeze_backbone the model's are already requires_grad=False, so this
        # narrows to the context modules without listing them by hand.
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


class _RealTrainProgress(L.Callback):
    """Low-noise progress for the streaming real-data stage.

    Prepared manifests know their row count but not the exact number of length-bucketed partial
    batches. ``ceil(rows / batch_size)`` is nevertheless accurate to within a few dozen batches
    across a roughly 350k-batch epoch, so report it explicitly as an approximate denominator.
    """

    def __init__(
        self,
        every: int = 100,
        metrics_path: str | Path | None = None,
        artifact_mirror=None,
        estimated_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.every = every
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self.artifact_mirror = artifact_mirror
        self.estimated_batches = estimated_batches
        self._started = 0.0
        self._examples = 0
        self._validation_check = 0

    def _write_validation_metrics(
        self, trainer: L.Trainer, pl_module: L.LightningModule | None = None
    ) -> None:
        if self.metrics_path is None:
            return
        values = {}
        for key, value in trainer.callback_metrics.items():
            if "/n" in key or not torch.is_tensor(value) or value.numel() != 1:
                continue
            values[key] = float(value.detach().cpu())
        lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        record: dict[str, Any] = {
            "validation_check": self._validation_check,
            "epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
            "lr": lr,
            **values,
        }
        # The per-dataset spectral-angle distribution, on the same grid as the published teacher
        # yardstick and the corpus replicate ceiling, so a check can be drawn against both rather
        # than compared as three unrelated means.
        histograms = getattr(pl_module, "val_sa_histograms", None)
        if histograms:
            record["val_sa_histogram_bin_edges"] = list(SA_HISTOGRAM_EDGES)
            record["val_sa_histogram"] = {
                name: [int(count) for count in counts]
                for name, counts in sorted(histograms.items())
            }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        if self.artifact_mirror is not None:
            self.artifact_mirror(self.metrics_path)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._started = time.perf_counter()
        self._examples = 0
        denominator = (
            f", approximately {self.estimated_batches:,} batches"
            if self.estimated_batches is not None
            else ""
        )
        trainer.print(
            f"[train] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started{denominator}"
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        n = batch_idx + 1
        self._examples += int(batch.base.inputs.charge.numel())
        if self.every > 0 and n % self.every == 0:
            loss = trainer.callback_metrics.get("train_ms2")
            loss_text = "" if loss is None else f", ms2={float(loss):.4f}"
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            elapsed = time.perf_counter() - self._started
            progress = f"batch {n:,}"
            eta_text = ""
            if self.estimated_batches is not None:
                fraction = min(n / self.estimated_batches, 1.0)
                remaining = max(self.estimated_batches - n, 0) * elapsed / n
                progress += f"/~{self.estimated_batches:,} ({fraction:.1%})"
                eta_text = f", epoch_eta={remaining / 3600:.2f}h"
            examples_per_second = self._examples / elapsed if elapsed > 0 else 0.0
            trainer.print(
                f"[train] epoch {trainer.current_epoch + 1}/{trainer.max_epochs}, "
                f"{progress}{loss_text}, lr={lr:.6g}, {examples_per_second:,.0f} examples/s, "
                f"elapsed={elapsed:.0f}s{eta_text}"
            )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        elapsed = time.perf_counter() - self._started
        trainer.print(
            f"[train] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} finished "
            f"in {elapsed:.1f}s, lr={float(trainer.optimizers[0].param_groups[0]['lr']):.6g}"
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._validation_check += 1
        self._write_validation_metrics(trainer, pl_module)
        names = sorted(k for k in trainer.callback_metrics if k.startswith("val/"))
        if names:
            # Keep the line compact; the full per-dataset values remain in callback_metrics and
            # are written to summary.json at the end of the run.
            preview = ", ".join(
                f"{k}={float(trainer.callback_metrics[k]):.4f}" for k in names if "/n" not in k
            )
            trainer.print(
                f"[val] check {self._validation_check}, epoch {trainer.current_epoch + 1}, "
                f"step {trainer.global_step:,}: {preview}"
            )


def _validation_metrics(trainer: L.Trainer, expected_keys: set[str]) -> dict[str, float]:
    """The scalar validation metrics of `expected_keys` this check logged.

    One reader for the three callbacks that watch validation, each of which reacts differently
    to an incomplete set: the checkpoint saves anyway, early stopping raises, the decay waits.
    """
    return {
        key: float(value.detach().cpu())
        for key, value in trainer.callback_metrics.items()
        if key in expected_keys and torch.is_tensor(value) and value.numel() == 1
    }


class _RealPlateauDecay(L.Callback):
    """Cut the learning rate when mean per-dataset agreement stops improving.

    Deliberately more impatient than early stopping, and validated as such where both are
    registered: they watch the same aggregate at the same validation checks, so a decay patience
    at or above the stopping patience would end the run before the rate ever moved.

    A horizon-based schedule is the obvious alternative and does not fit this stage. Cosine and
    OneCycle need to know where the end is; here the end is wherever early stopping lands, and
    the first full local run stopped at epoch 8 of a nominal 60 -- a cosine over that horizon
    would still have been at 98% of the initial rate. What that run actually showed was
    agreement oscillating in a 0.4% band from epoch 3 onward without trending, which is a
    converged-at-this-rate signature: more patience buys more oscillation, a smaller step does
    not.
    """

    def __init__(
        self, patience: int, factor: float, min_lr: float, min_delta: float, expected_keys: set[str]
    ) -> None:
        super().__init__()
        if not 0.0 < factor < 1.0:
            raise ValueError(f"lr_decay_factor must be between 0 and 1, got {factor}")
        if min_lr < 0:
            raise ValueError("lr_decay_min must be non-negative")
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.expected_keys = expected_keys
        self.best = float("-inf")
        self.bad = 0

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = _validation_metrics(trainer, self.expected_keys)
        if len(metrics) != len(self.expected_keys):
            return  # early stopping owns reporting an incomplete check
        current = sum(metrics.values()) / len(metrics)
        if current - self.best > self.min_delta:
            self.best, self.bad = current, 0
            return
        self.bad += 1
        if self.bad < self.patience:
            return
        groups = [group for optimizer in trainer.optimizers for group in optimizer.param_groups]
        if all(float(group["lr"]) <= self.min_lr for group in groups):
            # Nothing left to give. Say nothing and let early stopping end the run, rather than
            # reporting a decay every `patience` checks that does not change the rate.
            return
        # Reset the counter but keep `best`: the next decay has to earn its way from the best
        # agreement seen so far, not from whatever this plateau happens to sit at.
        self.bad = 0
        before = min(float(group["lr"]) for group in groups)
        for group in groups:
            group["lr"] = max(float(group["lr"]) * self.factor, self.min_lr)
        after = min(float(group["lr"]) for group in groups)
        getattr(trainer, "print", print)(
            f"[lr-decay] agreement plateaued at {current:.4f} (best {self.best:.4f}) -> lr "
            f"{before:.3e} to {after:.3e}"
        )


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
        metrics = _validation_metrics(trainer, self.expected_keys)
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
        current = sum(metrics.values()) / len(metrics)
        improved = current - self.best > self.min_delta
        if improved:
            self.best, self.bad = current, 0
        else:
            self.bad += 1
        trainer_print = getattr(trainer, "print", print)
        trainer_print(
            f"[early-stop] validation check at epoch {trainer.current_epoch + 1}, "
            f"step {trainer.global_step:,}: mean spectral agreement "
            f"current={current:.4f}, best={self.best:.4f}, "
            f"bad={self.bad}/{self.patience}"
            f"{' (new best)' if improved else ''}"
        )
        if self.bad >= self.patience:
            trainer.should_stop = True
            trainer_print(
                f"[early-stop] validation spectral agreement plateaued at {current:.4f} "
                f"(best {self.best:.4f}) -> stopping at step {trainer.global_step:,}"
            )


class _RealEpochValidation(L.Callback):
    """Validate at the epoch boundary too, unless that epoch already had a check.

    A wall-clock ``val_check_interval`` is the ONLY validation trigger a prepared streaming run
    has: the loader is unsized, ``check_val_every_n_epoch`` is None, and the boundary escape
    hatch in Lightning's own loop tests ``val_check_batch == inf``, which timed mode never sets.
    An epoch shorter than the interval therefore ends with no validation at all — no fresh
    metrics for the epoch-end checkpoint, and none for early stopping.

    The rule here asks one question: has a check run since this epoch began? That deliberately
    needs no estimate of how long an epoch takes. An epoch's duration is unknown before the
    first epoch and can change mid-run — a corpus, batch-size or hardware change moves steps per
    epoch — so an interval derived from an earlier epoch would go stale with nothing to notice it.
    Validation stays at most one interval (plus a batch) apart in either direction.
    """

    def __init__(self) -> None:
        super().__init__()
        # Validations at or before this step belong to an earlier epoch.
        self._counts_from = 0
        self._forced_pending = False

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        # The interval is parsed at Trainer construction, so a non-timed run can be rejected
        # before training starts. The stamp this class writes is only created once the fit loop
        # runs, so that half is checked at the write site instead.
        if getattr(trainer, "_val_check_time_interval", None) is None:
            raise RuntimeError(
                "epoch-boundary validation needs a timed val_check_interval; this trainer has "
                "no trainer._val_check_time_interval, so the boundary check cannot be triggered"
            )

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._counts_from = trainer.global_step

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._forced_pending and not trainer.sanity_checking:
            # A forced check runs on the first batch of the NEXT epoch, so it closes the epoch
            # that asked for it. Crediting it to the new epoch would satisfy that epoch's test as
            # well, and one check would then cover every second epoch.
            self._forced_pending = False
            self._counts_from = trainer.global_step

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: RealSpeclibModule) -> None:
        validated = pl_module.last_validation_step
        if validated is not None and validated > self._counts_from:
            return
        # Say so loudly if the stamp is gone: a silent no-op here would return the run to
        # validating on the interval alone, which is exactly the behaviour this class exists to fix.
        if not hasattr(trainer, "_last_val_time"):
            raise RuntimeError(
                "epoch-boundary validation forces a check by backdating trainer._last_val_time, "
                "which this Lightning version does not set; the boundary check cannot be triggered"
            )
        self._forced_pending = True
        # The interval test is ``now - _last_val_time >= interval``, and the evaluation loop
        # restamps it when a check runs — so backdating makes the next batch end due exactly once.
        trainer._last_val_time = float("-inf")


class _RealCheckpoint(L.Callback):
    """Persist inference-ready latest/best snapshots during real-data training."""

    def __init__(
        self, directory: str | Path, expected_keys: set[str], artifact_mirror=None
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.expected_keys = expected_keys
        self.artifact_mirror = artifact_mirror
        # spectral_angle is normalized agreement: 1.0 = identical, so higher is better.
        self.best = float("-inf")

    def _validation_values(self, trainer: L.Trainer) -> dict[str, float]:
        return _validation_metrics(trainer, self.expected_keys)

    def _save(
        self,
        name: str,
        trainer: L.Trainer,
        pl_module: RealSpeclibModule,
        values: dict[str, float] | None = None,
        validated_at_step: int | None = None,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        values = self._validation_values(trainer) if values is None else values
        # `values` empty is a real state, not a degenerate one: a run with no validation datasets
        # has no expected keys, and the epoch-end snapshot is taken before anything has validated.
        # Both make the count match at zero, where a mean does not exist.
        mean = (
            sum(values.values()) / len(values)
            if values and len(values) == len(self.expected_keys)
            else None
        )
        training_metadata = {
            "stage": "train",
            "checkpoint_kind": Path(name).stem,
            "global_step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch) + 1,
            "validation": {
                "metric": "mean_per_dataset_spectral_angle",
                "values": dict(sorted(values.items())),
                "mean": mean,
                "best_checkpoint_mean": self.best if math.isfinite(self.best) else None,
                # None when no validation has run yet, which the end-of-epoch snapshot below
                # depends on: an epoch shorter than the validation interval reaches its first
                # epoch boundary with nothing validated, and this checkpoint is exactly the one
                # that has to survive that.
                "validated_at_step": (
                    pl_module.last_validation_step
                    if validated_at_step is None
                    else int(validated_at_step)
                ),
            },
        }
        save_checkpoint(
            pl_module.model,
            self.directory / name,
            encoder=pl_module.encoder,
            runbook=pl_module.runbook,
            dataset_index=pl_module.dataset_index,
            training_metadata=training_metadata,
        )
        if self.artifact_mirror is not None:
            self.artifact_mirror(self.directory / name)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: RealSpeclibModule) -> None:
        # This snapshot is available even when validation is disabled or crashes afterwards.
        self._save("latest.ckpt", trainer, pl_module)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: RealSpeclibModule) -> None:
        if trainer.sanity_checking:
            return
        if not self.expected_keys:
            self._save("latest.ckpt", trainer, pl_module, {}, int(trainer.global_step))
            return
        values = self._validation_values(trainer)
        if len(values) != len(self.expected_keys):
            self._save("latest.ckpt", trainer, pl_module, values, int(trainer.global_step))
            return  # the early-stop callback below will report the missing keys
        current = sum(values.values()) / len(values)
        if current > self.best:
            self.best = current
            self._save("latest.ckpt", trainer, pl_module, values, int(trainer.global_step))
            self._save("best.ckpt", trainer, pl_module, values, int(trainer.global_step))
        else:
            self._save("latest.ckpt", trainer, pl_module, values, int(trainer.global_step))


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
    train_ds: BatchSource,
    val_ds: BatchSource | None,
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
    artifact_mirror=None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 1e-3,
    lr_decay_patience: int = 0,
    lr_decay_factor: float = 0.5,
    lr_decay_min: float = 0.0,
    residue_substitution_probability: float = 0.0,
    num_workers: int = 0,
    **trainer_kwargs,
) -> RealSpeclibModule:
    """Fit on datasets the caller already built.

    Takes any :class:`BatchSource`, which is how the prepared path passes a
    ``PreparedStreamingDataset`` for both train and validation. Normalisation is NOT touched
    here: the caller establishes the RT affine before training (see :func:`establish_rt_norm`).

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
        residue_substitution_probability=residue_substitution_probability,
    )

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    def loader(ds, shuffle: bool) -> DataLoader | None:
        if ds is None:
            return None
        if num_workers and not getattr(ds, "worker_partitioned", False):
            raise ValueError(
                "num_workers > 0 requires a dataset that partitions work between workers"
            )
        return DataLoader(
            BatchIterable(ds, batch_size, shuffle, seed),
            batch_size=None,
            num_workers=num_workers,
            # Linux defaults to fork, which can deadlock after Torch has initialized thread pools.
            # Spawn is slower to start but safe; persistent workers amortize that cost.
            multiprocessing_context="spawn" if num_workers else None,
            # Workers retain BatchIterable._epoch between epochs, so shard ordering advances
            # rather than resetting when DataLoader reconstructs its processes.
            persistent_workers=num_workers > 0,
        )

    callbacks = list(trainer_kwargs.pop("callbacks", []))
    if trainer_kwargs.get("logger"):
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    val_dataset_ids = (
        val_ds.dataset_ids_present
        if hasattr(val_ds, "dataset_ids_present")
        else set(np.unique(val_ds.dataset_id))
        if val_ds is not None and len(val_ds)
        else set()
    )
    if progress_log_every > 0:
        estimated_batches = math.ceil(len(train_ds) / batch_size) if len(train_ds) else None
        callbacks.append(
            _RealTrainProgress(
                progress_log_every,
                progress_metrics_path,
                artifact_mirror,
                estimated_batches=estimated_batches,
            )
        )
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative")
    if early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be non-negative")
    if lr_decay_patience < 0:
        raise ValueError("lr_decay_patience must be non-negative")
    if 0 < early_stop_patience <= lr_decay_patience:
        raise ValueError(
            f"lr_decay_patience {lr_decay_patience} must be below early_stop_patience "
            f"{early_stop_patience}, or the run stops before the rate is ever cut"
        )
    if val_ds is not None and val_dataset_ids:
        expected_names = {module.dataset_names[int(dataset_id)] for dataset_id in val_dataset_ids}
        expected_keys = {f"val/{name}/spectral_angle" for name in expected_names}
        # Decay before stop, so a plateau that the smaller rate resolves is not counted as a
        # stopping signal that a later callback has already acted on. Both take
        # `early_stop_min_delta`: what counts as an improvement is one decision, not two.
        if lr_decay_patience > 0:
            callbacks.append(
                _RealPlateauDecay(
                    lr_decay_patience,
                    lr_decay_factor,
                    lr_decay_min,
                    early_stop_min_delta,
                    expected_keys,
                )
            )
        if early_stop_patience > 0:
            callbacks.append(
                _RealValidationEarlyStop(early_stop_patience, early_stop_min_delta, expected_keys)
            )
    if checkpoint_dir is not None:
        checkpoint_keys = {
            f"val/{module.dataset_names[int(dataset_id)]}/spectral_angle"
            for dataset_id in val_dataset_ids
        }
        callbacks.insert(0, _RealCheckpoint(checkpoint_dir, checkpoint_keys, artifact_mirror))
    # Only a timed interval leaves the epoch boundary untriggered; a batch-count interval already
    # validates on the last batch of an unsized loader.
    if val_ds is not None and isinstance(trainer_kwargs.get("val_check_interval"), timedelta):
        callbacks.append(_RealEpochValidation())
    trainer = build_trainer(epochs, accelerator, grad_clip, callbacks=callbacks, **trainer_kwargs)
    train_loader = loader(train_ds, True)
    val_loader = loader(val_ds, False)
    trainer.fit(module, train_loader, val_loader)
    if val_loader is not None and module.last_validation_step != trainer.global_step:
        trainer.print(
            f"[val] running final check at step {trainer.global_step:,}; "
            "the last optimizer step has not been validated"
        )
        trainer.validate(module, val_loader, verbose=False)
    return module
