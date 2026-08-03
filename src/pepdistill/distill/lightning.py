"""Lightning wrappers for distillation.

Design: the :class:`StudentModel` *is* the shared backbone — a plain ``nn.Module`` holding
embeds + trunk + heads + norm buffers. Each training *regime* is a thin
:class:`lightning.LightningModule` that owns a reference to that same StudentModel and
defines only its loss/data contract. Multiple regimes constructed with the *same* model
instance share weights, so you can, e.g., pre-train with :class:`DistillModule` (teacher
soft labels) and then fine-tune the identical backbone with ``RealSpeclibModule``
(experimental spectra) — "pass the backbone around". :mod:`pepdistill.distill.pipeline`
chains exactly that (pretrain -> real train) from one config.
"""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader

from ..models.context import MSContextEncoder
from ..models.student import StudentModel
from .dataset import BatchIterable, DistillDataset, LabeledBatch
from .losses import distill_loss, mod_align_loss, spectral_angle


class _CappedOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    """OneCycle schedule that holds its final LR if a streaming loader runs long.

    Iterable datasets do not expose a reliable length before teacher labeling. A configured
    ``total_steps`` therefore describes the intended cycle, not a hard training limit.
    """

    def step(self, epoch: int | None = None) -> None:
        # OneCycleLR raises once ``last_epoch`` is already at the endpoint. Keep the final
        # learning rate instead; this makes an approximate stream-size estimate safe.
        if epoch is None and self.last_epoch >= self.total_steps:
            self._step_count += 1
            return
        super().step(epoch)


def build_trainer(epochs: int, accelerator: str, grad_clip: float, **trainer_kwargs) -> L.Trainer:
    """Shared Trainer defaults for the distill/real-speclib regimes: checkpointing and logging
    off unless the caller overrides via ``trainer_kwargs``; everything else passes straight
    through. One place so the two regimes' Trainer setup can't drift apart."""
    return L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        gradient_clip_val=grad_clip,
        enable_checkpointing=trainer_kwargs.pop("enable_checkpointing", False),
        logger=trainer_kwargs.pop("logger", False),
        **trainer_kwargs,
    )


class DistillModule(L.LightningModule):
    """Distillation regime: fit the student to teacher soft labels (MS2 + RT + CCS)."""

    def __init__(
        self,
        model: StudentModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        context_encoder: MSContextEncoder | None = None,
        mod_align_weight: float = 1.0,
        onecycle_max_lr: float | None = None,
        onecycle_total_steps: int | None = None,
        onecycle_pct_start: float = 0.3,
        onecycle_div_factor: float = 25.0,
        onecycle_final_div_factor: float = 1e4,
    ) -> None:
        super().__init__()
        self.model = model  # shared backbone; NOT a hyperparameter
        # Optional: condition on acquisition via the shared MSContextEncoder, so the teacher's
        # settings are factors rather than baked into the base. None -> context-free warmup.
        # When set, each batch MUST carry its own ms_factors (instrument/detector/fragmentation
        # ids + collision energy) — factors are never fabricated; supply them from the data
        # (streaming sweeps energy per-peptide).
        self.context_encoder = context_encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        # Ties mass_enc onto a stop-gradiented comp_enc (see losses.mod_align_loss); a separate
        # scalar rather than a 4th slot in loss_weights, which already means (ms2, rt, ccs).
        self.mod_align_weight = mod_align_weight
        if (onecycle_max_lr is None) != (onecycle_total_steps is None):
            raise ValueError(
                "onecycle_max_lr and onecycle_total_steps must be provided together"
            )
        if onecycle_max_lr is not None:
            if onecycle_max_lr <= 0:
                raise ValueError("onecycle_max_lr must be positive")
            if onecycle_total_steps < 1:
                raise ValueError("onecycle_total_steps must be positive")
            if not 0.0 <= onecycle_pct_start <= 1.0:
                raise ValueError("onecycle_pct_start must be between 0 and 1")
            if onecycle_div_factor <= 0 or onecycle_final_div_factor <= 0:
                raise ValueError("onecycle_div_factor and onecycle_final_div_factor must be positive")
        self.onecycle_max_lr = onecycle_max_lr
        self.onecycle_total_steps = onecycle_total_steps
        self.onecycle_pct_start = onecycle_pct_start
        self.onecycle_div_factor = onecycle_div_factor
        self.onecycle_final_div_factor = onecycle_final_div_factor

    def transfer_batch_to_device(self, batch: LabeledBatch, device, dataloader_idx: int):
        return batch.to(device)

    def _predict(self, batch: LabeledBatch) -> dict:
        if self.context_encoder is None:
            return self.model(batch.inputs)
        f = batch.ms_factors
        if f is None:
            raise ValueError(
                "context_encoder is set but the batch carries no ms_factors; the dataset must "
                "provide acquisition factors (they are never fabricated)"
            )
        ms_context = self.context_encoder(
            f.instrument_id, f.detector_id, f.fragmentation_id, f.energy
        )
        return self.model.forward(batch.inputs, ms_context=ms_context)

    def training_step(self, batch: LabeledBatch, batch_idx: int) -> torch.Tensor:
        out = self._predict(batch)
        rt_t = self.model.standardize_rt(batch.rt_target)
        ccs_t = self.model.standardize_ccs(batch.ccs_target)
        loss, parts = distill_loss(
            out, batch.ms2_target, rt_t, ccs_t, batch.inputs.frag_mask, self.loss_weights
        )
        if self.mod_align_weight:
            align = mod_align_loss(out["mod_g"], out["mod_m"], batch.inputs.mod_named)
            parts["mod_align"] = float(align.detach())
            loss = loss + self.mod_align_weight * align
        self.log_dict({f"train_{k}": v for k, v in parts.items()}, prog_bar=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: LabeledBatch, batch_idx: int) -> None:
        out = self.model.denormalize(self._predict(batch))
        sa = spectral_angle(out["ms2"], batch.ms2_target, batch.inputs.frag_mask).mean()
        rt_mae = (out["rt"] - batch.rt_target).abs().mean()
        ccs_mae = (out["ccs"] - batch.ccs_target).abs().mean()
        bs = batch.rt_target.shape[0]
        self.log_dict(
            {"val_spectral_angle": sa, "val_rt_mae": rt_mae, "val_ccs_mae": ccs_mae},
            prog_bar=True,
            batch_size=bs,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.onecycle_max_lr is None:
            return optimizer
        scheduler = _CappedOneCycleLR(
            optimizer,
            max_lr=self.onecycle_max_lr,
            total_steps=self.onecycle_total_steps,
            pct_start=self.onecycle_pct_start,
            div_factor=self.onecycle_div_factor,
            final_div_factor=self.onecycle_final_div_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class DistillDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_ds: DistillDataset,
        val_ds: DistillDataset | None,
        batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.seed = seed

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            BatchIterable(self.train_ds, self.batch_size, True, self.seed), batch_size=None
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_ds is None or not len(self.val_ds):
            return None
        return DataLoader(
            BatchIterable(self.val_ds, self.batch_size, False, self.seed), batch_size=None
        )


def fit_distill(
    model: StudentModel,
    train_ds: DistillDataset,
    val_ds: DistillDataset | None,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    grad_clip: float = 1.0,
    seed: int = 0,
    accelerator: str = "auto",
    context_encoder: MSContextEncoder | None = None,
    mod_align_weight: float = 1.0,
    **trainer_kwargs,
) -> DistillModule:
    """Set target norm from ``train_ds`` and run a Lightning distillation fit in place.

    Returns the LightningModule (its ``.model`` is the trained shared backbone). Pass a
    ``context_encoder`` to condition on acquisition (shared with a later real-speclib sink);
    the dataset must then carry per-example ``ms_factors`` (``LabeledBatch.ms_factors``).
    """
    L.seed_everything(seed, verbose=False)
    model.set_norm(*train_ds.rt_ccs_stats())
    module = DistillModule(
        model,
        lr=lr,
        weight_decay=weight_decay,
        loss_weights=loss_weights,
        context_encoder=context_encoder,
        mod_align_weight=mod_align_weight,
    )
    dm = DistillDataModule(train_ds, val_ds, batch_size=batch_size, seed=seed)
    trainer = build_trainer(epochs, accelerator, grad_clip, **trainer_kwargs)
    trainer.fit(module, datamodule=dm)
    return module
