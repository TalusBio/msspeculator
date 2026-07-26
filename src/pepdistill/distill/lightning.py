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

from ..models.student import StudentModel
from .dataset import BatchIterable, DistillDataset, LabeledBatch
from .losses import distill_loss, spectral_angle


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
        context_encoder=None,
        distill_analyzer: str = "FTMS",
        distill_fragmentation: str = "HCD",
    ) -> None:
        super().__init__()
        self.model = model  # shared backbone; NOT a hyperparameter
        # Optional: condition on acquisition via the shared ContextEncoder, so the teacher's
        # settings are factors rather than baked into the base. None -> context-free warmup.
        # When set, each batch MUST carry its own collision energy (batch.ce) — CE is never
        # fabricated; supply it from the data (streaming sweeps it per-peptide).
        self.context_encoder = context_encoder
        # The teacher's fixed analyzer/fragmentation (peptdeep defaults: Orbitrap FTMS + HCD).
        self.distill_analyzer = distill_analyzer
        self.distill_fragmentation = distill_fragmentation
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights

    def transfer_batch_to_device(self, batch: LabeledBatch, device, dataloader_idx: int):
        return batch.to(device)

    def _predict(self, batch: LabeledBatch) -> dict:
        if self.context_encoder is None:
            return self.model(batch.inputs)
        if batch.ce is None:
            raise ValueError(
                "context_encoder is set but the batch carries no collision energy; the dataset "
                "must provide per-example CE (it is never fabricated)"
            )
        ctx_acq = self.context_encoder.encode_batch(
            batch.ce, self.distill_analyzer, self.distill_fragmentation, self.device
        )
        return self.model.forward_context(batch.inputs, ctx_acq=ctx_acq, ctx_lc=None)

    def training_step(self, batch: LabeledBatch, batch_idx: int) -> torch.Tensor:
        out = self._predict(batch)
        rt_t = self.model.standardize_rt(batch.rt_target)
        ccs_t = self.model.standardize_ccs(batch.ccs_target)
        loss, parts = distill_loss(
            out, batch.ms2_target, rt_t, ccs_t, batch.inputs.frag_mask, self.loss_weights
        )
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


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
    context_encoder=None,
    distill_analyzer: str = "FTMS",
    distill_fragmentation: str = "HCD",
    **trainer_kwargs,
) -> DistillModule:
    """Set target norm from ``train_ds`` and run a Lightning distillation fit in place.

    Returns the LightningModule (its ``.model`` is the trained shared backbone). Pass a
    ``context_encoder`` to condition on acquisition (shared with a later real-speclib sink);
    the dataset must then carry per-example collision energy (``LabeledBatch.ce``).
    ``distill_analyzer``/``distill_fragmentation`` are the teacher's fixed acquisition factors.
    """
    L.seed_everything(seed, verbose=False)
    model.set_norm(*train_ds.rt_ccs_stats())
    module = DistillModule(
        model,
        lr=lr,
        weight_decay=weight_decay,
        loss_weights=loss_weights,
        context_encoder=context_encoder,
        distill_analyzer=distill_analyzer,
        distill_fragmentation=distill_fragmentation,
    )
    dm = DistillDataModule(train_ds, val_ds, batch_size=batch_size, seed=seed)
    trainer = build_trainer(epochs, accelerator, grad_clip, **trainer_kwargs)
    trainer.fit(module, datamodule=dm)
    return module
