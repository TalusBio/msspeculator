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
from torch.utils.data import DataLoader, IterableDataset

from ..models.student import StudentModel
from .dataset import DistillDataset, LabeledBatch
from .losses import distill_loss, spectral_angle


class DistillModule(L.LightningModule):
    """Distillation regime: fit the student to teacher soft labels (MS2 + RT + CCS)."""

    def __init__(
        self,
        model: StudentModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        context_encoder=None,
        distill_fallback_ce: float = 30.0,
    ) -> None:
        super().__init__()
        self.model = model  # shared backbone; NOT a hyperparameter
        # Optional: give the teacher its OWN acquisition context (from its NCE) via the shared
        # ContextEncoder, so the teacher's collision energy is a factor rather than baked into
        # the base. None -> context-free warmup (base = teacher condition).
        self.context_encoder = context_encoder
        # Fallback CE for the encoder when a batch carries no per-example CE (cached distill,
        # whose labels are all at one teacher NCE). Streaming supplies per-batch CE instead.
        self.distill_fallback_ce = distill_fallback_ce
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights

    def transfer_batch_to_device(self, batch: LabeledBatch, device, dataloader_idx: int):
        return batch.to(device)

    def _predict(self, batch: LabeledBatch) -> dict:
        if self.context_encoder is None:
            return self.model(batch.inputs)
        # Per-batch CE (streaming NCE sweep) if provided, else the constant fallback.
        ce = batch.ce if batch.ce is not None else torch.full(
            (batch.inputs.tokens.shape[0],), self.distill_fallback_ce, device=self.device
        )
        return self.model.forward_context(batch.inputs, ctx_acq=self.context_encoder(ce), ctx_lc=None)

    def training_step(self, batch: LabeledBatch, batch_idx: int) -> torch.Tensor:
        out = self._predict(batch)
        rt_t = (batch.rt_target - self.model.rt_mean) / self.model.rt_std
        ccs_t = (batch.ccs_target - self.model.ccs_mean) / self.model.ccs_std
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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class _BatchIterable(IterableDataset):
    """Yields ready-made :class:`LabeledBatch` objects from a DistillDataset.

    Batching/collation already live in ``DistillDataset.batches``; we wrap them as an
    IterableDataset so a ``DataLoader(batch_size=None)`` passes each LabeledBatch straight
    through. ``__iter__`` runs per epoch, so training reshuffles each epoch.
    """

    def __init__(self, ds: DistillDataset, batch_size: int, shuffle: bool, seed: int) -> None:
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        yield from self.ds.batches(self.batch_size, self.shuffle, gen)


class DistillDataModule(L.LightningDataModule):
    def __init__(
        self, train_ds: DistillDataset, val_ds: DistillDataset | None, batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.seed = seed

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            _BatchIterable(self.train_ds, self.batch_size, True, self.seed), batch_size=None
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_ds is None or not len(self.val_ds):
            return None
        return DataLoader(
            _BatchIterable(self.val_ds, self.batch_size, False, self.seed), batch_size=None
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
    distill_fallback_ce: float = 30.0,
    **trainer_kwargs,
) -> DistillModule:
    """Set target norm from ``train_ds`` and run a Lightning distillation fit in place.

    Returns the LightningModule (its ``.model`` is the trained shared backbone). Pass a
    ``context_encoder`` to give the teacher its own CE-driven acquisition context (shared with
    a later real-speclib sink); ``distill_fallback_ce`` is the fixed NCE those cached labels
    were generated at, fed to the encoder for every batch.
    """
    L.seed_everything(seed, verbose=False)
    model.set_norm(*train_ds.rt_ccs_stats())
    module = DistillModule(
        model, lr=lr, weight_decay=weight_decay, loss_weights=loss_weights,
        context_encoder=context_encoder, distill_fallback_ce=distill_fallback_ce,
    )
    dm = DistillDataModule(train_ds, val_ds, batch_size=batch_size, seed=seed)
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        gradient_clip_val=grad_clip,
        enable_checkpointing=trainer_kwargs.pop("enable_checkpointing", False),
        logger=trainer_kwargs.pop("logger", False),
        **trainer_kwargs,
    )
    trainer.fit(module, datamodule=dm)
    return module
