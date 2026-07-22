"""The distillation training loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from ..models.student import StudentModel
from .dataset import DistillDataset, LabeledBatch
from .losses import distill_loss, spectral_angle


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # ms2, rt, ccs
    device: str = "cpu"
    seed: int = 0
    grad_clip: float = 1.0


def _standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


@torch.no_grad()
def evaluate(model: StudentModel, ds: DistillDataset, cfg: TrainConfig) -> dict[str, float]:
    """Held-out metrics in native units: spectral angle, RT MAE, CCS MAE."""
    model.eval()
    gen = torch.Generator().manual_seed(cfg.seed)
    device = cfg.device
    sa_sum = rt_ae = ccs_ae = 0.0
    n = 0
    for batch in ds.batches(cfg.batch_size, shuffle=False, generator=gen):
        batch = batch.to(device)
        out = model.denormalize(model(batch.inputs))
        bs = batch.rt_target.shape[0]
        sa_sum += float(spectral_angle(out["ms2"], batch.ms2_target, batch.inputs.frag_mask).sum())
        rt_ae += float((out["rt"] - batch.rt_target).abs().sum())
        ccs_ae += float((out["ccs"] - batch.ccs_target).abs().sum())
        n += bs
    n = max(n, 1)
    return {"spectral_angle": sa_sum / n, "rt_mae": rt_ae / n, "ccs_mae": ccs_ae / n}


def train(
    model: StudentModel,
    train_ds: DistillDataset,
    val_ds: DistillDataset | None,
    cfg: TrainConfig,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, float]]:
    """Train in place. Returns per-epoch history (train loss parts + val metrics).

    ``log`` (if given) is called once per epoch with a one-line progress string.
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    model.to(device)

    rt_mean, rt_std, ccs_mean, ccs_std = train_ds.rt_ccs_stats()
    model.set_norm(rt_mean, rt_std, ccs_mean, ccs_std)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        running: dict[str, float] = {}
        n_batches = 0
        for batch in train_ds.batches(cfg.batch_size, shuffle=True, generator=gen):
            batch = batch.to(device)
            out = model(batch.inputs)
            rt_std_tgt = _standardize(batch.rt_target, model.rt_mean, model.rt_std)
            ccs_std_tgt = _standardize(batch.ccs_target, model.ccs_mean, model.ccs_std)
            loss, parts = distill_loss(
                out,
                batch.ms2_target,
                rt_std_tgt,
                ccs_std_tgt,
                batch.inputs.frag_mask,
                cfg.loss_weights,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

        n_batches = max(n_batches, 1)
        rec = {f"train_{k}": v / n_batches for k, v in running.items()}
        rec["epoch"] = epoch
        if val_ds is not None and len(val_ds):
            for k, v in evaluate(model, val_ds, cfg).items():
                rec[f"val_{k}"] = v
        history.append(rec)
        if log:
            log(
                f"epoch {epoch + 1}/{cfg.epochs} "
                f"train_total={rec.get('train_total', float('nan')):.3f} "
                f"val_SA={rec.get('val_spectral_angle', float('nan')):.3f} "
                f"rt_mae={rec.get('val_rt_mae', float('nan')):.4f} "
                f"ccs_mae={rec.get('val_ccs_mae', float('nan')):.2f}"
            )
    return history


def train_streaming(
    model: StudentModel,
    batches: Iterable[LabeledBatch],
    total_batches: int,
    cfg: TrainConfig,
    val_ds: DistillDataset | None = None,
    eval_every: int = 200,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, float]]:
    """Train from a live stream of pre-labeled batches (norm must be set beforehand).

    Steps once per incoming batch. Every ``eval_every`` steps it records train-loss and
    (optional) val metrics into the returned history.
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict[str, float]] = []
    running: dict[str, float] = {}
    since = 0

    for step, batch in enumerate(batches):
        if step >= total_batches:
            break
        batch = batch.to(device)
        out = model(batch.inputs)
        rt_std_tgt = _standardize(batch.rt_target, model.rt_mean, model.rt_std)
        ccs_std_tgt = _standardize(batch.ccs_target, model.ccs_mean, model.ccs_std)
        loss, parts = distill_loss(
            out, batch.ms2_target, rt_std_tgt, ccs_std_tgt, batch.inputs.frag_mask, cfg.loss_weights
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + v
        since += 1

        if (step + 1) % eval_every == 0 or step + 1 == total_batches:
            rec = {"step": step + 1, **{f"train_{k}": v / since for k, v in running.items()}}
            running, since = {}, 0
            if val_ds is not None and len(val_ds):
                model.eval()
                for k, v in evaluate(model, val_ds, cfg).items():
                    rec[f"val_{k}"] = v
                model.train()
            history.append(rec)
            if log:
                log(
                    f"step {rec['step']}/{total_batches} "
                    f"train_total={rec.get('train_total', float('nan')):.3f} "
                    f"val_SA={rec.get('val_spectral_angle', float('nan')):.3f}"
                )
    return history
