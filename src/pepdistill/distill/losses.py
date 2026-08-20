"""Distillation losses and evaluation metrics for MS2 / RT / CCS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_masked(
    pred: torch.Tensor, target: torch.Tensor, frag_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten (B, F, n_ion) -> (B, F*n_ion), zeroing padded fragment positions."""
    m = frag_mask.unsqueeze(-1).float()  # (B, F, 1)
    b = pred.shape[0]
    return (pred * m).reshape(b, -1), (target * m).reshape(b, -1)


def ms2_cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, frag_mask: torch.Tensor
) -> torch.Tensor:
    """1 - cosine similarity, averaged over the batch (padded peaks masked out)."""
    p, t = _flatten_masked(pred, target, frag_mask)
    cos = F.cosine_similarity(p, t, dim=1, eps=1e-8)
    return (1.0 - cos).mean()


def spectral_angle(
    pred: torch.Tensor, target: torch.Tensor, frag_mask: torch.Tensor
) -> torch.Tensor:
    """Normalized spectral contrast angle in [0, 1] (1 = identical). Reporting metric."""
    p, t = _flatten_masked(pred, target, frag_mask)
    cos = F.cosine_similarity(p, t, dim=1, eps=1e-8).clamp(-1.0, 1.0)
    return 1.0 - 2.0 * torch.arccos(cos) / torch.pi


def distill_loss(
    out: dict[str, torch.Tensor],
    ms2_target: torch.Tensor,
    rt_target_std: torch.Tensor,
    ccs_target_std: torch.Tensor,
    frag_mask: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted MS2(cosine) + RT(MSE) + CCS(MSE). rt/ccs targets are pre-standardized.

    A zero-weight term is SKIPPED entirely (not multiplied): sources without a property
    (e.g. PROSPECT has no CCS) pass NaN targets, and ``0 * NaN = NaN`` would otherwise
    poison the shared-trunk gradients. Weight 0 == "not supervised here".
    """
    w_ms2, w_rt, w_ccs = weights
    terms = {
        "ms2": (w_ms2, lambda: ms2_cosine_loss(out["ms2"], ms2_target, frag_mask)),
        "rt": (w_rt, lambda: F.mse_loss(out["rt"], rt_target_std)),
        "ccs": (w_ccs, lambda: F.mse_loss(out["ccs"], ccs_target_std)),
    }
    total = None
    parts: dict[str, float] = {}
    for name, (w, fn) in terms.items():
        if not w:  # unsupervised here — skip so NaN targets can't poison gradients
            continue
        loss = fn()
        parts[name] = float(loss.detach())
        total = w * loss if total is None else total + w * loss
    if total is None:
        raise ValueError("distill_loss: all term weights are zero")
    parts["total"] = float(total.detach())
    return total, parts


def labeled_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over the rows whose target is finite.

    Masked per row rather than per batch because the RT labels are per source: a spectral
    library reports its own run's retention time and no iRT, so requiring both would throw
    away that row's MS2 as well. Dropping the row is the strictly worse trade -- partial
    supervision on three heads beats none on all three.

    NaN targets are substituted before the subtraction, not masked after it: ``0 * NaN`` is
    still NaN, so a masked NaN residual would poison the shared trunk's gradients anyway.

    >>> labeled_mse(torch.tensor([1.0, 5.0]), torch.tensor([0.0, float("nan")]))
    tensor(1.)
    """
    labeled = torch.isfinite(target)
    n = labeled.sum()
    if int(n) == 0:
        # A batch drawn entirely from a source without this label. Zero, graph-connected, so
        # callers add the term unconditionally; the weighted sum stays differentiable.
        return prediction.sum() * 0.0
    residual = prediction - torch.where(labeled, target, torch.zeros_like(target))
    return ((residual * labeled) ** 2).sum() / n


def mod_align_loss(
    mod_g: torch.Tensor, mod_m: torch.Tensor, mod_has_composition: torch.Tensor
) -> torch.Tensor:
    """Pull the mass-only encoder onto the compositional encoder's shared space.

    MSE between ``mod_m`` and a stop-gradiented ``mod_g``, over sites where a composition is
    actually known. The stop-gradient is one-directional on purpose: ``g`` is shaped only by
    the prediction task, and without it ``g`` could shrink toward zero to make alignment cheap.

    Masked to named sites because unmodified positions are trivially aligned (both encoders
    are zeroed there) and would otherwise dominate the mean, which is almost all of any batch.
    """
    mask = mod_has_composition.unsqueeze(-1).expand_as(mod_m).to(mod_m.dtype)
    n = mask.sum()
    if float(n) == 0.0:
        # No supervision available. Return a graph-connected zero so callers can add it
        # unconditionally without a NaN or a detached-tensor surprise.
        return mod_m.sum() * 0.0
    return (((mod_m - mod_g.detach()) * mask) ** 2).sum() / n
