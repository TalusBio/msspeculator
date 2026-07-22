"""Source -> context-vector lookup for acquisition conditioning.

Separated from :class:`StudentModel` on purpose: the backbone consumes context *vectors*
(``ctx_acq`` / ``ctx_lc``), it does not know about sources or ids. This book maps a source
id to its learned vectors. Two independent id spaces so instrument and chromatography vary
independently — swap the instrument (``acq_id``) while keeping the gradient (``lc_id``).

Zero-init: a fresh source starts at the base model. A zero context vector is a strict
no-op (StudentModel's context projections have zero bias, so ctx=0 -> 0 bias), while the
projections' nonzero weights still pass gradient back here so the vectors can learn.
Parameter-efficient fine-tuning = freeze the StudentModel, add a row here, and optimize
only that row's 2*context_dim numbers.

Later this whole module is replaced by a hypernetwork that *generates* the vectors from
metadata (instrument type, collision energy, gradient) — same (ctx_acq, ctx_lc) output, so
nothing downstream changes.
"""

from __future__ import annotations

import torch
from torch import nn


class ContextBook(nn.Module):
    def __init__(self, n_acq: int, n_lc: int, context_dim: int = 16) -> None:
        super().__init__()
        self.acq = nn.Embedding(n_acq, context_dim)
        self.lc = nn.Embedding(n_lc, context_dim)
        nn.init.zeros_(self.acq.weight)
        nn.init.zeros_(self.lc.weight)

    def forward(
        self, acq_id: torch.Tensor, lc_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,) acquisition + chromatography ids -> ((B, ctx), (B, ctx)) vectors."""
        return self.acq(acq_id), self.lc(lc_id)

    def freeze_except(self, acq_ids: list[int] | None = None, lc_ids: list[int] | None = None) -> None:
        """Grad-mask for parameter-efficient fine-tuning: only the listed source rows train.

        Zeros the gradient of every row except those given (per table). Call after
        ``loss.backward()`` each step, or register as a hook; simplest is to gate grads.
        """
        for emb, keep in ((self.acq, acq_ids), (self.lc, lc_ids)):
            if keep is None or emb.weight.grad is None:
                continue
            mask = torch.zeros_like(emb.weight.grad)
            mask[keep] = 1.0
            emb.weight.grad *= mask
