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

DEFAULT_INSTRUMENTS = ("unknown", "Lumos", "QExactive", "Exploris", "timsTOF")
DEFAULT_DETECTORS = ("unknown", "FTMS", "ITMS", "TOF")
DEFAULT_FRAGMENTATIONS = ("unknown", "HCD", "CID", "ETD", "EThcD")


class MSContextEncoder(nn.Module):
    """Compose ``ms_context`` (MS2 side) from acquisition factors: instrument, detector,
    fragmentation (categorical embeddings, index 0 = unknown/blank -> zero row) plus collision
    energy (continuous, BatchNorm1d -> MLP so the normalization is LEARNED, not a fixed center).
    Every term is zero-init, so an all-unknown / energy-less input returns the zero vector — the
    context-free base. Collision energy is never fabricated: pass ``energy=None`` to omit it.
    """

    def __init__(
        self,
        context_dim: int = 16,
        instruments: tuple[str, ...] = DEFAULT_INSTRUMENTS,
        detectors: tuple[str, ...] = DEFAULT_DETECTORS,
        fragmentations: tuple[str, ...] = DEFAULT_FRAGMENTATIONS,
    ) -> None:
        super().__init__()
        self.instruments = tuple(instruments)
        self.detectors = tuple(detectors)
        self.fragmentations = tuple(fragmentations)
        self._inst_ix = {n: i for i, n in enumerate(self.instruments)}
        self._det_ix = {n: i for i, n in enumerate(self.detectors)}
        self._frag_ix = {n: i for i, n in enumerate(self.fragmentations)}
        self.inst_emb = nn.Embedding(len(self.instruments), context_dim)
        self.det_emb = nn.Embedding(len(self.detectors), context_dim)
        self.frag_emb = nn.Embedding(len(self.fragmentations), context_dim)
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, context_dim), nn.GELU(), nn.Linear(context_dim, context_dim)
        )
        for emb in (self.inst_emb, self.det_emb, self.frag_emb):
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.energy_mlp[-1].weight)  # energy term starts neutral (0)
        nn.init.zeros_(self.energy_mlp[-1].bias)

    @property
    def context_dim(self) -> int:
        return self.inst_emb.embedding_dim

    def instrument_id(self, name: str) -> int:
        return self._inst_ix.get(name, 0)

    def detector_id(self, name: str) -> int:
        return self._det_ix.get(name, 0)

    def fragmentation_id(self, name: str) -> int:
        return self._frag_ix.get(name, 0)

    def forward(
        self,
        instrument_id: torch.Tensor,
        detector_id: torch.Tensor,
        fragmentation_id: torch.Tensor,
        energy: torch.Tensor | None,
    ) -> torch.Tensor:
        out = (
            self.inst_emb(instrument_id)
            + self.det_emb(detector_id)
            + self.frag_emb(fragmentation_id)
        )
        if energy is not None:
            out = out + self.energy_mlp(energy.unsqueeze(-1))  # first Linear = learned affine
        return out


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

    def freeze_except(
        self, acq_ids: list[int] | None = None, lc_ids: list[int] | None = None
    ) -> None:
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


# Default acquisition vocab (index 0 = "unknown" -> zero-init embedding -> no-op). Covers the
# PROSPECT analyzers/fragmentations and the peptdeep teacher (FTMS/HCD). Fixed (not
# data-derived) so one encoder is shared across pretrain + every real run, and unseen values
# fall back to "unknown" rather than shifting the index space.
DEFAULT_ANALYZERS = ("unknown", "FTMS", "ITMS", "TOF")
DEFAULT_FRAGMENTATIONS = ("unknown", "HCD", "CID", "ETD", "EThcD")


class ContextEncoder(nn.Module):
    """Generate ``ctx_acq`` from acquisition factors: analyzer + fragmentation (categorical) and
    collision energy (continuous). Composed additively — the ``INSTR::FRAG::CE`` grammar.

    "Context from metadata": ``ctx_acq`` is a learned function of the acquisition settings, so
    the teacher (a known analyzer/frag/NCE) and every real run share ONE factor space — unseen
    CEs interpolate, unseen categoricals fall back to "unknown", and the teacher stops silently
    anchoring the base. CE is absolute NCE (teacher ``nce`` and PROSPECT ``*_collision_energy``
    agree). Analyzer/frag ids come from :meth:`analyzer_id`/:meth:`frag_id`.

    Zero-init -> ``ctx_acq`` = 0 for every input -> exact base model (the acq projection has
    zero bias); factor dependence is learned from data, so this strictly generalizes the
    no-context path. RT/chromatography is NOT here — keep ``ctx_lc`` on a per-run ``ContextBook``.
    """

    def __init__(
        self,
        context_dim: int = 16,
        ce_center: float = 30.0,
        ce_scale: float = 10.0,
        analyzers: tuple[str, ...] = DEFAULT_ANALYZERS,
        fragmentations: tuple[str, ...] = DEFAULT_FRAGMENTATIONS,
    ) -> None:
        super().__init__()
        self.ce_center = ce_center
        self.ce_scale = ce_scale
        self.analyzers = tuple(analyzers)
        self.fragmentations = tuple(fragmentations)
        self._ana_ix = {n: i for i, n in enumerate(self.analyzers)}
        self._frag_ix = {n: i for i, n in enumerate(self.fragmentations)}
        self.proj = nn.Linear(1, context_dim)  # collision energy
        self.ana_emb = nn.Embedding(len(self.analyzers), context_dim)
        self.frag_emb = nn.Embedding(len(self.fragmentations), context_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.ana_emb.weight)
        nn.init.zeros_(self.frag_emb.weight)

    def analyzer_id(self, name: str) -> int:
        """Vocab id for a mass-analyzer name; unknown -> 0 (no-op)."""
        return self._ana_ix.get(name, 0)

    def frag_id(self, name: str) -> int:
        """Vocab id for a fragmentation name; unknown -> 0 (no-op)."""
        return self._frag_ix.get(name, 0)

    def forward(
        self,
        ce: torch.Tensor,
        analyzer_id: torch.Tensor | None = None,
        frag_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B,) CE (+ optional (B,) analyzer/frag ids) -> (B, context_dim) ``ctx_acq``."""
        out = self.proj(((ce - self.ce_center) / self.ce_scale).unsqueeze(-1))
        if analyzer_id is not None:
            out = out + self.ana_emb(analyzer_id)
        if frag_id is not None:
            out = out + self.frag_emb(frag_id)
        return out

    def encode_batch(
        self, ce: torch.Tensor, analyzer: str, fragmentation: str, device
    ) -> torch.Tensor:
        """(B,) CE + one fixed analyzer/fragmentation NAME -> (B, context_dim) ``ctx_acq``.

        The single entry point for a batch sharing one acquisition instrument/fragmentation
        (the teacher's fixed setting, or a CLI ``--ms-context``): owns the name->id lookup and
        the broadcast so callers never re-roll it. Per-example CE still varies via ``ce``.
        """
        n = ce.shape[0]
        ana = torch.full((n,), self.analyzer_id(analyzer), device=device, dtype=torch.long)
        frag = torch.full((n,), self.frag_id(fragmentation), device=device, dtype=torch.long)
        return self(ce, ana, frag)
