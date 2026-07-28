"""Acquisition/chromatography context generators for :class:`StudentModel` conditioning.

Two modules, one per side of the backbone's context split:

- :class:`MSContextEncoder` composes ``ms_context`` (MS2 side) from acquisition factors —
  instrument, detector, fragmentation (categorical embeddings) plus collision energy
  (continuous, via an MLP) — so the vector is a learned function of metadata shared across
  every source, rather than a per-source id gradient-descended from scratch.
- :class:`ChromRunbook` generates ``chrom_context`` (RT side) from a per-dataset id, row 0
  reserved as the neutral/iRT row.

Both are zero-init: an all-unknown/energy-less input or the neutral runbook row reproduces
the base (context-free) model exactly (:class:`StudentModel`'s context projections have zero
bias, so ctx=0 -> 0 bias), while their nonzero weights still pass gradient back so the
vectors can learn. Parameter-efficient fine-tuning = freeze the StudentModel and optimize
only these modules.
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
    energy (continuous, via a plain MLP whose first ``Linear`` IS the learned affine — no
    BatchNorm1d: a single-NCE batch would collapse a batch-statistic normalization to a
    degenerate center/scale, so the affine is a trained parameter instead). Every term is
    zero-init, so an all-unknown / energy-less input returns the zero vector — the
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


class ChromRunbook(nn.Module):
    """Per-dataset chromatography context for the RT head. One embedding keyed by dataset with
    row 0 reserved as the iRT / neutral row (context-free). Zero-init, so an untrained book and
    the neutral row both reproduce the base (iRT) RT; other rows learn each dataset's LC offset.
    """

    def __init__(self, n_datasets: int, context_dim: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_datasets + 1, context_dim)  # +1: index 0 = neutral (iRT)
        nn.init.zeros_(self.emb.weight)

    @property
    def context_dim(self) -> int:
        return self.emb.embedding_dim

    @property
    def n_datasets(self) -> int:
        return self.emb.num_embeddings - 1

    def forward(self, dataset_id: torch.Tensor) -> torch.Tensor:
        return self.emb(dataset_id)

    def neutral(self, n: int, device: torch.device | str) -> torch.Tensor:
        return self.emb(torch.zeros(n, dtype=torch.long, device=device))
