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
    context-free base. Collision energy is never fabricated: a spectrum with no recorded energy
    carries NaN, and the NaN entries of the ``energy`` tensor are masked out per example, so
    that term contributes zero for exactly those rows. (``energy=None`` omits the term for a
    whole call; the real-data path always passes a tensor and relies on the mask, so it is
    reachable only from callers that have no energy axis at all, e.g. tests.)
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
            # Per-example masking, not imputation. A NaN energy means the run recorded none,
            # and it must contribute exactly zero rather than a value we invented.
            #
            # The mask is applied AFTER the MLP on purpose: energy_mlp's first Linear carries a
            # bias, so mlp(0) != 0 — filling missing energy with zero beforehand would inject a
            # learned constant, which is exactly the fabrication this avoids. nan_to_num only
            # stops NaN propagating through the lane that is about to be zeroed.
            present = torch.isfinite(energy)
            term = self.energy_mlp(torch.nan_to_num(energy, nan=0.0).unsqueeze(-1))
            out = out + present.unsqueeze(-1).to(term.dtype) * term
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
        # Per-dataset output affine on the RT head. `emb` above is an ADDITIVE bias in feature
        # space, which can bend the mapping but cannot express a rescale — yet a dataset's raw
        # RT differs from the iRT frame by SCALE as much as offset (gradient length, minutes vs
        # indexed units, dead volume). Both zero-init, so scale=exp(0)=1 and shift=0: the
        # neutral row and an untrained book are exactly identity.
        self.log_scale = nn.Embedding(n_datasets + 1, 1)
        self.shift = nn.Embedding(n_datasets + 1, 1)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.zeros_(self.shift.weight)

    @property
    def context_dim(self) -> int:
        return self.emb.embedding_dim

    @property
    def n_datasets(self) -> int:
        return self.emb.num_embeddings - 1

    def forward(self, dataset_id: torch.Tensor) -> torch.Tensor:
        return self.emb(dataset_id)

    def affine(self, dataset_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(scale, shift)``, each ``(B,)``, for the RT head's output.

        Scale is ``exp(log_scale)``: strictly positive by construction, exactly 1.0 at init,
        and well-behaved multiplicatively. Log space also means weight decay pulls the scale
        toward 1 rather than toward 0, which is the right prior for a rescale.

        Deliberately unclamped — a scale that runs away means the data disagrees with the
        model, and clamping would bury that signal under a value that merely looks poorly fit.
        """
        scale = torch.exp(self.log_scale(dataset_id).squeeze(-1))
        return scale, self.shift(dataset_id).squeeze(-1)

    def neutral(self, n: int, device: torch.device | str) -> torch.Tensor:
        return self.emb(torch.zeros(n, dtype=torch.long, device=device))
