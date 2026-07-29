"""Turn precursors into padded tensors for the student model.

Encoding is deliberately minimal (the whole point of distillation here): per-residue
amino-acid index, a charge scalar, and modifications split into four channels — element
composition (``mod_comp``), raw mass (``mod_mass``), and two boolean masks (``mod_present``,
``mod_named``) — so a modification can be routed through a compositional or a mass-only
encoder. No LSTM means everything is a fixed-shape, batch-parallel tensor.

Peptides are always wrapped with explicit N-/C-terminal tokens:

    [N] r1 r2 ... rL [C]

The termini give small models fixed anchors and extra context capacity, and serve as carriers
for terminal modifications. They are NOT residues: they carry no mass and are excluded from
MS2 fragment sites. Layout is position 0 = N-term, 1..L = residues, L+1 = C-term, then padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pepdistill_rs import (  # noqa: F401
    AA_OFFSET,
    CTERM_IDX,
    FRAG_OFFSET,
    NTERM_IDX,
    PAD_IDX,
    N_TOKENS,
)

from .precursors import Precursor

# Vocab/token layout (AA_OFFSET, PAD_IDX, NTERM_IDX, CTERM_IDX, N_TOKENS) is single-sourced
# in Rust (pepdistill_rs); re-exported above for existing importers (student.py, fast.py).
#
# FRAG_OFFSET is the same contract for the MS2 axis: with the mandatory N-term token at column
# 0, adjacent-pool row 0 pools [N] with residue 1, so the L-1 real inter-residue fragment sites
# start at row FRAG_OFFSET. Every slice of an MS2 output — here, predict/fast.py,
# predict/library.py, distill/dataset.py, and the Rust runtime — reads this one constant, so
# the torch and Rust paths cannot drift into off-by-one disagreement.

@dataclass(slots=True)
class Batch:
    tokens: torch.Tensor  # (B, T) long, T = maxL + 2
    mod_comp: torch.Tensor  # (B, T, 6) float — element composition delta
    mod_mass: torch.Tensor  # (B, T) float — mass delta in Daltons, unscaled
    mod_present: torch.Tensor  # (B, T) bool — any modification here
    mod_named: torch.Tensor  # (B, T) bool — composition known, so routable to comp_enc
    charge: torch.Tensor  # (B,) long
    lengths: torch.Tensor  # (B,) long, residue count L (excludes termini)
    pad_mask: torch.Tensor  # (B, T) bool, True where padded
    frag_mask: torch.Tensor  # (B, T-1) bool, True at valid inter-residue fragment sites

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(*(t.to(device) for t in (
            self.tokens, self.mod_comp, self.mod_mass, self.mod_present, self.mod_named,
            self.charge, self.lengths, self.pad_mask, self.frag_mask,
        )))


def collate(precursors: list[Precursor]) -> Batch:
    """Pad a list of precursors into a single :class:`Batch` (delegates to the ext).

    Layout is always [N] r1..rL [C]; residues sit at positions 1..L and the first fragment
    site is adjacent-pool index 1.
    """
    import pepdistill_rs as _rs

    peptides = [p.peptide for p in precursors]
    charges = [int(p.charge) for p in precursors]
    a = _rs.collate(peptides, charges)
    return Batch(
        torch.from_numpy(a["tokens"]), torch.from_numpy(a["mod_comp"]),
        torch.from_numpy(a["mod_mass"]), torch.from_numpy(a["mod_present"]),
        torch.from_numpy(a["mod_named"]), torch.from_numpy(a["charge"]),
        torch.from_numpy(a["lengths"]), torch.from_numpy(a["pad_mask"]),
        torch.from_numpy(a["frag_mask"]),
    )
