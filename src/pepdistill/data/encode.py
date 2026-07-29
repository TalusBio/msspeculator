"""Turn precursors into padded tensors for the student model.

Encoding is deliberately minimal (the whole point of distillation here): per-residue
amino-acid index + a single modification-mass channel, plus a charge scalar. No LSTM
means everything is a fixed-shape, batch-parallel tensor.

Peptides are always wrapped with explicit N-/C-terminal tokens:

    [N] r1 r2 ... rL [C]

The termini give small models fixed anchors and extra context capacity, and (in a later task)
serve as carriers for terminal modifications. They are NOT residues: they carry no mass and are
excluded from MS2 fragment sites. Layout is position 0 = N-term, 1..L = residues, L+1 = C-term,
then padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pepdistill_rs import AA_OFFSET, CTERM_IDX, MOD_SCALE, NTERM_IDX, PAD_IDX, N_TOKENS  # noqa: F401

from .precursors import Precursor

# Vocab/token layout (AA_OFFSET, PAD_IDX, NTERM_IDX, CTERM_IDX, N_TOKENS) and MOD_SCALE
# are single-sourced in Rust (pepdistill_rs); re-exported above for existing importers
# (student.py, fast.py).

@dataclass(slots=True)
class Batch:
    tokens: torch.Tensor  # (B, T) long, T = maxL + 2
    mod_delta: torch.Tensor  # (B, T) float
    charge: torch.Tensor  # (B,) long
    lengths: torch.Tensor  # (B,) long, residue count L (excludes termini)
    pad_mask: torch.Tensor  # (B, T) bool, True where padded
    frag_mask: torch.Tensor  # (B, T-1) bool, True at valid inter-residue fragment sites

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            self.tokens.to(device),
            self.mod_delta.to(device),
            self.charge.to(device),
            self.lengths.to(device),
            self.pad_mask.to(device),
            self.frag_mask.to(device),
        )


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
        torch.from_numpy(a["tokens"]), torch.from_numpy(a["mod_delta"]),
        torch.from_numpy(a["charge"]), torch.from_numpy(a["lengths"]),
        torch.from_numpy(a["pad_mask"]), torch.from_numpy(a["frag_mask"]),
    )
