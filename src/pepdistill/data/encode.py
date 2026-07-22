"""Turn precursors into padded tensors for the student model.

Encoding is deliberately minimal (the whole point of distillation here): per-residue
amino-acid index + a single modification-mass channel, plus a charge scalar. No LSTM
means everything is a fixed-shape, batch-parallel tensor.

Each peptide is wrapped with explicit N-terminal and C-terminal tokens:

    [N] r1 r2 ... rL [C]

The termini give small models fixed anchors and extra context capacity. They are NOT
residues: they carry no mass and are excluded from MS2 fragment sites. Token layout per
sample is position 0 = N-term, 1..L = residues, L+1 = C-term, then padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..chem import MOD_DELTA
from .precursors import Precursor

# Amino-acid token id = ord(aa) - ord('A'), so A=0 .. Z=25. No lookup table: this is a
# vectorized subtract on the raw sequence bytes (np.frombuffer view), the exact op the
# future Rust tokenizer will do when it packs the numpy array directly. Indices 26..28 sit
# just past 'Z': 26 = padding, 27/28 = the N/C-term tokens. Contract is soft (greenfield) —
# change freely, but keep encode/decode/fast in sync via these constants.
AA_OFFSET = ord("A")  # 65
AA_VOCAB: dict[str, int] = {aa: ord(aa) - AA_OFFSET for aa in "ACDEFGHIKLMNPQRSTVWY"}
PAD_IDX = 26
NTERM_IDX = 27
CTERM_IDX = 28
N_TOKENS = 29
# Scale mod deltas into a friendly range for the network.
MOD_SCALE = 100.0

# Whether to wrap peptides with N/C-term tokens. Default off: A/B on random and real
# (E. coli) peptides showed no accuracy benefit (pad_mask already encodes the boundary),
# and the 2 extra tokens cost ~12% sequence length. Kept as a runtime toggle so a future
# CLS-readout variant (read RT/CCS *from* the term tokens) can be tried. All encode/decode
# paths read frag_offset() so fragment-site indices stay consistent either way.
_USE_TERMINI = False


def use_termini() -> bool:
    return _USE_TERMINI


def set_termini(value: bool) -> None:
    global _USE_TERMINI
    _USE_TERMINI = value


def frag_offset() -> int:
    """Adjacent-pool index of the first inter-residue fragment site (1 with termini)."""
    return 1 if _USE_TERMINI else 0


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


def _mod_delta_vector(prec: Precursor, length: int) -> list[float]:
    vec = [0.0] * length
    for site, name in prec.peptide.mods:
        vec[site] += MOD_DELTA[name] / MOD_SCALE
    return vec


def collate(precursors: list[Precursor]) -> Batch:
    """Pad a list of precursors into a single :class:`Batch`.

    With termini, layout is [N] r1..rL [C]; residues sit at positions 1..L and the first
    fragment site is adjacent-pool index 1. Without termini, residues start at 0.
    """
    off = frag_offset()  # 1 with termini, else 0
    extra = 2 if use_termini() else 0
    lengths = [p.peptide.length for p in precursors]
    max_len = max(lengths)
    tok_len = max_len + extra
    b = len(precursors)

    tokens = torch.full((b, tok_len), PAD_IDX, dtype=torch.long)
    mod_delta = torch.zeros(b, tok_len, dtype=torch.float32)
    charge = torch.tensor([p.charge for p in precursors], dtype=torch.long)
    length_t = torch.tensor(lengths, dtype=torch.long)
    pad_mask = torch.ones(b, tok_len, dtype=torch.bool)
    frag_mask = torch.zeros(b, tok_len - 1, dtype=torch.bool)

    for i, prec in enumerate(precursors):
        seq = prec.peptide.sequence
        n = len(seq)
        if use_termini():
            tokens[i, 0] = NTERM_IDX
            tokens[i, 1 + n] = CTERM_IDX
        tokens[i, off : off + n] = torch.tensor([ord(a) - AA_OFFSET for a in seq], dtype=torch.long)
        mod_delta[i, off : off + n] = torch.tensor(_mod_delta_vector(prec, n), dtype=torch.float32)
        pad_mask[i, : n + extra] = False
        # Inter-residue site between residue j and j+1 is adjacent-pool index off+j-1 for
        # j=1..n-1, i.e. valid range [off, off+n-1).
        frag_mask[i, off : off + n - 1] = True

    return Batch(tokens, mod_delta, charge, length_t, pad_mask, frag_mask)
