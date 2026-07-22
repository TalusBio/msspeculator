"""Deterministic train/val/test assignment by hashing the stripped sequence.

Hashing (not RNG) means the split is stable across runs, machines, and dataset
growth: a peptide keeps its assignment even if the FASTA changes. All mod-forms and
charge states of one bare sequence land in the same split, so nothing leaks.
"""

from __future__ import annotations

import hashlib

from .config import SplitConfig

Split = str  # "train" | "val" | "test"


def _unit_hash(sequence: str, salt: str) -> float:
    """Map a sequence to a stable float in [0, 1)."""
    digest = hashlib.blake2b(f"{salt}:{sequence}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign_split(sequence: str, cfg: SplitConfig) -> Split:
    """Assign a bare (unmodified) sequence to a split."""
    h = _unit_hash(sequence, cfg.salt)
    if h < cfg.train:
        return "train"
    if h < cfg.train + cfg.val:
        return "val"
    return "test"
