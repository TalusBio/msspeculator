"""Deterministic train/val/test assignment by hashing the stripped sequence.

Hashing (not RNG) means the split is stable across runs, machines, and dataset
growth: a peptide keeps its assignment even if the FASTA changes. All mod-forms and
charge states of one bare sequence land in the same split, so nothing leaks.

The hash itself lives in Rust (``pepdistill_core::split``) because a library is split there
during a context fit while the corpus is split here, and two implementations that disagree would
put a peptide the model trained on into a held-out score with nothing failing. A reference
implementation of the hash is kept in ``tests/test_data.py`` purely to pin the port.
"""

from __future__ import annotations

import pepdistill_rs

from .config import SplitConfig

Split = str  # "train" | "val" | "test"


def assign_split(sequence: str, cfg: SplitConfig) -> Split:
    """Assign a bare (unmodified) sequence to a split."""
    return pepdistill_rs.assign_split(sequence, cfg.salt, cfg.train, cfg.val)
