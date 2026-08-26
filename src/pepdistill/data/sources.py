"""Sequence sources for streaming distillation: random generator and FASTA digests.

Both yield bare (unmodified) peptide sequences forever. :func:`precursors_from_sequences`
turns a batch of sequences into concrete precursors by sampling one mod-form + charge per
sequence, maximizing sequence diversity per (slow) teacher call.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..chem import Peptide
from .config import DigestConfig
from .digest import cleave_protein, parse_fasta
from .mod_rules import fixed_sites, sampled_sites
from .precursors import Precursor

_AA = "ACDEFGHIKLMNPQRSTVWY"
# Rough natural abundances so random peptides are not uniform noise.
_AA_W = np.array(
    [8, 1.4, 5.5, 6.8, 4, 7, 2.3, 6, 5.8, 10, 2.4, 4, 4.7, 4, 5.5, 6.6, 5.4, 7, 1, 3],
    dtype=np.float64,
)
_AA_W /= _AA_W.sum()


def random_peptide_stream(
    rng: np.random.Generator, min_len: int = 7, max_len: int = 30
) -> Iterator[str]:
    """Infinite stream of tryptic-looking random peptides (C-terminal K/R)."""
    aa = np.array(list(_AA))
    while True:
        n = int(rng.integers(min_len, max_len + 1))
        body = "".join(rng.choice(aa, size=n - 1, p=_AA_W))
        yield body + ("K" if rng.random() < 0.5 else "R")


def fasta_peptide_stream(
    path: str | Path, cfg: DigestConfig, rng: np.random.Generator, loop: bool = True
) -> Iterator[str]:
    """Stream digested peptides from a FASTA, shuffled per pass, optionally looping."""
    peptides = sorted({p for _h, seq in parse_fasta(path) for p in cleave_protein(seq, cfg)})
    if not peptides:
        raise ValueError(f"no peptides digested from {path}")
    peptides = np.array(peptides)
    while True:
        for i in rng.permutation(len(peptides)):
            yield str(peptides[i])
        if not loop:
            return


def unspecific_window_stream(
    path: str | Path, rng: np.random.Generator, min_len: int = 8, max_len: int = 11
) -> Iterator[str]:
    """Infinite stream of random protein sub-sequences (immunopeptidome-like, no enzyme).

    Samples a random protein then a random window, never enumerates the (combinatorially
    huge) full substring set, so it stays truly online. Length range defaults to the classic
    HLA class-I window (8-11).
    """
    prots = [seq for _h, seq in parse_fasta(path) if len(seq) >= min_len]
    if not prots:
        raise ValueError(f"no proteins >= {min_len} residues in {path}")
    prots = np.array(prots, dtype=object)
    while True:
        p = str(prots[rng.integers(len(prots))])
        upper = min(max_len, len(p))
        length = int(rng.integers(min_len, upper + 1))
        start = int(rng.integers(0, len(p) - length + 1))
        yield p[start : start + length]


def enumerate_tryptic_stream(
    path: str | Path, cfg: DigestConfig, loop: bool = False
) -> Iterator[str]:
    """Lazily yield tryptic peptides in FASTA order, no dedup (dup rate ~2%, not worth a
    proteome-scale seen-set). One pass ~= the whole digest; ``loop`` repeats forever."""
    while True:
        for _h, seq in parse_fasta(path):
            yield from cleave_protein(seq, cfg)
        if not loop:
            return


def enumerate_unspecific_stream(
    path: str | Path, min_len: int = 8, max_len: int = 11, loop: bool = False
) -> Iterator[str]:
    """Lazily yield every ``min_len..max_len`` protein sub-sequence (immunopeptidome-like),
    in FASTA order, no dedup. One pass covers the whole space exactly (± ~2% repeats)."""
    while True:
        for _h, seq in parse_fasta(path):
            n = len(seq)
            for w in range(min_len, max_len + 1):
                for i in range(n - w + 1):
                    yield seq[i : i + w]
        if not loop:
            return


def precursors_from_sequences(
    sequences: list[str],
    cfg: DigestConfig,
    rng: np.random.Generator,
    all_charge_states: bool = False,
) -> list[Precursor]:
    """Precursors for each sequence: fixed mods always, one sampled variable-mod form.

    ``all_charge_states=False`` samples a single charge per sequence, cheap, but a peptide is
    then only ever seen at one charge per pass, and charge is factored out of the trunk so the
    MS2 and CCS heads learn it from the contrast between charges of the SAME peptide. That
    contrast never appears.

    ``all_charge_states=True`` emits every charge in ``cfg.charges`` **consecutively**, so a
    peptide's charge states stay adjacent and land in one mini-batch. The sampled variable-mod
    form is held constant across them, making charge the only varying factor, otherwise the
    contrast is confounded by a different mod-form. Costs ``len(cfg.charges)``x the precursors,
    hence that much teacher time, in exchange for exhaustive and deterministic charge coverage.
    """
    charges = list(cfg.charges)
    # Parsed once per call rather than per sequence: a chunk is thousands of sequences and the
    # rules do not vary within one.
    fixed_rules = cfg.fixed_rules()
    variable_rules = cfg.variable_rules()
    out: list[Precursor] = []
    for seq in sequences:
        fixed = fixed_sites(seq, fixed_rules)
        # Each candidate site draws independently at its rule's rate, so a peptide with ten
        # serines is ten times as likely to carry a phosphate as one with a single serine. The
        # previous version picked a uniform *count* of modifications and then a random subset,
        # which made modification frequency a property of the config rather than of the peptide.
        chosen = sampled_sites(seq, variable_rules, rng, cfg.max_variable_mods)
        pep = Peptide(seq, tuple(fixed + chosen))
        if all_charge_states:
            out.extend(Precursor(pep, int(z), "train") for z in charges)
        else:
            out.append(Precursor(pep, int(charges[rng.integers(0, len(charges))]), "train"))
    return out
