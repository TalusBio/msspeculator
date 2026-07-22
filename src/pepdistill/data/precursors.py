"""Expand bare peptides into modified precursors (mod-forms x charges)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from ..chem import MOD_DELTA, Peptide
from .config import DigestConfig, SplitConfig
from .split import assign_split


@dataclass(frozen=True, slots=True)
class Precursor:
    peptide: Peptide
    charge: int
    split: str


def _mod_target(name: str) -> str:
    """Residue a mod applies to, from an ``X@R`` name (e.g. ``Oxidation@M`` -> ``M``)."""
    return name.split("@", 1)[1]


def _fixed_mod_sites(sequence: str, fixed_mods: tuple[str, ...]) -> list[tuple[int, str]]:
    sites: list[tuple[int, str]] = []
    for name in fixed_mods:
        target = _mod_target(name)
        sites += [(i, name) for i, aa in enumerate(sequence) if aa == target]
    return sites


def _variable_modforms(
    sequence: str, variable_mods: tuple[str, ...], max_var: int
) -> list[list[tuple[int, str]]]:
    """All variable-mod combinations (including the empty, unmodified form)."""
    candidates: list[tuple[int, str]] = []
    for name in variable_mods:
        target = _mod_target(name)
        candidates += [(i, name) for i, aa in enumerate(sequence) if aa == target]
    forms: list[list[tuple[int, str]]] = [[]]
    for k in range(1, min(max_var, len(candidates)) + 1):
        forms.extend([list(c) for c in combinations(candidates, k)])
    return forms


def enumerate_precursors(
    peptides: list[str], digest: DigestConfig, split: SplitConfig
) -> list[Precursor]:
    """Cartesian expansion of peptides over mod-forms and charge states."""
    for name in (*digest.fixed_mods, *digest.variable_mods):
        if name not in MOD_DELTA:
            raise ValueError(f"unknown modification {name!r}; known: {sorted(MOD_DELTA)}")

    out: list[Precursor] = []
    for seq in peptides:
        which_split = assign_split(seq, split)
        fixed = _fixed_mod_sites(seq, digest.fixed_mods)
        for var in _variable_modforms(seq, digest.variable_mods, digest.max_variable_mods):
            pep = Peptide(seq, tuple(fixed + var))
            for z in digest.charges:
                out.append(Precursor(pep, z, which_split))
    return out


def precursors_to_frame(precursors: list[Precursor]) -> pd.DataFrame:
    """Flat table for on-disk caching. ``mods`` is serialized as ``site:name;...``."""
    rows = []
    for p in precursors:
        pep = p.peptide
        rows.append(
            {
                "sequence": pep.sequence,
                "mods": ";".join(f"{s}:{n}" for s, n in pep.mods),
                "modified_sequence": pep.modified_sequence(),
                "length": pep.length,
                "charge": p.charge,
                "precursor_mz": pep.precursor_mz(p.charge),
                "split": p.split,
            }
        )
    return pd.DataFrame(rows)


def frame_to_precursors(df: pd.DataFrame) -> list[Precursor]:
    """Inverse of :func:`precursors_to_frame`."""
    out: list[Precursor] = []
    for row in df.itertuples(index=False):
        mods: tuple[tuple[int, str], ...] = ()
        if row.mods:
            mods = tuple(
                (int(site), name)
                for site, name in (pair.split(":", 1) for pair in row.mods.split(";"))
            )
        out.append(Precursor(Peptide(row.sequence, mods), int(row.charge), row.split))
    return out
