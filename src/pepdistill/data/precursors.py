"""Expand bare peptides into modified precursors (mod-forms x charges)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..chem import Peptide
from .config import DigestConfig, SplitConfig
from .mod_rules import enumerate_modforms, fixed_sites
from .split import assign_split


@dataclass(frozen=True, slots=True)
class Precursor:
    peptide: Peptide
    charge: int
    split: str


def enumerate_precursors(
    peptides: list[str], digest: DigestConfig, split: SplitConfig
) -> list[Precursor]:
    """Cartesian expansion of peptides over mod-forms and charge states.

    Exhaustive: every variable mod-form up to the per-peptide cap, ignoring the rules'
    probabilities. This is the library-generation path, where a missing modform is an
    identification that cannot be made. The pretrain stream samples instead -- see
    :func:`pepdistill.data.mod_rules.sampled_sites`.
    """
    fixed_rules = digest.fixed_rules()
    variable_rules = tuple(rule for rule, _ in digest.variable_rules())
    out: list[Precursor] = []
    for seq in peptides:
        which_split = assign_split(seq, split)
        fixed = fixed_sites(seq, fixed_rules)
        for var in enumerate_modforms(seq, variable_rules, digest.max_variable_mods):
            pep = Peptide(seq, tuple(fixed + var))
            for z in digest.charges:
                out.append(Precursor(pep, z, which_split))
    return out


def _render_site(site) -> str:
    return site if isinstance(site, str) else str(site)


def _render_spec(spec) -> str:
    return spec if isinstance(spec, str) else f"{spec:+}"


def precursors_to_frame(precursors: list[Precursor]) -> pd.DataFrame:
    """Flat table for on-disk caching. ``mods`` is serialized as ``site:spec;...``,
    e.g. ``"n:TMT6plex;1:+79.96633"`` (terminal site, mass-only spec)."""
    rows = []
    for p in precursors:
        pep = p.peptide
        rows.append(
            {
                "sequence": pep.sequence,
                "mods": ";".join(f"{_render_site(s)}:{_render_spec(m)}" for s, m in pep.mods),
                "modified_sequence": pep.modified_sequence(),
                "length": pep.length,
                "charge": p.charge,
                "precursor_mz": pep.precursor_mz(p.charge),
                "split": p.split,
            }
        )
    return pd.DataFrame(rows)


def _parse_site(tok: str):
    return tok if tok in ("n", "c") else int(tok)


def _parse_spec(tok: str):
    return float(tok) if tok[0] in "+-" else tok


def frame_to_precursors(df: pd.DataFrame) -> list[Precursor]:
    """Inverse of :func:`precursors_to_frame`."""
    out: list[Precursor] = []
    for row in df.itertuples(index=False):
        mods: tuple = ()
        if row.mods:
            mods = tuple(
                (_parse_site(site), _parse_spec(spec))
                for site, spec in (pair.split(":", 1) for pair in row.mods.split(";"))
            )
        out.append(Precursor(Peptide(row.sequence, mods), int(row.charge), row.split))
    return out
