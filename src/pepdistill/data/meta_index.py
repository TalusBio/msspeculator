"""Compact metadata index used while compiling prepared spectral-library chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..chem import Peptide
from .config import SplitConfig
from .prospect import ProspectSchema, parse_modseq
from .split import assign_split


@dataclass(frozen=True, slots=True)
class SpectrumMeta:
    """Metadata joined to one ``(raw_file, scan_number)`` spectrum."""

    peptide: Peptide
    charge: int
    irt: float
    raw_rt: float
    split: str
    mass_analyzer: str
    fragmentation: str
    energy: float
    andromeda: float


@dataclass
class MetaIndex:
    by_key: dict[tuple[str, int], SpectrumMeta] = field(default_factory=dict)

    def irt_stats(self, splits: frozenset[str]) -> tuple[int, float, float]:
        """Return ``(count, sum, sum_of_squares)`` for the selected split(s)."""
        selected = (meta.irt for meta in self.by_key.values() if meta.split in splits)
        values = list(selected)
        return len(values), sum(values), sum(value * value for value in values)


def build_meta_index_from_frame(
    frame: pd.DataFrame,
    split_cfg: SplitConfig | None = None,
    schema: ProspectSchema | None = None,
) -> MetaIndex:
    """Build an index from the metadata rows selected for one annotation shard."""
    s = schema or ProspectSchema()
    split_cfg = split_cfg or SplitConfig()
    required = [s.raw_file, s.scan_number, s.modified_sequence, s.charge]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"metadata frame missing required columns {missing}")
    irt_col = s.indexed_retention_time if s.indexed_retention_time in frame else s.retention_time
    raw_col = s.retention_time if s.retention_time in frame else irt_col
    if irt_col not in frame or raw_col not in frame:
        raise ValueError(f"metadata frame missing retention-time columns {irt_col!r}/{raw_col!r}")

    parsed: dict[str, tuple[str, tuple]] = {}
    index = MetaIndex()
    for row in frame.itertuples(index=False):
        values = row._asdict()
        modseq = str(values[s.modified_sequence])
        if modseq not in parsed:
            parsed[modseq] = parse_modseq(modseq)
        stripped, mods = parsed[modseq]
        key = (str(values[s.raw_file]), int(values[s.scan_number]))
        if key in index.by_key:
            continue
        collision_energy = values.get(s.collision_energy)
        index.by_key[key] = SpectrumMeta(
            peptide=Peptide(stripped, mods),
            charge=int(values[s.charge]),
            irt=float(values[irt_col]),
            raw_rt=float(values[raw_col]),
            split=assign_split(stripped, split_cfg),
            mass_analyzer=str(values.get(s.mass_analyzer, "")),
            fragmentation=str(values.get(s.fragmentation, "")),
            energy=(float(collision_energy) if collision_energy is not None else float("nan")),
            andromeda=(
                float(values[s.andromeda_score])
                if s.andromeda_score in values and values[s.andromeda_score] is not None
                else float("nan")
            ),
        )
    if not index.by_key:
        raise ValueError("metadata frame contains no rows")
    return index


__all__ = ["SpectrumMeta", "MetaIndex", "build_meta_index_from_frame"]
