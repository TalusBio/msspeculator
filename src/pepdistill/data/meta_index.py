"""Compact metadata index used while compiling prepared spectral-library chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

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
    precursor_intensity: float


@dataclass
class MetaIndex:
    by_key: dict[tuple[str, int], SpectrumMeta] = field(default_factory=dict)
    #: Spectra dropped because the same peptide was reported with more than one modification
    #: placement. Reported so the loss is visible rather than inferred from a row count.
    ambiguous_localization_spectra: int = 0

    def irt_stats(self, splits: frozenset[str]) -> tuple[int, float, float]:
        """Return ``(count, sum, sum_of_squares)`` for the selected split(s)."""
        selected = (meta.irt for meta in self.by_key.values() if meta.split in splits)
        values = list(selected)
        return len(values), sum(values), sum(value * value for value in values)


#: A UNIMOD token, or the ProForma separator between a terminal token and the sequence. Stripping
#: both leaves the bare residues, which is all the localization rule needs -- so the peptide never
#: has to be parsed in Python just to group by it.
_MOD_TOKEN_PATTERN = r"\[UNIMOD:\d+\]|-"


def _fill_optional(frame: pl.DataFrame, s: ProspectSchema) -> pl.DataFrame:
    """Give the optional columns their absent-value defaults, as columns.

    An unrecorded measurement becomes NaN and an unrecorded label an empty string. Done here so
    the row loop is pure reads: as a per-row conditional this was five branches per spectrum, and
    as a helper call it was five calls, either of which costs more than filling a column once.
    """
    numeric = (s.collision_energy, s.andromeda_score, s.precursor_intensity)
    labels = (s.mass_analyzer, s.fragmentation)
    return frame.with_columns(
        [
            (
                pl.col(name).fill_null(float("nan"))
                if name in frame.columns
                else pl.lit(float("nan"))
            ).alias(name)
            for name in numeric
        ]
        + [
            (pl.col(name).fill_null("") if name in frame.columns else pl.lit("")).alias(name)
            for name in labels
        ]
    )


def _resolve_localizations(frame: pl.DataFrame, s: ProspectSchema) -> tuple[pl.DataFrame, int]:
    """Keep one best-scoring localization per spectrum, dropping the ones that cannot be localized.

    PROSPECT reports a spectrum once per candidate placement. Where the scores differ the engine
    did localize and the best row is the label; where the maximum is tied it did not, and keeping
    whichever row came first in the file would teach a site-specific model a coin flip.

    Expressed over columns rather than in a row loop: this decides which of ~500k rows per source
    survive, and the surviving rows are the only ones worth building Python objects for.
    """
    group = [s.raw_file, s.scan_number, "_stripped"]
    scored = frame.with_columns(
        pl.col(s.modified_sequence).str.replace_all(_MOD_TOKEN_PATTERN, "").alias("_stripped")
    )
    if s.andromeda_score in scored.columns:
        # `max` skips nulls, so a group scored entirely by nulls keeps every row and stays
        # ambiguous -- the same verdict as an explicit tie, which is what an absent score means.
        scored = scored.with_columns(pl.col(s.andromeda_score).max().over(group).alias("_best"))
        scored = scored.filter(
            (pl.col(s.andromeda_score) == pl.col("_best")) | pl.col("_best").is_null()
        )
    best = scored.group_by(group).agg(
        pl.col(s.modified_sequence).n_unique().alias("_localizations"),
        pl.all().exclude(group).first(),
    )
    ambiguous = int((best["_localizations"] > 1).sum())
    return best.filter(pl.col("_localizations") == 1), ambiguous


def build_meta_index_from_frame(
    frame: pl.DataFrame,
    split_cfg: SplitConfig | None = None,
    schema: ProspectSchema | None = None,
) -> MetaIndex:
    """Build an index from the metadata rows selected for one annotation shard.

    Takes a Polars frame because the caller already has one: the localization rule is expressed
    over columns, so converting to pandas on the way in only to convert back was pure overhead on
    every shard.
    """
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

    localized, ambiguous = _resolve_localizations(frame, s)
    localized = _fill_optional(localized, s)
    # Checked once over columns rather than by casting every field of every row: Polars already
    # hands back str/int/float, so those casts only ever served as a null guard, and paying one
    # per field per row for it cost more than the rest of the loop body.
    required_values = [s.raw_file, s.scan_number, s.modified_sequence, s.charge, irt_col, raw_col]
    null_counts = localized.select(pl.col(required_values).null_count()).row(0, named=True)
    empty = {name: count for name, count in null_counts.items() if count}
    if empty:
        raise ValueError(f"metadata frame has null values in required columns: {empty}")
    # Everything derived purely from the modified sequence is cached on it. A source has ~40
    # replicate spectra per peptidoform, so parsing, building the peptide and hashing its split
    # once each rather than per row also leaves one shared Peptide per peptidoform instead of one
    # per spectrum. The peptide is immutable (Rust-backed, read-only properties), so sharing is
    # safe; the split depends only on the stripped sequence by definition.
    peptides: dict[str, tuple[Peptide, str]] = {}
    index = MetaIndex(ambiguous_localization_spectra=ambiguous)
    for values in localized.iter_rows(named=True):
        modseq = values[s.modified_sequence]
        cached = peptides.get(modseq)
        if cached is None:
            stripped, mods = parse_modseq(modseq)
            cached = (Peptide(stripped, mods), assign_split(stripped, split_cfg))
            peptides[modseq] = cached
        peptide, split = cached
        index.by_key[(values[s.raw_file], values[s.scan_number])] = SpectrumMeta(
            peptide=peptide,
            charge=values[s.charge],
            irt=values[irt_col],
            raw_rt=values[raw_col],
            split=split,
            mass_analyzer=values[s.mass_analyzer],
            fragmentation=values[s.fragmentation],
            energy=values[s.collision_energy],
            andromeda=values[s.andromeda_score],
            precursor_intensity=values[s.precursor_intensity],
        )
    if not index.by_key:
        # Distinguish the two ways this ends up empty: an empty selection is a caller error, while
        # every spectrum being unlocalizable is a real (if extreme) property of the source.
        if ambiguous:
            raise ValueError(
                f"all {ambiguous:,} spectra in the metadata frame report more than one equally "
                "scored modification placement, so none can be localized"
            )
        raise ValueError("metadata frame contains no rows")
    return index


__all__ = ["SpectrumMeta", "MetaIndex", "build_meta_index_from_frame"]
