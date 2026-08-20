"""Compact metadata index used while compiling prepared spectral-library chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ..chem import Peptide
from .config import SplitConfig
from .proforma import MOD_TOKEN_PATTERN
from .prospect import ProspectSchema
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
    #: Spectra dropped because more than one *peptide* was reported for them. Counted apart from
    #: localization because the cause differs: nothing at all distinguishes the candidates.
    ambiguous_identification_spectra: int = 0

    def irt_stats(self, splits: frozenset[str]) -> tuple[int, float, float]:
        """Return ``(count, sum, sum_of_squares)`` for the selected split(s)."""
        selected = (meta.irt for meta in self.by_key.values() if meta.split in splits)
        values = list(selected)
        return len(values), sum(values), sum(value * value for value in values)


def _fill_optional(
    frame: pl.DataFrame, s: ProspectSchema, retention: tuple[str, ...]
) -> pl.DataFrame:
    """Give the optional columns their absent-value defaults, as columns.

    An unrecorded measurement becomes NaN and an unrecorded label an empty string. Done here so
    the row loop is pure reads: as a per-row conditional this was five branches per spectrum, and
    as a helper call it was five calls, either of which costs more than filling a column once.

    Retention time is in this group rather than treated as required. One source carries 100 null
    retention times, and a NaN label is masked out of the RT loss for that row alone
    (:func:`pepdistill.distill.losses.labeled_mse`), so a missing one costs that row's RT
    supervision rather than failing a whole shard. Pandas surfaced these as NaN, which is why
    they never surfaced before.
    """
    numeric = (s.collision_energy, s.andromeda_score, s.precursor_intensity, *retention)
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


def _resolve_localizations(frame: pl.DataFrame, s: ProspectSchema) -> tuple[pl.DataFrame, int, int]:
    """Reduce a source to one labelled row per spectrum, dropping the spectra nothing can label.

    PROSPECT reports a spectrum once per candidate assignment. Two kinds of ambiguity follow, and
    neither can be resolved by picking a row:

    * the same peptide with the modification placed differently. Where the scores differ the engine
      did localize and the best row is the label; where the maximum is tied it did not.
    * two different peptides for one spectrum. There is no basis at all for choosing, and the
      spectrum's identity is what everything downstream keys on.

    Both are dropped and counted. The result is exactly one row per ``(raw_file, scan_number)``, so
    the caller cannot silently overwrite one label with another.

    Expressed over columns rather than in a row loop: this decides which of ~500k rows per source
    survive, and the surviving rows are the only ones worth building Python objects for.
    """
    spectrum = [s.raw_file, s.scan_number]
    group = spectrum + ["_stripped"]
    scored = frame.with_columns(
        pl.col(s.modified_sequence).str.replace_all(MOD_TOKEN_PATTERN, "").alias("_stripped")
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
    # Both counts are in spectra, not rows: one spectrum reported with two peptides is one loss,
    # and it contributes two rows here.
    unlocalizable = best.filter(pl.col("_localizations") > 1).select(spectrum).n_unique()
    localized = best.filter(pl.col("_localizations") == 1).with_columns(
        pl.col("_stripped").n_unique().over(spectrum).alias("_peptides")
    )
    unidentifiable = localized.filter(pl.col("_peptides") > 1).select(spectrum).n_unique()
    return localized.filter(pl.col("_peptides") == 1), unlocalizable, unidentifiable


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

    localized, unlocalizable, unidentifiable = _resolve_localizations(frame, s)
    localized = _fill_optional(localized, s, (irt_col, raw_col))
    # Checked once over columns rather than by casting every field of every row: Polars already
    # hands back str/int/float, so those casts only ever served as a null guard, and paying one
    # per field per row for it cost more than the rest of the loop body. Only the columns that
    # identify a spectrum are required; a missing measurement is filled above.
    null_counts = localized.select(pl.col(required).null_count()).row(0, named=True)
    empty = {name: count for name, count in null_counts.items() if count}
    if empty:
        raise ValueError(f"metadata frame has null values in required columns: {empty}")
    # Everything derived purely from the modified sequence is cached on it. A source has ~40
    # replicate spectra per peptidoform, so parsing, building the peptide and hashing its split
    # once each rather than per row also leaves one shared Peptide per peptidoform instead of one
    # per spectrum. The peptide is immutable (Rust-backed, read-only properties), so sharing is
    # safe; the split depends only on the stripped sequence by definition.
    peptides: dict[str, tuple[Peptide, str]] = {}
    index = MetaIndex(
        ambiguous_localization_spectra=unlocalizable,
        ambiguous_identification_spectra=unidentifiable,
    )
    for values in localized.iter_rows(named=True):
        modseq = values[s.modified_sequence]
        cached = peptides.get(modseq)
        if cached is None:
            peptide = Peptide.from_prospect(modseq)
            cached = (peptide, assign_split(peptide.sequence, split_cfg))
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
        # Distinguish the ways this ends up empty: an empty selection is a caller error, while
        # every spectrum being ambiguous is a real (if extreme) property of the source.
        dropped = unlocalizable + unidentifiable
        if dropped:
            raise ValueError(
                f"every one of the {dropped:,} spectra in the metadata frame is ambiguous "
                f"({unlocalizable:,} cannot be localized, {unidentifiable:,} report more than one "
                "peptide), so none can be labelled"
            )
        raise ValueError("metadata frame contains no rows")
    return index


__all__ = ["SpectrumMeta", "MetaIndex", "build_meta_index_from_frame"]
