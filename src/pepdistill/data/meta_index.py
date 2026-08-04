"""Per-run index of the pool meta rows a set of shards actually needs.

One annotation shard covers one ``raw_file``. A pool's meta covers every raw_file in the pool
— measured, 3,205,172 rows / 1,328 MB in pandas for a shard that needs 4,208 of them (0.13%).
Reading all of it, then ``drop_duplicates`` and ``parse_modseq`` over all of it, was 61% of
decode time and was repeated per shard.

So: one projected, ``raw_file``-filtered read per source (0.83 s / 1.4 MB measured), parsed
once, held in RAM for the whole run. Epochs never touch the meta again.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pyarrow.dataset as pads

from ..chem import Peptide
from .config import SplitConfig
from .prospect import ProspectSchema, ProspectSource, parse_modseq
from .split import assign_split


@dataclass(frozen=True, slots=True)
class SpectrumMeta:
    """Everything one spectrum needs that does NOT come from its fragments.

    Acquisition factors are held PER SPECTRUM. A single PROSPECT raw file mixes them —
    measured, ``…-2xIT_2xHCD-1h-R1`` is 15,325 ITMS against 14,649 FTMS with four distinct NCE
    values — so a per-raw-file value would be fabricated for about half its spectra, and these
    feed ``MSContextEncoder`` directly.
    """

    peptide: Peptide
    charge: int
    irt: float  # indexed_retention_time — the context-free base RT target
    raw_rt: float  # run-dependent retention time — the chrom_context target
    split: str
    mass_analyzer: str
    fragmentation: str
    energy: float  # NaN when the row carries none -> the encoder masks the term
    andromeda: float  # val dedup quality; see val_winner_keys


@dataclass
class MetaIndex:
    by_key: dict[tuple[str, int], SpectrumMeta] = field(default_factory=dict)

    def allowed_keys(
        self, raw_files: list[str], splits: frozenset[str]
    ) -> set[tuple[str, int]]:
        """``(raw_file, scan_number)`` pairs in ``raw_files`` whose split is in ``splits``.

        This is the split filter the streaming reader applies. It keys on the join key rather
        than the sequence: ``scan_number`` plus ``raw_file`` is 0.3% of a shard's bytes, while
        the sequence column is 27% of the projected read.
        """
        wanted = set(raw_files)
        return {
            key
            for key, sm in self.by_key.items()
            if key[0] in wanted and sm.split in splits
        }

    def val_winner_keys(self, raw_files: list[str]) -> set[tuple[str, int]]:
        """The one val scan to keep per ``(modified_sequence, charge)`` within ``raw_files``.

        The dataset is not part of the group key because it cannot vary inside one call: the
        caller (``collect_val_examples``) groups shards by ``dataset_id`` first and calls this
        once per group, so ``raw_files`` already scopes the reduction to a single dataset. A
        ``dataset`` component would have been a constant in every tuple.

        Quality is ``andromeda_score``. Measured against a leave-one-out consensus over 300
        keys, it picks a spectrum with mean cosine 0.9374 to that peptide's other observations
        where the previous rule (summed matched-fragment intensity) managed 0.8871, winning on
        only 28.3% of keys (Wilcoxon p = 1.36e-17). It also hits the elution apex 6.0% of the
        time against 2.3%, which is chance.

        Because the score is in meta, this runs before any fragment is read, so val reduces to
        a scan allowlist rather than a reduction over decoded spectra.

        Ties break on the lower ``(raw_file, scan_number)`` so the set is reproducible.
        """
        wanted = set(raw_files)
        best: dict[tuple, tuple[float, tuple[str, int]]] = {}
        for key, sm in self.by_key.items():
            if key[0] not in wanted or sm.split != "val":
                continue
            group = (sm.peptide.modified_sequence(), sm.charge)
            cur = best.get(group)
            if cur is None or sm.andromeda > cur[0] or (
                sm.andromeda == cur[0] and key < cur[1]
            ):
                best[group] = (sm.andromeda, key)
        return {key for _, key in best.values()}

    def irt_stats(self, splits: frozenset[str]) -> tuple[int, float, float]:
        """``(count, sum, sum_of_squares)`` of iRT over the selected splits.

        Returned as sufficient statistics rather than mean/std so several sources combine by
        addition. This is the population the global RT affine is established from.
        """
        n = 0
        total = 0.0
        sumsq = 0.0
        for sm in self.by_key.values():
            if sm.split in splits:
                n += 1
                total += sm.irt
                sumsq += sm.irt * sm.irt
        return n, total, sumsq


def build_meta_index(
    src: ProspectSource,
    meta_filename: str,
    raw_files: list[str],
    split_cfg: SplitConfig | None = None,
    schema: ProspectSchema | None = None,
) -> MetaIndex:
    """Read the pool meta projected to the needed columns and filtered to ``raw_files``."""
    s = schema or src.schema
    split_cfg = split_cfg or SplitConfig()
    cols = [
        s.raw_file,
        s.scan_number,
        s.modified_sequence,
        s.charge,
        s.retention_time,
        s.indexed_retention_time,
        s.collision_energy,
        s.mass_analyzer,
        s.fragmentation,
        s.andromeda_score,
    ]
    path = src.resolve_file(meta_filename)
    dataset = pads.dataset(path)
    present = [c for c in cols if c in dataset.schema.names]
    required = (s.raw_file, s.scan_number, s.modified_sequence, s.charge)
    missing = [c for c in required if c not in present]
    if missing:
        raise ValueError(f"meta {meta_filename!r} missing required columns {missing}")

    # Absent (not merely all-null) factor columns are not fatal -- acquisition_key's
    # documented "missing categorical columns are dropped (not fatal)" contract holds here
    # too -- but every spectrum then falls back to the unknown category below. Without this,
    # a schema misconfiguration (wrong column name) is silently indistinguishable from a pool
    # that genuinely carries no such metadata. One warning per absent column per call.
    for col, label, consequence in (
        (s.mass_analyzer, "mass_analyzer", "the unknown analyzer category"),
        (s.fragmentation, "fragmentation", "the unknown fragmentation category"),
        (s.collision_energy, "collision_energy", "NaN"),
    ):
        if col not in present:
            warnings.warn(
                f"meta {meta_filename!r} has no {label!r} column ({col!r}); "
                f"every spectrum will encode {consequence} for it",
                UserWarning,
                stacklevel=2,
            )

    # Scan record batches instead of calling ``to_table``.  A large pool such as
    # TUM_isoform has 9.5M metadata rows: materializing the full Arrow table and then a
    # Python list for every column briefly held several copies of the same data and could
    # exhaust a 30 GB cloud worker before the compact index was complete.
    scanner = dataset.scanner(
        columns=present,
        filter=pads.field(s.raw_file).isin(list(raw_files)),
        batch_size=65_536,
    )
    irt_col = s.indexed_retention_time if s.indexed_retention_time in present else s.retention_time
    raw_col = s.retention_time if s.retention_time in present else irt_col

    # parse_modseq is the only per-peptide Python cost; memoise across batches, shards, and
    # sources.  Only one batch's ``to_pylist`` representation is resident at a time.
    parsed: dict[str, tuple[str, tuple]] = {}
    index = MetaIndex()
    for batch in scanner.to_batches():
        cd = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
        for i in range(batch.num_rows):
            modseq = str(cd[s.modified_sequence][i])
            if modseq not in parsed:
                parsed[modseq] = parse_modseq(modseq)
            stripped, mods = parsed[modseq]
            rf = str(cd[s.raw_file][i])
            scan = int(cd[s.scan_number][i])
            key = (rf, scan)
            if key in index.by_key:
                continue  # first row wins, matching the previous drop_duplicates behaviour
            ce = cd[s.collision_energy][i] if s.collision_energy in cd else None
            index.by_key[key] = SpectrumMeta(
                peptide=Peptide(stripped, mods),
                charge=int(cd[s.charge][i]),
                irt=float(cd[irt_col][i]),
                raw_rt=float(cd[raw_col][i]),
                split=assign_split(stripped, split_cfg),
                mass_analyzer=str(cd[s.mass_analyzer][i]) if s.mass_analyzer in cd else "",
                fragmentation=str(cd[s.fragmentation][i]) if s.fragmentation in cd else "",
                # Never fabricated: absent energy stays NaN and the encoder masks the term.
                energy=float(ce) if ce is not None else float("nan"),
                andromeda=(
                    float(cd[s.andromeda_score][i])
                    if s.andromeda_score in cd and cd[s.andromeda_score][i] is not None
                    else float("nan")
                ),
            )
    if not index.by_key:
        raise ValueError(
            f"no meta rows in {meta_filename!r} for raw_files {sorted(raw_files)}; "
            "the shard selection and the meta file disagree"
        )
    return index


__all__ = ["SpectrumMeta", "MetaIndex", "build_meta_index"]
