"""PROSPECT (ProteomeTools spectrum compendium) as a real-spectra source.

PROSPECT is experimental data (not a teacher): real MS2/RT with acquisition metadata
(collision energy, mass analyzer, fragmentation) — the multi-source signal the context
conditioning consumes. Files are parquet on Zenodo, fetched through :class:`FileCache` so an
S3 mirror shields Zenodo.

Schema is NOT hardcoded blindly: column names live in :class:`ProspectSchema` (defaults are
the documented PROSPECT names) and are validated against the actual file, failing loudly
with the available columns if they differ. Decoding rows into student targets (mod-string
parsing + the Prosit-style intensity vector -> our per-fragment matrix) is deliberately
deferred behind :meth:`to_labels` until a real file pins the exact layout — the cache,
listing, read, and acquisition-factor extraction below need no such assumption.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import fsspec
import numpy as np
import pandas as pd
import pepdistill_rs as _rs

from ..chem import ION_TYPES, Peptide
from ..teacher.base import PrecursorLabels
from .cache import FileCache, default_cache_dir, http_origin
from .config import SplitConfig
from .precursors import Precursor
from .prospect_catalog import load_catalog
from .split import assign_split
from .zenodo import ZenodoRecord

if TYPE_CHECKING:
    from .meta_index import MetaIndex

_UNIMOD_TOKEN = re.compile(r"([A-Z])|\[UNIMOD:(\d+)\]")


def parse_modseq(modseq: str) -> tuple[str, tuple[tuple, ...]]:
    """Parse a ProForma UNIMOD string -> (stripped_sequence, mods) in OUR mod names.

    ``[UNIMOD:737]ET[UNIMOD:21]TLHLVLR`` -> ("ETTLHLVLR", (("n","TMT6plex"),(1,"Phospho"))).
    A mod token attaches to the residue it follows; a leading token (before any residue) is
    routed to the N-terminal site. Every accession resolves against the vendored UNIMOD table
    (``pepdistill_rs.unimod_name``: our alias if one exists, else the UNIMOD title) — an
    accession absent from that table raises ``ValueError`` rather than silently dropping the
    peptide later.

    Resolving is not enough: the name must also be *encodable*, i.e. project onto the model's
    six-element basis. Iodo (UNIMOD:129) and the Se-containing mods resolve to a perfectly good
    name and then fail inside ``collate``. Checking here turns a multi-hour training run that
    aborts mid-epoch into a ``ValueError`` at parse time, next to the row that caused it.
    """
    residues: list[str] = []
    mods: list[tuple] = []
    pos = -1
    for m in _UNIMOD_TOKEN.finditer(modseq):
        aa = m.group(1)
        if aa:
            residues.append(aa)
            pos += 1
        else:
            n = int(m.group(2))
            name = _rs.unimod_name(n)
            if name is None:
                raise ValueError(
                    f"unknown UNIMOD accession {n} in {modseq!r}: not in the vendored "
                    "unimod table (regenerate with tools/gen_unimod.py)"
                )
            try:
                _rs.mod_element_comp(name)
            except Exception as exc:
                raise ValueError(
                    f"UNIMOD:{n} ({name}) in {modseq!r} resolves but cannot be encoded: {exc}"
                ) from exc
            site = "n" if pos < 0 else pos
            mods.append((site, name))
    return "".join(residues), tuple(mods)


# record name -> Zenodo record id (from the PROSPECT repo).
RECORDS: dict[str, str] = {
    "prospect": "6602020",
    "tmt": "8221499",
    "multi_ptm": "11472525",
    "tmt_ptm": "11474099",
    "test_ptm": "11477731",
}


@dataclass(frozen=True, slots=True)
class ShardInfo:
    """One parquet shard inside an annotation zip, as read from the central directory.

    ``raw_bytes`` is what decoding has to hold; ``packed_bytes`` is only what crosses the wire.
    For these pools the two are often equal — parquet is already compressed, so the zip adds
    little — which means download size is a fair proxy for memory pressure here.
    """

    name: str
    packed_bytes: int
    raw_bytes: int

    @property
    def short_name(self) -> str:
        return self.name.split("/")[-1]


@dataclass(frozen=True, slots=True)
class ProspectSchema:
    """Column-name mapping. Defaults follow the documented PROSPECT columns; override per
    file if a variant differs (validated at read time)."""

    # Defaults verified against a real test_ptm meta file (columns: modified_sequence,
    # precursor_charge, aligned/orig_collision_energy, mass_analyzer, fragmentation,
    # retention_time, indexed_retention_time, ...). Meta files carry NO stripped `sequence`
    # column and NO intensities — intensities live in the annotation parquet (a .zip).
    modified_sequence: str = "modified_sequence"
    sequence: str = "sequence"  # absent in meta; strip modified_sequence when missing
    charge: str = "precursor_charge"
    collision_energy: str = "aligned_collision_energy"
    mass_analyzer: str = "mass_analyzer"
    fragmentation: str = "fragmentation"
    retention_time: str = "retention_time"
    indexed_retention_time: str = "indexed_retention_time"
    # Join keys (meta <-> annotation) and annotation (long-format fragment) columns.
    raw_file: str = "raw_file"
    scan_number: str = "scan_number"
    ann_ion_type: str = "ion_type"
    ann_ordinal: str = "no"
    ann_frag_charge: str = "charge"
    ann_intensity: str = "intensity"
    ann_neutral_loss: str = "neutral_loss"
    andromeda_score: str = "andromeda_score"  # val dedup quality

    # Columns required to even treat a file as PROSPECT (identity + acquisition context).
    def required(self) -> list[str]:
        return [self.modified_sequence, self.charge, self.collision_energy]

    # Columns that define an acquisition "source" for context conditioning.
    def acquisition_factors(self) -> list[str]:
        return [self.mass_analyzer, self.fragmentation]


def make_cache(
    local_dir: str | None = None, s3_prefix: str | None = None, write_through: bool = True
) -> FileCache:
    """Build the tiered cache: local first, optional S3 mirror second.

    ``s3_prefix`` like ``s3://my-bucket/prospect``. S3 creds come from the standard AWS
    chain (env / profile); this never embeds them.
    """
    tiers = [local_dir or default_cache_dir()]
    if s3_prefix:
        tiers.append(s3_prefix)
    return FileCache(tiers, write_through=write_through)


def decode_fragments(
    index: "MetaIndex", frag: pd.DataFrame, schema: ProspectSchema
) -> tuple["RealLabels", list[tuple[str, int]]]:
    """Scatter one already-filtered fragment chunk into per-spectrum MS2 matrices.

    ``frag`` must already be restricted to b/y ions at fragment charge 1-2 with no neutral
    loss, and to the keys the caller wants; it carries its own ``raw_file`` column, so one call
    handles a shard holding several raw files. This function does the (site, col) arithmetic,
    the max-collapse of duplicate cells, and the per-spectrum scatter — nothing else. It is the
    single scatter implementation: both the streaming reader and the in-memory ``to_labels``
    path go through here.

    Returns the labels plus each emitted example's ``(raw_file, scan_number)`` key, in order,
    so the caller can attach that spectrum's own acquisition factors — which vary within a raw
    file and must not be collapsed to one value per run.

    Spectra whose key is absent from ``index`` are skipped: the meta join is what says a
    spectrum is usable, and a fragment row without one has no peptide to attach to.

    Raises ``ValueError`` if ``frag`` was not actually pre-filtered to b/y ions at fragment
    charge 1-2, or if its ``scan_number`` column is not cleanly integral (a NaN or fractional
    scan number would otherwise miss the index silently rather than fail loudly).
    """
    s = schema
    n_ion = len(ION_TYPES)
    empty = RealLabels([], [], [], [], {})
    if frag.empty:
        return empty, []

    # Both preconditions run on ~0.7-1M filtered rows per shard per epoch, and each is checked
    # the way that is actually fastest for its dtype -- not uniformly "with numpy".
    #
    # ion_type is a parquet string column, so it arrives as OBJECT dtype and np.unique falls
    # back to a Python-level comparison sort: measured at 1M rows, np.unique is 0.2520 s
    # against 0.0123 s for set(tolist()) -- 20x SLOWER. Hashing wins on object dtype; do not
    # "vectorize" this one.
    ion_types = frag[s.ann_ion_type].to_numpy()
    bad_ion = sorted(set(ion_types.tolist()) - {"b", "y"})
    if bad_ion:
        raise ValueError(
            f"decode_fragments requires ion_type in {{'b', 'y'}} only; got {bad_ion}. "
            "Pre-filter with fragment_filter_mask before calling."
        )
    # Fragment charge IS numeric, so np.isin is 0.0003 s against 0.0542 s for the per-row
    # `set(int(x) for x in z_raw)` this replaced -- and that `int(x)` also turned a NaN charge
    # into a bare "cannot convert float NaN to integer" instead of the named error below.
    z_raw = frag[s.ann_frag_charge].to_numpy()
    bad_z_mask = ~np.isin(z_raw, (1, 2))  # NaN is never in (1, 2), so it reports here
    if bad_z_mask.any():
        bad_z = np.unique(z_raw[bad_z_mask]).tolist()  # np.unique is already sorted + distinct
        raise ValueError(
            f"decode_fragments requires fragment charge in {{1, 2}} only; got {bad_z}. "
            "Pre-filter with fragment_filter_mask before calling."
        )
    z = z_raw.astype(np.int64)

    scan_raw = frag[s.scan_number].to_numpy()
    if not np.issubdtype(scan_raw.dtype, np.integer):
        scan_f = scan_raw.astype(np.float64)
        bad = np.isnan(scan_f) | (scan_f != np.floor(scan_f))
        if bad.any():
            raise ValueError(
                f"{s.scan_number!r} must be integral scan numbers with no NaN; "
                f"found {int(bad.sum())} bad value(s)"
            )
    scans = scan_raw.astype(np.int64)
    raws = frag[s.raw_file].astype(str).to_numpy()

    # Per-spectrum (not per-row) index lookup: a filtered real shard is 1-3M fragment rows over
    # far fewer distinct spectra, so factorizing first turns this into one dict lookup per
    # spectrum instead of one per row.
    codes, uniq = pd.factorize(pd.MultiIndex.from_arrays([raws, scans]))
    uniq_len = np.array(
        [len(index.by_key[k].peptide.sequence) if k in index.by_key else 0 for k in uniq],
        dtype=np.int64,
    )
    lengths = uniq_len[codes]

    is_b = ion_types == "b"
    ordinal = frag[s.ann_ordinal].to_numpy().astype(np.int64)
    # ION_TYPES order is b1,y1,b2,y2 -> col = (b?0:1) + 2*(z-1); b site = ord-1, y site = n-1-ord.
    site = np.where(is_b, ordinal - 1, lengths - 1 - ordinal)
    col = np.where(is_b, 0, 1) + 2 * (z - 1)
    keep = (lengths > 1) & (site >= 0) & (site < lengths - 1)
    if not keep.any():
        return empty, []

    work = frag.assign(
        _raw=raws, _scan=scans, _site=site, _col=col,
        _inten=frag[s.ann_intensity].to_numpy(dtype=np.float32),
    )[keep]
    agg = (
        work.groupby(["_raw", "_scan", "_site", "_col"], sort=False)["_inten"]
        .max()
        .reset_index()
    )

    precursors: list[Precursor] = []
    labels: list[PrecursorLabels] = []
    raw_rt: list[float] = []
    source_ids: list[str] = []
    out_keys: list[tuple[str, int]] = []
    acquisition: dict[str, dict] = {}
    for (rf, scan), grp in agg.groupby(["_raw", "_scan"], sort=False):
        key = (str(rf), int(scan))
        sm = index.by_key[key]
        ms2 = np.zeros((len(sm.peptide.sequence) - 1, n_ion), dtype=np.float32)
        ms2[grp["_site"].to_numpy(), grp["_col"].to_numpy()] = grp["_inten"].to_numpy()
        if not ms2.any():
            continue
        precursors.append(Precursor(sm.peptide, sm.charge, sm.split))
        labels.append(PrecursorLabels(ms2=ms2, rt=sm.irt, ccs=float("nan")))
        raw_rt.append(sm.raw_rt)
        source_ids.append(key[0])
        out_keys.append(key)
        # Kept for RealLabels' documented shape. The per-example factors that actually reach
        # the encoder come from SpectrumMeta via out_keys, not from this map.
        acquisition.setdefault(
            key[0],
            {
                "mass_analyzer": sm.mass_analyzer,
                "fragmentation": sm.fragmentation,
                "collision_energy": sm.energy,
            },
        )
    return RealLabels(precursors, labels, raw_rt, source_ids, acquisition), out_keys


def fragment_filter_mask(ann: pd.DataFrame, schema: ProspectSchema) -> np.ndarray:
    """b/y ions, fragment charge 1-2, no neutral loss. Measured to keep 10-35% by pool."""
    return (
        ann[schema.ann_ion_type].isin(("b", "y")).to_numpy()
        & ann[schema.ann_frag_charge].isin((1, 2)).to_numpy()
        & (ann[schema.ann_neutral_loss].fillna("") == "").to_numpy()
    )


class ProspectSource:
    def __init__(
        self,
        record: str = "prospect",
        cache: FileCache | None = None,
        schema: ProspectSchema | None = None,
    ) -> None:
        if record not in RECORDS:
            raise ValueError(f"unknown PROSPECT record {record!r}; known: {sorted(RECORDS)}")
        self.record = record
        self.schema = schema or ProspectSchema()
        self.cache = cache or make_cache()
        self.record_id = RECORDS[record]
        # Checked-in catalog is the offline listing; ZenodoRecord is the live fallback.
        self._catalog = load_catalog()["records"].get(record, {}).get("files", {})
        self.zenodo = ZenodoRecord(self.record_id, self.cache)

    def files(self) -> list[str]:
        """Parquet files in the record, from the checked-in catalog (no network)."""
        if self._catalog:
            return sorted(k for k in self._catalog if k.endswith(".parquet"))
        return sorted(k for k in self.zenodo.list_files() if k.endswith(".parquet"))

    def resolve_file(self, filename: str) -> str:
        """Local path for one file: catalog URL through the cache, else live Zenodo."""
        entry = self._catalog.get(filename)
        if entry:
            key = f"zenodo/{self.record_id}/{filename}"
            return self.cache.resolve(key, http_origin(entry["url"]))
        return self.zenodo.resolve_file(filename)

    def read(self, filename: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Fetch one parquet through the cache and read it, validating the schema."""
        path = self.resolve_file(filename)
        df = pd.read_parquet(path, columns=columns)
        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.schema.required() if c not in df.columns]
        if missing:
            raise ValueError(
                f"PROSPECT file missing required columns {missing}; "
                f"available: {list(df.columns)}. Adjust ProspectSchema to match."
            )

    def acquisition_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-row acquisition factors for context conditioning: analyzer, fragmentation, NCE.

        Categoricals feed the acq context embedding; NCE is the continuous factor. Missing
        categorical columns are dropped (not fatal) so partial files still yield NCE.
        """
        s = self.schema
        cols = {c: df[c] for c in s.acquisition_factors() if c in df.columns}
        cols["collision_energy"] = df[s.collision_energy]
        return pd.DataFrame(cols)

    def read_annotation(self, zip_filename: str, max_members: int | None = None) -> pd.DataFrame:
        """Fetch an annotation .zip through the cache and read its parquet member(s).

        Each PROSPECT pool ships annotations as a zip of per-raw-file parquet shards;
        ``max_members`` caps how many shards to read (handy for a quick look).
        """
        zpath = self.resolve_file(zip_filename)
        frames = []
        with zipfile.ZipFile(zpath) as z:
            members = [n for n in z.namelist() if n.endswith(".parquet")]
            for name in members[:max_members] if max_members else members:
                frames.append(pd.read_parquet(io.BytesIO(z.read(name))))
        if not frames:
            raise ValueError(f"no parquet members in {zip_filename}")
        return pd.concat(frames, ignore_index=True)

    def annotation_shard_info(self, zip_filename: str) -> list[ShardInfo]:
        """Name and size of every parquet shard in an annotation zip, WITHOUT downloading it.

        A zip's central directory sits at the end of the file, so a couple of range requests
        list the whole archive — names, packed and unpacked sizes — for a few KB regardless of
        the zip's size. Verified against a 6.7 GB pool: 662 shards enumerated with no download.

        The sizes are the point. Shard count alone does not tell you what a run will cost:
        every selected shard is extracted to local parquet and re-read each epoch, so disk and
        per-epoch read time, not download time, are what bound how many shards a run can take.
        Check here before committing to a pool.
        """
        with self._open_remote_zip(zip_filename) as z:
            return [
                ShardInfo(i.filename, i.compress_size, i.file_size)
                for i in z.infolist()
                if i.filename.endswith(".parquet")
            ]

    def annotation_shards(self, zip_filename: str) -> list[str]:
        """Parquet shard names inside an annotation zip, without downloading it.

        See :meth:`annotation_shard_info` for the sizes, which is usually what you want when
        deciding how many shards to take.
        """
        return [s.name for s in self.annotation_shard_info(zip_filename)]

    def read_annotation_streaming(
        self, zip_filename: str, members: list[str] | None = None, max_members: int | None = None
    ) -> pd.DataFrame:
        """Read chosen annotation shards by RANGE-streaming the remote zip.

        Unlike :meth:`read_annotation`, this never materializes the whole zip locally: the
        central directory plus only the requested members' bytes are fetched. Use it to train
        on a slice of a huge pool (e.g. one 90 MB shard of a 1.5 GB zip) without the full pull.
        ``members`` selects shards by name; else the first ``max_members`` (or all) are read.
        """
        frames = []
        with self._open_remote_zip(zip_filename) as z:
            names = [n for n in z.namelist() if n.endswith(".parquet")]
            chosen = (
                members if members is not None else (names[:max_members] if max_members else names)
            )
            for name in chosen:
                frames.append(pd.read_parquet(io.BytesIO(z.read(name))))
        if not frames:
            raise ValueError(f"no parquet members read from {zip_filename}")
        return pd.concat(frames, ignore_index=True)

    def _open_remote_zip(self, zip_filename: str) -> zipfile.ZipFile:
        """Open the record's zip as a seekable remote file (range requests), or local if cached."""
        local = self.cache._local_path(f"zenodo/{self.record_id}/{zip_filename}")
        if os.path.exists(local):
            return zipfile.ZipFile(local)
        entry = self._catalog.get(zip_filename)
        url = entry["url"] if entry else self.zenodo.file_url(zip_filename)
        return zipfile.ZipFile(fsspec.open(url, "rb").open())

    def to_labels(
        self, meta_df: pd.DataFrame, ann_df: pd.DataFrame, split: SplitConfig | None = None
    ) -> "RealLabels":
        """Decode meta + long-format annotation into real student examples.

        Kept for the in-memory path (tests, small one-shot decodes). The streaming path builds
        its MetaIndex once per run instead of once per call and goes straight to
        ``decode_fragments``; both share that one scatter implementation.
        """
        from .meta_index import MetaIndex, SpectrumMeta  # local: avoids a circular import

        split = split or SplitConfig()
        s = self.schema
        for col in (s.modified_sequence, s.charge, s.raw_file, s.scan_number):
            if col not in meta_df.columns:
                raise ValueError(f"meta missing {col!r}")
        irt_col = (
            s.indexed_retention_time
            if s.indexed_retention_time in meta_df.columns
            else s.retention_time
        )
        raw_col = s.retention_time if s.retention_time in meta_df.columns else irt_col

        meta_u = meta_df.drop_duplicates([s.raw_file, s.scan_number], keep="first")
        index = MetaIndex()
        parsed: dict[str, tuple[str, tuple]] = {}
        for row in meta_u.itertuples(index=False):
            modseq = str(getattr(row, s.modified_sequence))
            if modseq not in parsed:
                parsed[modseq] = parse_modseq(modseq)
            stripped, mods = parsed[modseq]
            rf = str(getattr(row, s.raw_file))
            andromeda_v = getattr(row, s.andromeda_score, None)
            index.by_key[(rf, int(getattr(row, s.scan_number)))] = SpectrumMeta(
                peptide=Peptide(stripped, mods),
                charge=int(getattr(row, s.charge)),
                irt=float(getattr(row, irt_col)),
                raw_rt=float(getattr(row, raw_col)),
                split=assign_split(stripped, split),
                mass_analyzer=str(getattr(row, s.mass_analyzer, "")),
                fragmentation=str(getattr(row, s.fragmentation, "")),
                energy=float(getattr(row, s.collision_energy, float("nan"))),
                andromeda=float(andromeda_v) if andromeda_v is not None else float("nan"),
            )

        kept = ann_df.loc[fragment_filter_mask(ann_df, s)]
        real, _keys = decode_fragments(index, kept, s)
        return real


@dataclass
class RealLabels:
    """Decoded real examples. ``labels[i].rt`` is iRT (context-free base target); ``raw_rt``
    is the run-dependent retention time (the ``chrom_context`` target). ``source_ids`` (raw_file) is the
    context stratification key; ``acquisition`` maps raw_file -> analyzer/fragmentation/NCE,
    so per-raw_file context vectors can later be regressed onto those factors."""

    precursors: list
    labels: list
    raw_rt: list
    source_ids: list
    acquisition: dict
