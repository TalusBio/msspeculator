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

import fsspec
import numpy as np
import pandas as pd

from ..chem import ION_TYPES, MOD_DELTA, Peptide
from ..teacher.base import PrecursorLabels
from .cache import FileCache, default_cache_dir, http_origin
from .config import SplitConfig
from .precursors import Precursor
from .prospect_catalog import load_catalog
from .split import assign_split
from .zenodo import ZenodoRecord

_UNIMOD_TOKEN = re.compile(r"([A-Z])|\[UNIMOD:(\d+)\]")

# Ingest-side translation: external UNIMOD accession -> OUR mod name (chem.MOD_DELTA). This
# is the only place PROSPECT's identifier scheme is known; our chemistry stays decoupled.
# Accessions absent here parse to a "UNIMOD:N" sentinel (not in MOD_DELTA) so to_labels skips
# those peptides rather than silently mis-massing them.
_UNIMOD_TO_NAME: dict[int, str] = {
    4: "Carbamidomethyl@C",
    21: "Phospho",
    35: "Oxidation@M",
    737: "TMT6plex",
}


def parse_modseq(modseq: str) -> tuple[str, tuple[tuple, ...]]:
    """Parse a ProForma UNIMOD string -> (stripped_sequence, mods) in OUR mod names.

    ``[UNIMOD:737]ET[UNIMOD:21]TLHLVLR`` -> ("ETTLHLVLR", (("n","TMT6plex"),(1,"Phospho"))).
    A mod token attaches to the residue it follows; a leading token (before any residue) is
    routed to the N-terminal site. Unknown accessions map to a "UNIMOD:N" sentinel (not in
    chem.MOD_DELTA).
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
            site = "n" if pos < 0 else pos
            mods.append((site, _UNIMOD_TO_NAME.get(n, f"UNIMOD:{n}")))
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

    def annotation_shards(self, zip_filename: str) -> list[str]:
        """List parquet shard names inside an annotation zip WITHOUT downloading it.

        Reads only the zip central directory via HTTP range requests (a few KB), so a
        multi-GB pool can be inspected — and a subset chosen — before pulling any spectra.
        """
        with self._open_remote_zip(zip_filename) as z:
            return [n for n in z.namelist() if n.endswith(".parquet")]

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

    def iter_annotation_shards(self, zip_filename: str, indices):
        """Yield ``(name, DataFrame)`` for each shard index from a SINGLE open of the remote zip.

        Same range-streaming as :meth:`read_annotation_streaming`, but opens the zip once for
        all requested shards (instead of reopening — re-reading the central directory — per
        shard), while still yielding one shard at a time so the caller can decode and release
        each before the next. Bounded memory over a multi-shard pull.
        """
        with self._open_remote_zip(zip_filename) as z:
            names = [n for n in z.namelist() if n.endswith(".parquet")]
            for i in indices:
                yield names[i], pd.read_parquet(io.BytesIO(z.read(names[i])))

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

        Joins on (raw_file, scan_number); keeps only b/y ions at fragment charge 1-2 with no
        neutral loss; places each fragment intensity at its (site, ion) cell. RT is the
        indexed_retention_time; CCS is NaN — PROSPECT has no ion mobility, so a real-data
        regime must not supervise CCS from it. Each example carries its ``raw_file`` as the
        context-stratification source id, plus per-run acquisition metadata.
        """
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
        n_ion = len(ION_TYPES)

        # De-dup meta on the join key (first wins), parse each unique mod-string ONCE
        # (parse_modseq is the only per-peptide Python cost; rows reuse the cached result).
        meta_u = meta_df.drop_duplicates([s.raw_file, s.scan_number], keep="first")
        parsed = {ms: parse_modseq(str(ms)) for ms in meta_u[s.modified_sequence].unique()}
        n_of = {ms: len(v[0]) for ms, v in parsed.items()}
        ok_mod = {  # peptide is usable: >=2 residues and every mod maps to our chemistry
            ms: (len(v[0]) >= 2 and all(nm in MOD_DELTA for _, nm in v[1]))
            for ms, v in parsed.items()
        }

        acq_cols = [
            f for f in ("mass_analyzer", "fragmentation") if getattr(s, f) in meta_df.columns
        ]
        has_ce = s.collision_energy in meta_df.columns
        # Scalar meta columns to attach to each fragment row — no object columns keeps the merge fast.
        carry = {s.raw_file, s.scan_number, s.modified_sequence, s.charge, irt_col, raw_col}
        carry.update(getattr(s, f) for f in acq_cols)
        if has_ce:
            carry.add(s.collision_energy)

        # Vectorized fragment filter: b/y ions, fragment charge 1-2, no neutral loss.
        keep = (
            ann_df[s.ann_ion_type].isin(("b", "y"))
            & ann_df[s.ann_frag_charge].isin((1, 2))
            & (ann_df[s.ann_neutral_loss].fillna("") == "")
        )
        frag = ann_df.loc[
            keep,
            [
                s.raw_file,
                s.scan_number,
                s.ann_ion_type,
                s.ann_ordinal,
                s.ann_frag_charge,
                s.ann_intensity,
            ],
        ].merge(meta_u[list(carry)], on=[s.raw_file, s.scan_number], how="inner")

        # (site, col) computed for every surviving fragment at once. ION_TYPES order is
        # b1,y1,b2,y2 -> col = (b?0:1) + 2*(z-1); b site = ord-1, y site = n-1-ord.
        is_b = frag[s.ann_ion_type].to_numpy() == "b"
        z = frag[s.ann_frag_charge].to_numpy().astype(np.int64)
        ordinal = frag[s.ann_ordinal].to_numpy().astype(np.int64)
        n_arr = frag[s.modified_sequence].map(n_of).to_numpy(dtype=np.int64)
        modok = frag[s.modified_sequence].map(ok_mod).to_numpy(dtype=bool)
        site = np.where(is_b, ordinal - 1, n_arr - 1 - ordinal)
        col = np.where(is_b, 0, 1) + 2 * (z - 1)
        mask = modok & (site >= 0) & (site < n_arr - 1)
        frag = frag.assign(
            _site=site, _col=col, _inten=frag[s.ann_intensity].to_numpy(dtype=np.float32)
        )[mask]

        # Max-collapse duplicate (site, col) cells in C, then scatter per spectrum.
        agg = (
            frag.groupby([s.raw_file, s.scan_number, "_site", "_col"], sort=False)["_inten"]
            .max()
            .reset_index()
        )

        keymeta = frag.drop_duplicates([s.raw_file, s.scan_number], keep="first")
        meta_at = {
            (str(r), int(sc)): i
            for i, (r, sc) in enumerate(
                zip(keymeta[s.raw_file].to_numpy(), keymeta[s.scan_number].to_numpy())
            )
        }
        km_seq = keymeta[s.modified_sequence].to_numpy()
        km_charge = keymeta[s.charge].to_numpy()
        km_irt = keymeta[irt_col].to_numpy()
        km_raw = keymeta[raw_col].to_numpy()
        km_ce = keymeta[s.collision_energy].to_numpy() if has_ce else None
        km_acq = {f: keymeta[getattr(s, f)].to_numpy() for f in acq_cols}

        precursors: list[Precursor] = []
        labels: list[PrecursorLabels] = []
        raw_rt: list[float] = []
        source_ids: list[str] = []
        acquisition: dict[str, dict] = {}
        for (raw_file, scan), grp in agg.groupby([s.raw_file, s.scan_number], sort=False):
            i = meta_at.get((str(raw_file), int(scan)))
            if i is None:
                continue
            stripped, mods = parsed[km_seq[i]]
            ms2 = np.zeros((n_of[km_seq[i]] - 1, n_ion), dtype=np.float32)
            ms2[grp["_site"].to_numpy(), grp["_col"].to_numpy()] = grp["_inten"].to_numpy()
            if not ms2.any():
                continue
            precursors.append(
                Precursor(Peptide(stripped, mods), int(km_charge[i]), assign_split(stripped, split))
            )
            # labels.rt = iRT (context-free base target); raw_rt = run-dependent target.
            labels.append(PrecursorLabels(ms2=ms2, rt=float(km_irt[i]), ccs=float("nan")))
            raw_rt.append(float(km_raw[i]))
            rf = str(raw_file)
            source_ids.append(rf)
            if rf not in acquisition:
                acquisition[rf] = {f: km_acq[f][i] for f in acq_cols} | {
                    "collision_energy": float(km_ce[i]) if has_ce else float("nan")
                }
        return RealLabels(precursors, labels, raw_rt, source_ids, acquisition)


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


def merge_real_labels(parts: list[RealLabels]) -> RealLabels:
    """Concatenate decoded shards into one :class:`RealLabels` (union of acquisition maps).

    Lets a full pool load shard-by-shard — decode each, merge, drop the raw annotation — so
    peak memory stays at one shard rather than the whole multi-GB zip decompressed at once.
    """
    precursors: list = []
    labels: list = []
    raw_rt: list = []
    source_ids: list = []
    acquisition: dict = {}
    for p in parts:
        precursors += list(p.precursors)
        labels += list(p.labels)
        raw_rt += list(p.raw_rt)
        source_ids += list(p.source_ids)
        acquisition.update(p.acquisition)
    return RealLabels(precursors, labels, raw_rt, source_ids, acquisition)
