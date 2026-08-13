"""PROSPECT schema and fragment decoding used by the prepared-data ETL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pepdistill_rs as _rs

from ..chem import ION_TYPES
from ..teacher.base import PrecursorLabels
from .precursors import Precursor

if TYPE_CHECKING:
    from .meta_index import MetaIndex


_UNIMOD_TOKEN = re.compile(r"([A-Z])|\[UNIMOD:(\d+)\]")
# The ProForma terminal separator is matched by position, not as another token alternative. Both
# forms must read back -- PROSPECT writes `[UNIMOD:737]SEQ`, our own `modified_sequence()` writes
# the compliant `[UNIMOD:737]-SEQ` -- but a `-` accepted anywhere is worse than one accepted
# nowhere: a trailing `-[UNIMOD:21]` then attached to the preceding residue instead of the
# C-terminus, and `PEP-TIDE` parsed as `PEPTIDE`. Both are wrong peptides with plausible fragments.
_NTERM_PREFIX = re.compile(r"^\[UNIMOD:(\d+)\]-?")
_CTERM_SUFFIX = re.compile(r"-\[UNIMOD:(\d+)\]$")


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

    # Peel the terminal modifications off by position first, so the body below can reject every
    # remaining `-` rather than guessing which residue a stray one belongs to.
    body = modseq
    cterm_accession: int | None = None
    nterm_accession: int | None = None
    if (suffix := _CTERM_SUFFIX.search(body)) is not None:
        cterm_accession = int(suffix.group(1))
        body = body[: suffix.start()]
    if (prefix := _NTERM_PREFIX.match(body)) is not None:
        nterm_accession = int(prefix.group(1))
        body = body[prefix.end() :]

    consumed = 0
    for m in _UNIMOD_TOKEN.finditer(body):
        # Unmatched text between tokens is not skippable: the residue alternative matches bare
        # capitals, so a descriptor this reader does not understand gets partly reinterpreted as
        # residues. `AC[Carbamidomethyl@C]DEK` silently became `ACCCDEK` with the modification
        # dropped, and `P[+79.96633]EPTIDE` became an unmodified `PEPTIDE` -- a wrong peptide with
        # entirely plausible fragments. Require full coverage so an unsupported notation stops here.
        if m.start() != consumed:
            raise ValueError(
                f"cannot parse {modseq!r}: unexpected {body[consumed : m.start()]!r} at offset "
                f"{consumed} of its residue body. This reader accepts only ProForma UNIMOD "
                "accessions (`[UNIMOD:n]`), with '-' only between a terminal one and the sequence."
            )
        consumed = m.end()
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
            # A leading token in the body is still N-terminal: PROSPECT omits the separator, so
            # `[UNIMOD:737]ET...` never matched the prefix above.
            mods.append(("n" if pos < 0 else pos, name))
    if consumed != len(body):
        raise ValueError(
            f"cannot parse {modseq!r}: unexpected trailing {body[consumed:]!r}. This reader "
            "accepts only ProForma UNIMOD accessions (`[UNIMOD:n]`)."
        )

    def resolve(accession: int) -> str:
        name = _rs.unimod_name(accession)
        if name is None:
            raise ValueError(
                f"unknown UNIMOD accession {accession} in {modseq!r}: not in the vendored "
                "unimod table (regenerate with tools/gen_unimod.py)"
            )
        return name

    # Assembled in reading order -- N-terminal, residues, C-terminal -- so the result matches the
    # order the string presents rather than the order this function happened to peel them off in.
    ordered: list[tuple] = []
    if nterm_accession is not None:
        ordered.append(("n", resolve(nterm_accession)))
    ordered.extend(mods)
    if cterm_accession is not None:
        ordered.append(("c", resolve(cterm_accession)))
    return "".join(residues), tuple(ordered)


@dataclass(frozen=True, slots=True)
class ProspectSchema:
    """Column-name mapping. Defaults follow the documented PROSPECT columns; override per
    file if a variant differs (validated at read time)."""

    # Defaults verified against real PROSPECT metadata. Meta files carry no stripped `sequence`
    # column. They do carry precursor-level intensity used for chromatographic curation; fragment
    # intensities remain in the long annotation Parquet inside the archive.
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
    precursor_intensity: str = "precursor_intensity"

    # Columns required to even treat a file as PROSPECT (identity + acquisition context).
    def required(self) -> list[str]:
        return [self.modified_sequence, self.charge, self.collision_energy]

    # Columns that define an acquisition "source" for context conditioning.
    def acquisition_factors(self) -> list[str]:
        return [self.mass_analyzer, self.fragmentation]


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
        _raw=raws,
        _scan=scans,
        _site=site,
        _col=col,
        _inten=frag[s.ann_intensity].to_numpy(dtype=np.float32),
    )[keep]
    agg = work.groupby(["_raw", "_scan", "_site", "_col"], sort=False)["_inten"].max().reset_index()
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
    """b/y ions, fragment charge 1-2, no neutral loss. Measured to keep 10-35% by pool.
    ``neutral_loss`` may arrive as ``category`` dtype (the streaming reader reads it
    dictionary-encoded to avoid materializing a Python string per row). ``.fillna("")`` raises
    on a categorical whose categories do not already include ``""`` -- true of any shard whose
    fragments all carry a neutral loss -- so the "no neutral loss" check is instead
    ``isna() | (col == "")``, which needs no such category to exist and is equivalent to the
    old ``fillna("") == ""`` for both ``object`` and ``category`` dtype.
    """
    no_loss = ann[schema.ann_neutral_loss]
    return (
        ann[schema.ann_ion_type].isin(("b", "y")).to_numpy()
        & ann[schema.ann_frag_charge].isin((1, 2)).to_numpy()
        & (no_loss.isna() | (no_loss == "")).to_numpy()
    )


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


__all__ = [
    "ProspectSchema",
    "parse_modseq",
    "decode_fragments",
    "fragment_filter_mask",
    "RealLabels",
]
