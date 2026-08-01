"""Per-epoch streaming reader over extracted PROSPECT annotation shards.

Train examples are never materialised. Each epoch walks the shards in a seeded order, reads
each one row-group by row-group with a seven-column projection, applies the fragment filter and
the split filter, reassembles spectra, and feeds a shuffle buffer.

Two facts shape this:

- A spectrum's fragments can straddle a row-group boundary, so filtered rows are accumulated
  for a whole shard before the scatter runs. That accumulation is bounded: the filter keeps
  10-35% of rows depending on pool, and the projection drops the 81% of shard bytes that
  ``experimental_mass``/``theoretical_mass``/``fragment_score`` occupy. Reading is done
  row-group by row-group via ``ParquetFile.read_row_group`` (not ``iter_batches``, whose default
  batch size silently coalesces many row groups into one and would defeat the bounded-memory
  point on a real multi-hundred-MB shard).
- ``peptide_sequence`` is excluded from the projection even though the split filter is about
  sequences: it duplicates the meta's ``modified_sequence`` and costs 27% of the projected
  read. The filter keys on ``scan_number`` (0.3% of bytes) against a set from the meta index.

Residency is one SHARD, not one batch: a shard is decoded whole and its ``RealExample`` objects
(~60k for third-pool shard 0) stay alive until the shuffle buffer drains them. The bound this
module offers is "one shard at a time", never "one batch at a time".
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq
import torch

from ..data.meta_index import MetaIndex
from ..data.prospect import ProspectSchema, decode_fragments, fragment_filter_mask
from ..models.context import MSContextEncoder
from .context_regime import RealBatch, RealExample, RealSpeclibDataset

# The only columns the scatter needs. raw_file IS read — a shard can hold several, and the
# column is 226 bytes compressed in a 90.7 MB shard. peptide_sequence is not: it duplicates the
# meta's modified_sequence and is 27% of the projected read.
STREAM_COLUMNS: tuple[str, ...] = (
    "ion_type", "no", "charge", "intensity", "neutral_loss", "scan_number", "raw_file",
)

# The only values SpectrumMeta.split ever holds (see data/split.py's assign_split). It answers
# "does this shard match ANY meta row at all, in any split" -- a question distinct from "does it
# match the requested split", and the two are decided in one pass (_allowed_keys_for_shard).
_ALL_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


@dataclass(frozen=True, slots=True)
class ShardSpec:
    """One extracted shard and the per-source facts that apply to every example in it."""

    path: str
    raw_files: tuple[str, ...]  # plural: a third-pool shard holds several (see shard_raw_files)
    dataset_id: int  # ChromRunbook row (0 reserved for neutral/iRT)
    instrument: str


def _raise_zero_usable(shard_path: str, splits: frozenset[str]) -> None:
    raise ValueError(
        f"shard {shard_path!r} decoded to zero usable examples for splits "
        f"{sorted(splits)}; a real shard always yields some, so this is a damaged export"
    )


def _allowed_keys_for_shard(
    shard: ShardSpec, index: MetaIndex, splits: frozenset[str]
) -> set[tuple[str, int]]:
    """The shard's ``(raw_file, scan_number)`` keys whose split is in ``splits``.

    ONE pass over ``index.by_key``, not two. This used to be two ``MetaIndex.allowed_keys``
    calls — one to build the answer, one whose only purpose was to ask "does this shard match
    any meta row at all", discarding the set it built to do so. Against the merged index of a
    six-shard run (338,734 entries) that was 2 x 6 x epochs full dictionary scans; the answer
    is identical every epoch because the index and the splits are immutable for the run, so the
    caller caches it per shard (see :class:`StreamingRealDataset`).

    Raises if ``shard.raw_files`` matches no meta row in ANY split: that means the raw_files
    were derived wrong (e.g. from the filename instead of ``shard_raw_files``) or the tuple is
    empty, not that the shard is legitimately empty for the requested split. A shard that
    matches meta rows but none in ``splits`` returns an empty set instead -- that emptiness is
    real.
    """
    wanted = set(shard.raw_files)
    allowed: set[tuple[str, int]] = set()
    matched_any = False
    for key, sm in index.by_key.items():
        if key[0] not in wanted or sm.split not in _ALL_SPLITS:
            continue
        matched_any = True
        if sm.split in splits:
            allowed.add(key)
    if not matched_any:
        raise ValueError(
            f"shard {shard.path!r} raw_files {shard.raw_files!r} match no meta rows in any "
            "split; raw_files was likely derived from the filename instead of "
            "shard_raw_files(path), or is empty -- this shard is misconfigured, not merely "
            "outside the requested split"
        )
    return allowed


def _examples_from_shard(
    shard: ShardSpec,
    index: MetaIndex,
    encoder: MSContextEncoder,
    splits: frozenset[str],
    schema: ProspectSchema,
    only_keys: set[tuple[str, int]] | None = None,
    strict: bool = True,
    allowed: set[tuple[str, int]] | None = None,
) -> list[RealExample]:
    """Read one shard row-group by row-group and decode it into examples.

    ``only_keys`` narrows further than ``splits`` — the val path passes the winner scans, so a
    val_only source decodes just those instead of everything it would then discard.

    ``allowed`` is the shard's split-filtered key set; pass a cached one to skip the index scan
    (see :func:`_allowed_keys_for_shard`). It is never mutated here.

    ``strict`` controls what "no surviving rows" means. On the train path it is a damaged
    export — a whole shard read for a whole split cannot legitimately scatter to nothing — so
    it raises. On the val path ``only_keys`` has already narrowed the read to a handful of
    hand-picked winner scans that may not live in this shard at all, so ``strict=False`` makes
    that ordinary emptiness return ``[]``.

    Acquisition factors are resolved PER SPECTRUM from its own meta row. They used to be one
    value per raw file taken from its first spectrum, but a PROSPECT raw file mixes analyzers,
    fragmentation modes and NCE within itself, so that assigned a fabricated factor to about
    half the spectra of a mixed run. Categorical ids are memoised per distinct value, so this
    costs a dict lookup per example, not an embedding lookup.
    """
    if allowed is None:
        allowed = _allowed_keys_for_shard(shard, index, splits)
    if only_keys is not None:
        allowed = allowed & only_keys  # new set: `allowed` may be the caller's cached one
    if not allowed:
        return []
    pf = pq.ParquetFile(shard.path)
    kept: list[pd.DataFrame] = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=list(STREAM_COLUMNS)).to_pandas()
        mask = fragment_filter_mask(df, schema)
        if mask.any():
            sub = df.loc[mask]
            midx = pd.MultiIndex.from_arrays([sub[schema.raw_file], sub[schema.scan_number]])
            sub = sub.loc[midx.isin(allowed)]
            if not sub.empty:
                kept.append(sub)
    if not kept:
        if not strict:
            return []
        _raise_zero_usable(shard.path, splits)
    frag = pd.concat(kept, ignore_index=True)
    # concat has copied every row group's slice into `frag`; holding `kept` as well doubles the
    # filtered-fragment peak for as long as the scatter runs.
    kept.clear()
    real, keys = decode_fragments(index, frag, schema)
    if not real.precursors:
        if not strict:
            return []
        _raise_zero_usable(shard.path, splits)

    inst_id = encoder.instrument_id(shard.instrument)
    det_ids: dict[str, int] = {}
    frag_ids: dict[str, int] = {}
    out = []
    for p, lab, rrt, key in zip(real.precursors, real.labels, real.raw_rt, keys):
        sm = index.by_key[key]
        if sm.mass_analyzer not in det_ids:
            det_ids[sm.mass_analyzer] = encoder.detector_id(sm.mass_analyzer)
        if sm.fragmentation not in frag_ids:
            frag_ids[sm.fragmentation] = encoder.fragmentation_id(sm.fragmentation)
        out.append(
            RealExample(
                precursor=p, label=lab, raw_rt=float(rrt),
                instrument_id=inst_id,
                detector_id=det_ids[sm.mass_analyzer],
                fragmentation_id=frag_ids[sm.fragmentation],
                energy=sm.energy, dataset_id=shard.dataset_id,
            )
        )
    return out


def _collate(examples: list[RealExample], generator: torch.Generator) -> RealBatch:
    """Collate one already-sized chunk into a single :class:`RealBatch`.

    Builds a throwaway :class:`RealSpeclibDataset` sized so its own ``batches`` emits exactly
    one batch, and takes it. Not free (six numpy allocations per call), but it reuses the one
    collate implementation instead of a second copy.
    """
    return next(RealSpeclibDataset(examples).batches(len(examples), False, generator))


class StreamingRealDataset:
    """Streams :class:`RealExample` from extracted shards; exposes the ``batches`` protocol
    that :class:`~pepdistill.distill.dataset.BatchIterable` already wraps."""

    def __init__(
        self,
        shards: list[ShardSpec],
        index: MetaIndex,
        encoder: MSContextEncoder,
        splits: frozenset[str],
        seed: int = 0,
        shuffle_buffer: int = 50_000,
        schema: ProspectSchema | None = None,
    ) -> None:
        self.shards = shards
        self.index = index
        self.encoder = encoder
        self.splits = splits
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.schema = schema or ProspectSchema()
        # shard.path -> its split-filtered key set. The meta index and the splits are both
        # immutable for the run, so this is the same answer every epoch; recomputing it meant
        # a full scan of the MERGED index per shard per epoch (O(index x shards x epochs)) in
        # the one function that exists to run at per-epoch scale.
        self._allowed: dict[str, set[tuple[str, int]]] = {}

    def _allowed_for(self, shard: ShardSpec) -> set[tuple[str, int]]:
        cached = self._allowed.get(shard.path)
        if cached is None:
            cached = _allowed_keys_for_shard(shard, self.index, self.splits)
            self._allowed[shard.path] = cached
        return cached

    def iter_examples(self, epoch: int, shuffle: bool = True) -> Iterator[RealExample]:
        """Walk every shard once. ``shuffle=False`` is sequential — shard order untouched and
        the shuffle buffer disabled for this pass — regardless of the configured
        ``shuffle_buffer``, matching every other dataset in ``distill/`` where ``shuffle=False``
        means "in order" (an eval pass through ``BatchIterable(..., shuffle=False)`` relies on
        this)."""
        buf_size = self.shuffle_buffer if shuffle else 0
        # hash(), not `seed + epoch`: addition makes (seed=0, epoch=5) and (seed=5, epoch=0)
        # collide onto the same stream. hash() of an int tuple is stable across processes (only
        # str/bytes hashing is salted by PYTHONHASHSEED), so this stays reproducible.
        rng = random.Random(hash((self.seed, epoch)))
        order = list(self.shards)
        if buf_size:
            rng.shuffle(order)
        buf: list[RealExample] = []
        for shard in order:
            for ex in _examples_from_shard(
                shard, self.index, self.encoder, self.splits, self.schema,
                allowed=self._allowed_for(shard),
            ):
                if not buf_size:
                    yield ex
                    continue
                buf.append(ex)
                if len(buf) >= buf_size:
                    j = rng.randrange(len(buf))
                    buf[j], buf[-1] = buf[-1], buf[j]
                    yield buf.pop()
        rng.shuffle(buf)
        yield from buf

    def batches(
        self, batch_size: int, shuffle: bool, generator: torch.Generator
    ) -> Iterator[RealBatch]:
        """Collate the example stream into :class:`RealBatch`. ``shuffle`` is honoured by the
        shuffle buffer and the shard order, not by a permutation — there is no index to
        permute. ``generator`` seeds the epoch so repeated passes differ; it is only drawn from
        when ``shuffle`` is true, so an unshuffled pass never perturbs the caller's generator."""
        epoch = (
            int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
            if shuffle
            else 0
        )
        pending: list[RealExample] = []
        for ex in self.iter_examples(epoch, shuffle=shuffle):
            pending.append(ex)
            if len(pending) == batch_size:
                yield _collate(pending, generator)
                pending = []
        if pending:
            yield _collate(pending, generator)


def collect_val_examples(
    shards: list[ShardSpec],
    index: MetaIndex,
    encoder: MSContextEncoder,
    dataset_names: dict[int, str],
    schema: ProspectSchema | None = None,
    log=print,
) -> list[RealExample]:
    """Decode the val winners, at most one per (dataset, modified_sequence, charge).

    The winners are chosen by ``MetaIndex.val_winner_keys`` — argmax ``andromeda_score``, which
    lives in meta — so they are known before any fragment is read. Val is therefore a scan
    allowlist exactly like the split filter, and a val_only source decodes only its winners
    rather than everything it will later discard.

    Dedup is scoped PER DATASET, not per shard: a pool spans several shards (split is by
    sequence hash, independent of shard boundaries), so one dedup key can have observations in
    raw files that land in different shards. Shards are grouped by ``dataset_id`` first and
    ``val_winner_keys`` is called ONCE per dataset over the union of that dataset's raw files,
    so the argmax runs across every shard of the dataset instead of restarting per shard (which
    would silently emit one "winner" per shard for the same key).

    ``dataset_names`` must map every shard's ``dataset_id`` — a missing entry raises. The row
    is baked into the exported artifact and named in the checkpoint, so a shard whose row has
    no name means the shard selection and the dataset index disagree; there is nothing sensible
    to log the val set against, and nothing to name in the summary.

    "At most one", not "exactly one": the winner is picked from meta before any fragment is
    read, and a winner whose fragments scatter to an all-zero MS2 is dropped by
    ``decode_fragments`` with no fallback to the runner-up. That shortfall is logged per
    dataset rather than repaired — a ranked fallback would be a different design.

    The result is small (1,029 keys for a shard with 8.7M fragment rows) and is kept for the
    run: it makes the val set identical across every epoch by construction, and re-decoding the
    same handful of spectra 80 times would buy nothing.
    """
    schema = schema or ProspectSchema()
    by_dataset: dict[int, list[ShardSpec]] = {}
    for shard in shards:
        by_dataset.setdefault(shard.dataset_id, []).append(shard)

    out: list[RealExample] = []
    for dataset_id, group in by_dataset.items():
        if dataset_id not in dataset_names:
            raise KeyError(
                f"dataset_id={dataset_id!r} (shard path(s) {[s.path for s in group]!r}) has no "
                f"entry in dataset_names (known: {sorted(dataset_names)}); every shard's "
                "dataset_id is a ChromRunbook row that gets baked into the exported artifact, "
                "so an unnamed row means the shard selection and the dataset index disagree"
            )
        raw_files = [rf for shard in group for rf in shard.raw_files]
        winners = index.val_winner_keys(raw_files)
        if not winners:
            continue
        got = 0
        for shard in group:
            # strict=False: only_keys has narrowed this read to the winner scans, which need
            # not live in THIS shard of the dataset. Empty there is routine, not a damaged
            # export, and the strict message would name the wrong cause.
            decoded = _examples_from_shard(shard, index, encoder, frozenset({"val"}), schema,
                                           only_keys=winners, strict=False)
            got += len(decoded)
            out.extend(decoded)
        # Winners are chosen pre-decode, so one that scatters to an all-zero MS2 vanishes here
        # with no runner-up to fall back on. Say so rather than let the val set quietly shrink.
        if got < len(winners):
            log(
                f"[val] dataset {dataset_names[dataset_id]!r}: {got} of {len(winners)} winner "
                f"scan(s) decoded; {len(winners) - got} produced no usable spectrum and have "
                "no runner-up fallback"
            )
    return out


__all__ = ["ShardSpec", "StreamingRealDataset", "collect_val_examples", "STREAM_COLUMNS"]
