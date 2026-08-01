"""Per-epoch streaming reader over extracted PROSPECT annotation shards.

Train examples are never materialised. Each epoch walks the shards in a seeded order, reads
each one row-group by row-group with a six-column projection, applies the fragment filter and
the split filter, reassembles spectra, and feeds a shuffle buffer.

Two facts shape this:

- A spectrum's fragments can straddle a row-group boundary, so filtered rows are accumulated
  for a whole shard before the scatter runs. That accumulation is bounded: the filter keeps
  10-35% of rows depending on pool, and the projection drops the 81% of shard bytes that
  ``experimental_mass``/``theoretical_mass``/``fragment_score`` occupy.
- ``peptide_sequence`` is excluded from the projection even though the split filter is about
  sequences: it duplicates the meta's ``modified_sequence`` and costs 27% of the projected
  read. The filter keys on ``scan_number`` (0.3% of bytes) against a set from the meta index.
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


@dataclass(frozen=True, slots=True)
class ShardSpec:
    """One extracted shard and the per-source facts that apply to every example in it."""

    path: str
    raw_files: tuple[str, ...]  # plural: a third-pool shard holds several (see shard_raw_files)
    dataset_id: int  # ChromRunbook row (0 reserved for neutral/iRT)
    instrument: str


def _examples_from_shard(
    shard: ShardSpec,
    index: MetaIndex,
    encoder: MSContextEncoder,
    splits: frozenset[str],
    schema: ProspectSchema,
    only_keys: set[tuple[str, int]] | None = None,
) -> list[RealExample]:
    """Read one shard row-group by row-group and decode it into examples.

    ``only_keys`` narrows further than ``splits`` — the val path passes the winner scans, so a
    val_only source decodes just those instead of everything it would then discard.

    Acquisition factors are resolved PER SPECTRUM from its own meta row. They used to be one
    value per raw file taken from its first spectrum, but a PROSPECT raw file mixes analyzers,
    fragmentation modes and NCE within itself, so that assigned a fabricated factor to about
    half the spectra of a mixed run. Categorical ids are memoised per distinct value, so this
    costs a dict lookup per example, not an embedding lookup.
    """
    allowed = index.allowed_keys(list(shard.raw_files), splits)
    if only_keys is not None:
        allowed &= only_keys
    if not allowed:
        return []
    pf = pq.ParquetFile(shard.path)
    kept: list[pd.DataFrame] = []
    for batch in pf.iter_batches(columns=list(STREAM_COLUMNS)):
        df = batch.to_pandas()
        mask = fragment_filter_mask(df, schema)
        if mask.any():
            sub = df.loc[mask]
            keys = list(zip(sub[schema.raw_file], sub[schema.scan_number]))
            sub = sub.loc[[k in allowed for k in keys]]
            if not sub.empty:
                kept.append(sub)
    if not kept:
        raise ValueError(
            f"shard {shard.path!r} decoded to zero usable examples for splits "
            f"{sorted(splits)}; a real shard always yields some, so this is a damaged export"
        )
    real, keys = decode_fragments(index, pd.concat(kept, ignore_index=True), schema)
    if not real.precursors:
        raise ValueError(
            f"shard {shard.path!r} decoded to zero usable examples for splits "
            f"{sorted(splits)}; a real shard always yields some, so this is a damaged export"
        )

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

    def iter_examples(self, epoch: int) -> Iterator[RealExample]:
        rng = random.Random(self.seed + epoch)
        order = list(self.shards)
        if self.shuffle_buffer:
            rng.shuffle(order)
        buf: list[RealExample] = []
        for shard in order:
            for ex in _examples_from_shard(
                shard, self.index, self.encoder, self.splits, self.schema
            ):
                if not self.shuffle_buffer:
                    yield ex
                    continue
                buf.append(ex)
                if len(buf) >= self.shuffle_buffer:
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
        permute. ``generator`` seeds the epoch so repeated passes differ."""
        epoch = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        pending: list[RealExample] = []
        for ex in self.iter_examples(epoch if shuffle else 0):
            pending.append(ex)
            if len(pending) == batch_size:
                yield RealSpeclibDataset(pending).batches(
                    batch_size, False, generator
                ).__next__()
                pending = []
        if pending:
            yield RealSpeclibDataset(pending).batches(
                len(pending), False, generator
            ).__next__()


def collect_val_examples(
    shards: list[ShardSpec],
    index: MetaIndex,
    encoder: MSContextEncoder,
    dataset_names: dict[int, str],
    schema: ProspectSchema | None = None,
) -> list[RealExample]:
    """Decode the val winners, one per (dataset, modified_sequence, charge).

    The winners are chosen by ``MetaIndex.val_winner_keys`` — argmax ``andromeda_score``, which
    lives in meta — so they are known before any fragment is read. Val is therefore a scan
    allowlist exactly like the split filter, and a val_only source decodes only its winners
    rather than everything it will later discard.

    The result is small (1,029 keys for a shard with 8.7M fragment rows) and is kept for the
    run: it makes the val set identical across every epoch by construction, and re-decoding the
    same handful of spectra 80 times would buy nothing.
    """
    schema = schema or ProspectSchema()
    out: list[RealExample] = []
    for shard in shards:
        winners = index.val_winner_keys(
            list(shard.raw_files), dataset_names.get(shard.dataset_id)
        )
        if not winners:
            continue
        out.extend(
            _examples_from_shard(shard, index, encoder, frozenset({"val"}), schema,
                                 only_keys=winners)
        )
    return out


__all__ = ["ShardSpec", "StreamingRealDataset", "collect_val_examples", "STREAM_COLUMNS"]
