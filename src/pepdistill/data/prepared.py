"""Streaming reader for prepared, chunked real-speclib assets.

The preparation job writes one compact row per spectrum.  Fragment labels are stored as a
flattened ``ms2`` list, so training never needs to parse an annotation ZIP, build a global
metadata dictionary, or repeat fragment scattering.  Chunks are ordinary Parquet objects and
may live entirely on S3; the reader opens one object at a time and keeps the existing
``RealExample``/``BatchIterable`` protocol used by the Lightning trainer.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import fsspec
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import get_worker_info

from ..chem import ION_TYPES, Peptide
from ..data.encode import Batch, FRAG_OFFSET
from ..teacher.base import PrecursorLabels
from .precursors import Precursor
from .prepared_schema import read_prepared_parquet, read_validation_winners

if TYPE_CHECKING:
    from ..distill.context_regime import RealBatch, RealExample


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    uri: str
    dataset: str
    rows: int
    source_shard: str = ""


@dataclass(frozen=True, slots=True)
class PreparedManifest:
    version: int
    chunks: tuple[PreparedChunk, ...]
    datasets: dict[str, int]
    val_winners: frozenset[int] = frozenset()
    irt_stats: tuple[int, float, float] = (0, 0.0, 0.0)
    split_rows: dict[str, int] | None = None
    split_datasets: dict[str, frozenset[str]] | None = None

    @classmethod
    def load(cls, prefix: str | Path) -> "PreparedManifest":
        uri = f"{str(prefix).rstrip('/')}/manifest.json"
        with fsspec.open(uri, "rt") as stream:
            raw = json.load(stream)
        if int(raw.get("version", 0)) != 1:
            raise ValueError(f"unsupported prepared manifest version: {raw.get('version')!r}")
        chunks = tuple(
            PreparedChunk(
                uri=str(row["uri"]),
                dataset=str(row["dataset"]),
                rows=int(row.get("rows", 0)),
                source_shard=str(row.get("source_shard", "")),
            )
            for row in raw.get("chunks", [])
        )
        datasets = {str(name): int(row) for name, row in raw.get("datasets", {}).items()}
        if not chunks:
            raise ValueError(f"prepared manifest {uri!r} has no chunks")
        winners = frozenset(int(x) for x in raw.get("val_winners", []))
        winners_uri = raw.get("val_winners_uri")
        if not winners and winners_uri:
            with fsspec.open(str(winners_uri), "rb") as stream:
                winner_ids = read_validation_winners(stream)["spectrum_id"]
                winners = frozenset(int(x) for x in winner_ids.to_list())
        stats = raw.get("irt_stats", [0, 0.0, 0.0])
        split_rows = {str(k): int(v) for k, v in raw.get("split_rows", {}).items()}
        split_datasets = {
            str(name): frozenset(str(split) for split in splits)
            for name, splits in raw.get("split_datasets", {}).items()
        }
        return cls(
            version=1,
            chunks=chunks,
            datasets=datasets,
            val_winners=winners,
            irt_stats=(int(stats[0]), float(stats[1]), float(stats[2])),
            split_rows=split_rows,
            split_datasets=split_datasets,
        )


def _parse_site(token: str):
    return token if token in ("n", "c") else int(token)


def _parse_mods(serialized: str) -> tuple:
    if not serialized:
        return ()
    return tuple(
        (_parse_site(site), float(spec) if spec[:1] in "+-" else spec)
        for site, spec in (pair.split(":", 1) for pair in serialized.split(";"))
    )


def _open_parquet(uri: str):
    if "://" in uri:
        return fsspec.open(uri, "rb").open()
    return uri


def _row_examples(frame: pd.DataFrame, dataset_id: int, encoder) -> list[RealExample]:
    from ..distill.context_regime import RealExample

    out: list[RealExample] = []
    n_ions = len(ION_TYPES)
    for row in frame.itertuples(index=False):
        sequence = str(row.sequence)
        mods = _parse_mods(str(row.mods or ""))
        precursor = Precursor(
            peptide=Peptide(sequence, mods),
            charge=int(row.charge),
            split=str(row.split),
        )
        flat = np.asarray(row.ms2, dtype=np.float32)
        expected = max(len(sequence) - 1, 0) * n_ions
        if flat.size != expected:
            raise ValueError(
                f"prepared row {getattr(row, 'spectrum_id', '?')!r} has ms2 length {flat.size}; "
                f"expected {expected} for sequence length {len(sequence)}"
            )
        labels = PrecursorLabels(
            ms2=flat.reshape((max(len(sequence) - 1, 0), n_ions)),
            rt=float(row.irt),
            ccs=float("nan"),
        )
        out.append(
            RealExample(
                precursor=precursor,
                label=labels,
                raw_rt=float(row.raw_rt),
                instrument_id=encoder.instrument_id(str(row.instrument)),
                detector_id=encoder.detector_id(str(row.detector)),
                fragmentation_id=encoder.fragmentation_id(str(row.fragmentation)),
                energy=float(row.energy),
                dataset_id=dataset_id,
            )
        )
    return out


def _frame_batch(frame: pl.DataFrame, encoder) -> RealBatch:
    """Collate one prepared frame directly, without a Pandas/RealExample round trip."""
    from ..distill.context_regime import RealBatch
    from ..distill.dataset import LabeledBatch, MSFactors

    import pepdistill_rs as _rs

    sequences = frame["sequence"].to_list()
    arrays = _rs.collate_prepared(
        sequences,
        frame["mods"].fill_null("").to_list(),
        frame["charge"].to_list(),
    )
    inputs = Batch(
        torch.from_numpy(arrays["tokens"]),
        torch.from_numpy(arrays["mod_comp"]),
        torch.from_numpy(arrays["mod_mass"]),
        torch.from_numpy(arrays["mod_present"]),
        torch.from_numpy(arrays["mod_named"]),
        torch.from_numpy(arrays["charge"]),
        torch.from_numpy(arrays["lengths"]),
        torch.from_numpy(arrays["pad_mask"]),
        torch.from_numpy(arrays["frag_mask"]),
    )
    lengths = [len(sequence) for sequence in sequences]
    sites = max(lengths[0] - 1, 0)
    if any(length != lengths[0] for length in lengths):
        raise ValueError("prepared training batches must contain one peptide length")
    # Equal-length buckets also have equal-length fragment vectors. Convert the Arrow list
    # column directly to one dense array, avoiding a Python list per spectrum.
    flat_ms2 = (
        frame["ms2"].list.to_array(sites * len(ION_TYPES)).to_numpy().astype(np.float32, copy=True)
    )
    expected_shape = (frame.height, sites * len(ION_TYPES))
    if flat_ms2.shape != expected_shape:
        raise ValueError(
            f"prepared batch has ms2 shape {flat_ms2.shape}; expected {expected_shape}"
        )
    ms2 = torch.zeros(frame.height, inputs.frag_mask.shape[1], len(ION_TYPES), dtype=torch.float32)
    ms2[:, FRAG_OFFSET : FRAG_OFFSET + sites] = torch.from_numpy(
        flat_ms2.reshape((frame.height, sites, len(ION_TYPES)))
    )

    def float_tensor(name: str) -> torch.Tensor:
        # Polars may expose a read-only Arrow-backed view; own the small batch array before
        # handing it to PyTorch, whose tensors are mutable by contract.
        return torch.from_numpy(frame[name].to_numpy().astype(np.float32, copy=True))

    def id_tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64)

    factors = MSFactors(
        instrument_id=id_tensor([encoder.instrument_id(x) for x in frame["instrument"]]),
        detector_id=id_tensor([encoder.detector_id(x) for x in frame["detector"]]),
        fragmentation_id=id_tensor([encoder.fragmentation_id(x) for x in frame["fragmentation"]]),
        energy=float_tensor("energy"),
    )
    labeled = LabeledBatch(
        inputs=inputs,
        ms2_target=ms2,
        rt_target=float_tensor("irt"),
        ccs_target=torch.full((frame.height,), float("nan"), dtype=torch.float32),
    )
    return RealBatch(
        base=labeled,
        raw_rt=float_tensor("raw_rt"),
        dataset_id=torch.from_numpy(frame["_dataset_id"].to_numpy().astype(np.int64, copy=True)),
        ms_factors=factors,
    )


class PreparedStreamingDataset:
    """Manifest-backed streaming dataset for the real-spectrum training stage."""

    # ``fit_realspeclib_datasets`` uses this marker to reject multiprocessing for generic
    # iterable sources that would otherwise replay the complete dataset in every worker.
    worker_partitioned = True

    def __init__(
        self,
        manifest: PreparedManifest,
        encoder,
        splits: frozenset[str],
        seed: int = 0,
        row_group_size: int = 65_536,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.manifest = manifest
        self.encoder = encoder
        self.splits = splits
        self.seed = seed
        self.row_group_size = row_group_size
        self.log = log
        self.dataset_ids = manifest.datasets

    def __len__(self) -> int:
        if self.manifest.split_rows is not None:
            return sum(self.manifest.split_rows.get(split, 0) for split in self.splits)
        return sum(chunk.rows for chunk in self.manifest.chunks)

    @property
    def dataset_ids_present(self) -> frozenset[int]:
        if self.manifest.split_datasets:
            return frozenset(
                self.dataset_ids[name]
                for name, splits in self.manifest.split_datasets.items()
                if self.splits.intersection(splits)
            )
        return frozenset(self.dataset_ids[chunk.dataset] for chunk in self.manifest.chunks)

    def _iter_frames(self, epoch: int, shuffle: bool) -> Iterator[pl.DataFrame]:
        chunks = list(self.manifest.chunks)
        rng = random.Random(hash((self.seed, epoch)))
        if shuffle:
            rng.shuffle(chunks)
        indexed_chunks = list(enumerate(chunks, start=1))
        worker = get_worker_info()
        if worker is not None:
            # Every worker receives the same epoch/shuffle seed, then takes a disjoint stride
            # through that common ordering. This preserves exactly-once shard coverage while
            # allowing S3 reads, Parquet decode, row conversion, and collation to overlap.
            indexed_chunks = indexed_chunks[worker.id :: worker.num_workers]
        for chunk_index, chunk in indexed_chunks:
            dataset_id = self.dataset_ids[chunk.dataset]
            opened_at = time.perf_counter()
            stream = _open_parquet(chunk.uri)
            open_seconds = time.perf_counter() - opened_at
            try:
                # PROSPECT hashes occupy the full unsigned 64-bit range. Polars therefore
                # persisted some shard IDs as Int128; PyArrow refuses to open a Parquet schema
                # containing Int128 even when that column is not selected. Read with Polars,
                # use the ID only for validation selection, and drop it before Pandas conversion.
                read_at = time.perf_counter()
                frame = read_prepared_parquet(stream).filter(
                    pl.col("split").is_in(self.splits)
                    & pl.col("irt").is_finite()
                    & pl.col("raw_rt").is_finite()
                )
                if self.splits == frozenset({"val"}) and self.manifest.val_winners:
                    keep = [
                        int(value) in self.manifest.val_winners
                        for value in frame["spectrum_id"].to_list()
                    ]
                    # An empty split produces ``keep=[]``. Without an explicit dtype Polars
                    # constructs a Null Series, which is not a valid filter predicate and used
                    # to crash validation at the first train-only shard after a long epoch.
                    frame = frame.filter(pl.Series("is_winner", keep, dtype=pl.Boolean))
                # Transformer cost is set by the longest sequence in each batch. Grouping equal
                # lengths inside an already-shuffled shard removes padding work without adding
                # a large row-level shuffle buffer or changing which observations are trained.
                frame = frame.with_columns(
                    pl.col("sequence").str.len_chars().alias("_sequence_length")
                ).sort("_sequence_length")
                frame = frame.drop("spectrum_id")
                read_seconds = time.perf_counter() - read_at
                if self.log is not None:
                    size = getattr(stream, "size", None)
                    size_mib = float(size) / (1024 * 1024) if size is not None else None
                    transfer = (
                        f", size={size_mib:.1f} MiB, effective={size_mib / read_seconds:.1f} MiB/s"
                        if size_mib is not None and read_seconds > 0
                        else ""
                    )
                    self.log(
                        f"[data] shard {chunk_index:,}/{len(chunks):,}, dataset={chunk.dataset}, "
                        f"rows={frame.height:,}, open={open_seconds:.3f}s, "
                        f"read_decode={read_seconds:.3f}s{transfer}"
                    )
                frame = frame.with_columns(pl.lit(dataset_id, dtype=pl.Int64).alias("_dataset_id"))
                for offset in range(0, frame.height, self.row_group_size):
                    yield frame.slice(offset, self.row_group_size)
            finally:
                if hasattr(stream, "close"):
                    stream.close()

    def iter_examples(self, epoch: int, shuffle: bool = True) -> Iterator[RealExample]:
        for frame in self._iter_frames(epoch, shuffle):
            pandas_frame = frame.drop("_dataset_id", "_sequence_length").to_pandas()
            dataset_id = int(frame["_dataset_id"][0])
            yield from _row_examples(pandas_frame, dataset_id, self.encoder)

    def batches(
        self, batch_size: int, shuffle: bool, generator: torch.Generator
    ) -> Iterator[RealBatch]:
        epoch = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()) if shuffle else 0
        pending: dict[int, pl.DataFrame] = {}
        for frame in self._iter_frames(epoch, shuffle):
            for group in frame.partition_by("_sequence_length", maintain_order=True):
                length = int(group["_sequence_length"][0])
                carried = pending.pop(length, None)
                if carried is not None:
                    needed = batch_size - carried.height
                    taken = min(needed, group.height)
                    carried = pl.concat([carried, group.head(taken)], rechunk=False)
                    group = group.slice(taken)
                    if carried.height == batch_size:
                        yield _frame_batch(carried, self.encoder)
                    else:
                        pending[length] = carried
                        continue
                full_rows = group.height - group.height % batch_size
                for offset in range(0, full_rows, batch_size):
                    yield _frame_batch(group.slice(offset, batch_size), self.encoder)
                if full_rows < group.height:
                    pending[length] = group.slice(full_rows)
        for length in sorted(pending):
            yield _frame_batch(pending[length], self.encoder)


__all__ = ["PreparedChunk", "PreparedManifest", "PreparedStreamingDataset"]
