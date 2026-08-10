"""Polars-backed PROSPECT preparation.

This is deliberately a separate stage from training.  It scans one archive's already
extracted parquet members, applies the expensive fragment filter and metadata join once, and
writes one compact row per usable spectrum.  Training then streams those rows from a manifest
without reopening ZIPs or retaining a global metadata index.

The first implementation processes one annotation shard at a time.  That bounds memory even
when the source metadata file contains millions of rows; Polars still performs the projection,
predicate pushdown, and streaming Parquet scan for each shard.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import fsspec
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ..data.meta_index import build_meta_index_from_frame
from ..data.prospect import ProspectSchema, decode_fragments
from .config import PrepareConfig, PrepareGroup, PrepareSource


def _uri_join(prefix: str | Path, name: str) -> str:
    return f"{str(prefix).rstrip('/')}/{name.lstrip('/')}"


def _uri_exists(uri: str) -> bool:
    fs, _, paths = fsspec.get_fs_token_paths(uri)
    return bool(fs.exists(paths[0]))


def _load_json(uri: str) -> dict[str, Any] | None:
    try:
        with fsspec.open(uri, "rt") as stream:
            value = json.load(stream)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _list_shards(source_prefix: str | Path, archive_stem: str) -> list[str]:
    """List extracted members under ``<prefix>/<stem>/<stem>``."""
    root = _uri_join(source_prefix, f"{archive_stem}/{archive_stem}")
    if "://" not in root:
        return sorted(str(p) for p in Path(root).glob("*.parquet"))
    # Resolve the directory first.  Passing a glob to ``get_fs_token_paths`` can
    # expand it eagerly, leaving ``paths[0]`` as a concrete object and silently
    # reducing the result to one shard on some fsspec implementations.
    fs, _, paths = fsspec.get_fs_token_paths(root)
    pattern = f"{paths[0].rstrip('/')}/*.parquet"
    return sorted(fs.unstrip_protocol(path) for path in fs.glob(pattern))


def _serialize_mods(mods: tuple) -> str:
    def site(value: Any) -> str:
        return value if isinstance(value, str) else str(value)

    def spec(value: Any) -> str:
        return value if isinstance(value, str) else f"{float(value):+g}"

    return ";".join(f"{site(pos)}:{spec(name)}" for pos, name in mods)


def _spectrum_id(dataset: str, raw_file: str, scan: int) -> int:
    digest = hashlib.blake2b(
        f"{dataset}\0{raw_file}\0{scan}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


def _write_parquet(frame: pl.DataFrame, uri: str) -> None:
    if "://" not in uri:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(uri, compression="zstd", row_group_size=65_536)
        return
    with fsspec.open(uri, "wb") as stream:
        frame.write_parquet(stream, compression="zstd", row_group_size=65_536)


def _shard_raw_files(uri: str) -> list[str]:
    """Read the distinct raw-file keys without materializing annotation fragments."""
    if "://" in uri:
        with fsspec.open(uri, "rb") as stream:
            table = pq.read_table(stream, columns=["raw_file"], read_dictionary=["raw_file"])
    else:
        table = pq.read_table(uri, columns=["raw_file"], read_dictionary=["raw_file"])
    return [str(value) for value in table.column("raw_file").unique().to_pylist()]


def _meta_columns(meta_uri: str, schema: ProspectSchema) -> list[str]:
    required = [schema.raw_file, schema.scan_number, schema.modified_sequence, schema.charge]
    optional = [
        schema.retention_time,
        schema.indexed_retention_time,
        schema.collision_energy,
        schema.mass_analyzer,
        schema.fragmentation,
        schema.andromeda_score,
    ]
    available = set(pl.scan_parquet(meta_uri).collect_schema().names())
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"metadata {meta_uri!r} missing required columns {missing}")
    return [name for name in required + optional if name in available]


def _fragment_columns(schema: ProspectSchema) -> list[str]:
    return [
        schema.raw_file,
        schema.scan_number,
        schema.ann_ion_type,
        schema.ann_ordinal,
        schema.ann_frag_charge,
        schema.ann_intensity,
        schema.ann_neutral_loss,
    ]


def _rows_for_shard(
    meta_uri: str,
    shard_uri: str,
    raw_files: list[str],
    dataset: str,
    instrument: str,
    schema: ProspectSchema,
) -> list[dict[str, Any]]:
    available_meta = _meta_columns(meta_uri, schema)
    meta = (
        pl.scan_parquet(meta_uri)
        .select(available_meta)
        .filter(pl.col(schema.raw_file).is_in(raw_files))
        .collect(engine="streaming")
        .to_pandas()
    )
    if meta.empty:
        raise ValueError(
            f"metadata {meta_uri!r} has no rows for shard {shard_uri!r} raw_files={raw_files!r}"
        )
    index = build_meta_index_from_frame(meta, schema=schema)

    s = schema
    fragments = (
        pl.scan_parquet(shard_uri)
        .select(_fragment_columns(s))
        .filter(
            pl.col(s.ann_ion_type).is_in(["b", "y"])
            & pl.col(s.ann_frag_charge).is_in([1, 2])
            & (pl.col(s.ann_neutral_loss).is_null() | (pl.col(s.ann_neutral_loss) == ""))
        )
        .collect(engine="streaming")
        .to_pandas()
    )
    if fragments.empty:
        return []
    real, keys = decode_fragments(index, fragments, s)
    rows: list[dict[str, Any]] = []
    for precursor, label, raw_rt, key in zip(
        real.precursors, real.labels, real.raw_rt, keys, strict=True
    ):
        spectrum = index.by_key[key]
        rows.append(
            {
                "spectrum_id": _spectrum_id(dataset, key[0], key[1]),
                "dataset": dataset,
                "raw_file": key[0],
                "scan_number": key[1],
                "sequence": precursor.peptide.sequence,
                "mods": _serialize_mods(precursor.peptide.mods),
                "charge": int(precursor.charge),
                "split": precursor.split,
                "irt": float(label.rt),
                "raw_rt": float(raw_rt),
                "instrument": instrument,
                "detector": spectrum.mass_analyzer,
                "fragmentation": spectrum.fragmentation,
                "energy": float(spectrum.energy),
                "andromeda_score": float(spectrum.andromeda),
                "ms2": label.ms2.reshape(-1).astype(np.float32).tolist(),
            }
        )
    return rows


def _val_winners(chunks: list[str], out_uri: str) -> list[int]:
    if not chunks:
        return []
    values = (
        pl.scan_parquet(chunks)
        .filter(pl.col("split") == "val")
        .sort(
            ["dataset", "sequence", "charge", "andromeda_score", "spectrum_id"],
            descending=[False, False, False, True, False],
            nulls_last=True,
        )
        .unique(subset=["dataset", "sequence", "charge"], keep="first", maintain_order=True)
        .select("spectrum_id")
        .collect(engine="streaming")
    )
    winners = [int(value) for value in values["spectrum_id"].to_list()]
    _write_parquet(values, out_uri)
    return winners


def _irt_stats(chunks: list[str]) -> tuple[int, float, float]:
    row = (
        pl.scan_parquet(chunks)
        .filter(pl.col("split") == "train")
        .select(
            pl.len().alias("n"),
            pl.col("irt").sum().alias("sum"),
            (pl.col("irt") * pl.col("irt")).sum().alias("sumsq"),
        )
        .collect(engine="streaming")
        .row(0)
    )
    return int(row[0]), float(row[1] or 0.0), float(row[2] or 0.0)


def _split_rows(chunks: list[str]) -> dict[str, int]:
    rows = (
        pl.scan_parquet(chunks)
        .group_by("split")
        .agg(pl.len().alias("rows"))
        .collect(engine="streaming")
    )
    return {str(row["split"]): int(row["rows"]) for row in rows.to_dicts()}


def _split_datasets(chunks: list[str]) -> dict[str, list[str]]:
    rows = (
        pl.scan_parquet(chunks)
        .select(["dataset", "split"])
        .unique()
        .collect(engine="streaming")
    )
    result: dict[str, list[str]] = {}
    for row in rows.to_dicts():
        result.setdefault(str(row["dataset"]), []).append(str(row["split"]))
    return {name: sorted(splits) for name, splits in result.items()}


def _write_json(uri: str, value: dict[str, Any]) -> None:
    if "://" not in uri:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    with fsspec.open(uri, "wt") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def _source_prefix(config: PrepareConfig, source: PrepareSource) -> str:
    return source.source_prefix or config.source_prefix


def _list_archive_stems(prefix: str | Path) -> list[str]:
    """Discover extracted archive directories below a record prefix."""
    if "://" not in str(prefix):
        root = Path(prefix)
        return sorted(
            child.name
            for child in root.iterdir()
            if child.is_dir() and list((child / child.name).glob("*.parquet"))
        )
    fs, _, paths = fsspec.get_fs_token_paths(str(prefix))
    stems: list[str] = []
    for entry in fs.ls(paths[0], detail=True):
        raw_path = entry.get("name") if isinstance(entry, dict) else entry
        if raw_path is None or not fs.isdir(raw_path):
            continue
        name = str(raw_path).rstrip("/").rsplit("/", 1)[-1]
        try:
            if fs.glob(f"{str(raw_path).rstrip('/')}/{name}/*.parquet"):
                stems.append(name)
        except OSError:
            continue
    return sorted(stems)


def _group_dataset(group: PrepareGroup, archive: str) -> str:
    name = archive
    if group.strip_prefix and name.startswith(group.strip_prefix):
        name = name[len(group.strip_prefix) :]
    if group.strip_number_suffix:
        name = re.sub(r"_[1-9][0-9]*$", "", name)
    prefix = group.dataset_prefix or group.record
    return f"{prefix}_{name}".lower()


def _sources_from_groups(config: PrepareConfig) -> tuple[PrepareSource, ...]:
    sources: list[PrepareSource] = []
    for group in config.groups:
        prefix = group.source_prefix or _uri_join(config.source_prefix, group.record)
        for archive in _list_archive_stems(prefix):
            if not any(fnmatch.fnmatchcase(archive, pattern) for pattern in group.include):
                continue
            if any(fnmatch.fnmatchcase(archive, pattern) for pattern in group.exclude):
                continue
            sources.append(
                PrepareSource(
                    id=f"{group.record}_{archive}",
                    dataset=_group_dataset(group, archive),
                    meta=f"{archive}{group.meta_suffix}",
                    archive=archive,
                    instrument=group.instrument,
                    source_prefix=prefix,
                )
            )
    return tuple(sources)


def discover_catalog(config: PrepareConfig) -> dict[str, Any]:
    """Expand configured sources into a deterministic, globally numbered shard catalog."""
    tasks: list[dict[str, Any]] = []
    sources = (*config.sources, *_sources_from_groups(config))
    ids = [source.id for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError(f"prepare source ids must be unique after group expansion; got {ids}")
    for source in sources:
        prefix = _source_prefix(config, source)
        shards = _list_shards(prefix, source.archive)
        if source.shards != "all":
            invalid = [index for index in source.shards if index < 0 or index >= len(shards)]
            if invalid:
                raise ValueError(
                    f"source {source.id!r}: shard index(es) {invalid} outside 0..{len(shards) - 1}"
                )
            selected = [(index, shards[index]) for index in source.shards]
        else:
            selected = list(enumerate(shards))
        for shard_index, shard_uri in selected:
            tasks.append(
                {
                    "ordinal": len(tasks),
                    "source_id": source.id,
                    "dataset": source.dataset,
                    "meta_uri": _uri_join(prefix, source.meta),
                    "shard_uri": shard_uri,
                    "shard_index": shard_index,
                    "archive": source.archive,
                    "instrument": source.instrument,
                    "config_fingerprint": config.fingerprint,
                }
            )
    return {
        "version": 1,
        "config_fingerprint": config.fingerprint,
        "output_prefix": config.output_prefix,
        "tasks": tasks,
    }


def ensure_catalog(config: PrepareConfig, force: bool = False) -> dict[str, Any]:
    uri = _uri_join(config.output_prefix, "catalog.json")
    if not force:
        existing = _load_json(uri)
        if existing and existing.get("version") == 1 and existing.get("config_fingerprint") == config.fingerprint:
            return existing
    catalog = discover_catalog(config)
    _write_json(uri, catalog)
    return catalog


def _task_output_prefix(output_prefix: str, task: dict[str, Any]) -> str:
    return _uri_join(
        output_prefix,
        f"shards/{task['source_id']}/{int(task['shard_index']):06d}",
    )


def prepare_task(
    output_prefix: str,
    task: dict[str, Any],
    force: bool = False,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Prepare one catalog task; completion is committed by writing its manifest last."""
    prefix = _task_output_prefix(output_prefix, task)
    manifest_uri = _uri_join(prefix, "manifest.json")
    data_uri = _uri_join(prefix, "data.parquet")

    def emit(message: str) -> None:
        if log is not None:
            log(
                f"[prepare source={task['source_id']} shard={int(task['shard_index']):06d}]"
                f" {message}"
            )

    if not force:
        existing = _load_json(manifest_uri)
        if (
            existing
            and existing.get("version") == 1
            and existing.get("task", {}).get("config_fingerprint") == task["config_fingerprint"]
            and _uri_exists(str(existing.get("data_uri", "")))
        ):
            emit("complete manifest exists; skipping")
            return {**existing, "_skipped": True}

    started = time.perf_counter()
    emit("discovering raw files")
    raw_files = _shard_raw_files(task["shard_uri"])
    emit(f"{len(raw_files)} raw file(s); filtering and decoding")
    rows = _rows_for_shard(
        task["meta_uri"],
        task["shard_uri"],
        raw_files,
        task["dataset"],
        task["instrument"],
        ProspectSchema(),
    )
    if not rows:
        raise ValueError(f"task {task['source_id']}/{task['shard_index']} produced no usable spectra")
    frame = pl.DataFrame(rows).with_columns(pl.col("ms2").cast(pl.List(pl.Float32)))
    _write_parquet(frame, data_uri)
    emit(f"wrote {frame.height:,} spectra in {time.perf_counter() - started:.1f}s")
    manifest = {
        "version": 1,
        "task": task,
        "data_uri": data_uri,
        "rows": frame.height,
        "source_shard": Path(task["shard_uri"]).name,
    }
    _write_json(manifest_uri, manifest)
    return manifest


def prepare_range(
    config: PrepareConfig,
    start: int | None = None,
    stop: int | None = None,
    force: bool = False,
    log: Callable[[str], None] | None = print,
) -> list[dict[str, Any]]:
    catalog = ensure_catalog(config)
    tasks = catalog["tasks"]
    start = 0 if start is None else start
    stop = len(tasks) if stop is None else stop
    if not 0 <= start <= stop <= len(tasks):
        raise ValueError(f"range {start}:{stop} outside catalog of {len(tasks)} shard task(s)")
    if log is not None:
        log(f"[prepare] catalog has {len(tasks):,} shard task(s); processing [{start}:{stop})")
    return [prepare_task(config.output_prefix, task, force=force, log=log) for task in tasks[start:stop]]


def finalize_catalog(config: PrepareConfig, log: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Validate all shard manifests and write the worker-independent training manifest."""
    catalog = ensure_catalog(config)
    manifests: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in catalog["tasks"]:
        uri = _uri_join(_task_output_prefix(config.output_prefix, task), "manifest.json")
        manifest = _load_json(uri)
        if manifest is None or not _uri_exists(str(manifest.get("data_uri", ""))):
            missing.append(f"{task['source_id']}/{int(task['shard_index']):06d}")
        else:
            manifests.append(manifest)
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"cannot finalize: {len(missing)} shard task(s) missing ({preview})")

    chunks = [
        {
            "uri": manifest["data_uri"],
            "dataset": manifest["task"]["dataset"],
            "rows": int(manifest["rows"]),
            "source_shard": manifest["source_shard"],
        }
        for manifest in manifests
    ]
    chunk_uris = [chunk["uri"] for chunk in chunks]
    validation_uri = _uri_join(config.output_prefix, "validation/val_winners.parquet")
    winners = _val_winners(chunk_uris, validation_uri)
    datasets = sorted({chunk["dataset"] for chunk in chunks})
    manifest = {
        "version": 1,
        "catalog_uri": _uri_join(config.output_prefix, "catalog.json"),
        "datasets": {name: index for index, name in enumerate(datasets, start=1)},
        "chunks": chunks,
        "val_winners": [],
        "val_winners_uri": validation_uri,
        "irt_stats": list(_irt_stats(chunk_uris)),
        "split_rows": _split_rows(chunk_uris),
        "split_datasets": _split_datasets(chunk_uris),
        "source": {"config_fingerprint": config.fingerprint, "shards": len(chunks)},
    }
    _write_json(_uri_join(config.output_prefix, "manifest.json"), manifest)
    if log is not None:
        log(f"[prepare] finalized {len(chunks):,} shard(s), {len(winners):,} validation winners")
    return manifest


def catalog_status(config: PrepareConfig) -> dict[str, int]:
    catalog = ensure_catalog(config)
    complete = 0
    for task in catalog["tasks"]:
        uri = _uri_join(_task_output_prefix(config.output_prefix, task), "manifest.json")
        manifest = _load_json(uri)
        if manifest is not None and _uri_exists(str(manifest.get("data_uri", ""))):
            complete += 1
    return {"complete": complete, "missing": len(catalog["tasks"]) - complete, "total": len(catalog["tasks"])}
