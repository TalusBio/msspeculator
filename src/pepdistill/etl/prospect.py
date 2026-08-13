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

from bisect import bisect_left
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from collections.abc import Callable
from typing import Any

import fsspec
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ..data.meta_index import build_meta_index_from_frame
from ..data.prepared_schema import (
    PREPARED_SPECTRA_SCHEMA,
    VALIDATION_WINNER_SCHEMA,
    canonical_prepared_scan,
    require_schema,
)
from ..data.prospect import ProspectSchema, decode_fragments
from ..data.prospect_catalog import load_catalog, load_shard_index
from .config import PrepareConfig, PrepareGroup, PrepareSource

_CATALOG_VERSION = 2


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


def _load_json_paths(fs: Any, paths: list[str]) -> dict[str, dict[str, Any]]:
    """Read explicit JSON paths using the filesystem's batched/concurrent implementation."""
    if not paths:
        return {}
    blobs = fs.cat(paths, on_error="return")
    if not isinstance(blobs, dict):
        blobs = {paths[0]: blobs}
    loaded: dict[str, dict[str, Any]] = {}
    for path, blob in blobs.items():
        if isinstance(blob, BaseException):
            continue
        try:
            value = json.loads(blob)
        except (TypeError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            loaded[str(path)] = value
    return loaded


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
    digest = hashlib.blake2b(f"{dataset}\0{raw_file}\0{scan}".encode(), digest_size=8).digest()
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
        schema.precursor_intensity,
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
                "precursor_intensity": float(spectrum.precursor_intensity),
            }
        )
    return rows


def _val_winners(chunks: list[str], out_uri: str) -> list[int]:
    if not chunks:
        return []
    values = (
        canonical_prepared_scan(chunks)
        .filter(
            (pl.col("split") == "val") & pl.col("irt").is_finite() & pl.col("raw_rt").is_finite()
        )
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
    require_schema(values, VALIDATION_WINNER_SCHEMA, "validation winners")
    _write_parquet(values, out_uri)
    return winners


def _irt_stats(chunks: list[str]) -> tuple[int, float, float]:
    row = (
        canonical_prepared_scan(chunks)
        .filter(
            (pl.col("split") == "train") & pl.col("irt").is_finite() & pl.col("raw_rt").is_finite()
        )
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
        canonical_prepared_scan(chunks)
        .filter(pl.col("irt").is_finite() & pl.col("raw_rt").is_finite())
        .group_by("split")
        .agg(pl.len().alias("rows"))
        .collect(engine="streaming")
    )
    return {str(row["split"]): int(row["rows"]) for row in rows.to_dicts()}


def _split_datasets(chunks: list[str]) -> dict[str, list[str]]:
    rows = (
        canonical_prepared_scan(chunks)
        .filter(pl.col("irt").is_finite() & pl.col("raw_rt").is_finite())
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


def _source_prefix(config: PrepareConfig, source: PrepareSource) -> str | None:
    return source.source_prefix or config.cache_prefix or config.source_prefix


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


def _group_metadata_name(group: PrepareGroup, archive: str) -> str:
    """Return the record-level metadata filename for an archive.

    Numbered PROSPECT ZIPs are physical partitions of one logical pool. For example,
    ``TUM_isoform_1.zip`` and ``TUM_isoform_2.zip`` both join against
    ``TUM_isoform_meta_data.parquet``.
    """
    stem = re.sub(r"_[1-9][0-9]*$", "", archive) if group.strip_number_suffix else archive
    return f"{stem}{group.meta_suffix}"


def _sources_from_groups(config: PrepareConfig) -> tuple[PrepareSource, ...]:
    catalog = load_catalog()["records"]
    sources: list[PrepareSource] = []
    for group in config.groups:
        if group.record not in catalog:
            raise ValueError(f"unknown PROSPECT record {group.record!r}; known: {sorted(catalog)}")
        record = catalog[group.record]
        prefix = (
            group.cache_prefix or group.source_prefix or config.cache_prefix or config.source_prefix
        )
        for archive_filename, entry in sorted(record["files"].items()):
            if not archive_filename.endswith(".zip"):
                continue
            archive = archive_filename.removesuffix(".zip")
            if not any(fnmatch.fnmatchcase(archive, pattern) for pattern in group.include):
                continue
            if any(fnmatch.fnmatchcase(archive, pattern) for pattern in group.exclude):
                continue
            meta_name = _group_metadata_name(group, archive)
            meta_entry = record["files"].get(meta_name)
            if not isinstance(meta_entry, dict) or not meta_entry.get("url"):
                raise ValueError(
                    f"PROSPECT catalog has no metadata {meta_name!r} for "
                    f"{group.record}/{archive_filename}"
                )
            sources.append(
                PrepareSource(
                    id=f"{group.record}_{archive}",
                    dataset=_group_dataset(group, archive),
                    meta=meta_name,
                    archive=archive,
                    instrument=group.instrument,
                    source_prefix=prefix,
                    record=group.record,
                    record_id=str(record["record_id"]),
                    archive_url=str(entry["url"]),
                    meta_url=str(meta_entry["url"]),
                )
            )
    return tuple(sources)


def _cached_shard_uri(prefix: str | None, source: PrepareSource, member: str) -> str:
    if not prefix or source.record is None:
        return ""
    return _uri_join(
        prefix,
        f"shards/{source.record or 'local'}/{source.archive}/{Path(member).name}",
    )


def _upload_cache(local: str, uri: str) -> None:
    if not uri or "://" not in uri:
        return
    fs, _, paths = fsspec.get_fs_token_paths(uri)
    try:
        fs.makedirs(paths[0].rsplit("/", 1)[0], exist_ok=True)
        fs.put_file(local, paths[0])
    except Exception:
        # The cache is an optimization. A read-through failure must not make Zenodo unusable.
        return


def _download_origin(url: str, local: str) -> None:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    temporary = f"{local}.part"
    with fsspec.open(url, "rb") as source, open(temporary, "wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    os.replace(temporary, local)


def _resolve_meta(task: dict[str, Any]) -> str:
    cached = str(task.get("meta_uri", ""))
    if cached and _uri_exists(cached):
        return cached
    url = str(task.get("meta_url", ""))
    if not url:
        raise FileNotFoundError(f"metadata unavailable for {task['source_id']}")
    record = str(task.get("record") or "local")
    local = str(
        Path(tempfile.gettempdir()) / "pepdistill-origin" / record / Path(cached or url).name
    )
    if not Path(local).exists():
        _download_origin(url, local)
    if cached:
        _upload_cache(local, cached)
    return local


def _resolve_shard(task: dict[str, Any]) -> tuple[str, str | None]:
    """Return a readable shard path and an optional temporary directory to remove."""
    cached = str(task.get("shard_uri", ""))
    if cached and _uri_exists(cached):
        return cached, None
    prefix = str(task.get("cache_prefix", ""))
    if prefix and task.get("record"):
        legacy = _uri_join(
            prefix, f"{task['archive']}/{task['archive']}/{Path(task['shard_name']).name}"
        )
        if _uri_exists(legacy):
            return legacy, None
    url = str(task.get("archive_url", ""))
    member = str(task.get("shard_name", ""))
    if not url or not member:
        raise FileNotFoundError(f"shard unavailable for {task['source_id']}/{task['shard_index']}")
    temporary_dir = tempfile.mkdtemp(prefix="pepdistill-shard-")
    local = str(Path(temporary_dir) / Path(member).name)
    with fsspec.open(url, "rb") as stream, zipfile.ZipFile(stream) as archive:
        with archive.open(member) as source, open(local, "wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if cached:
        _upload_cache(local, cached)
    return local, temporary_dir


def discover_catalog(config: PrepareConfig) -> dict[str, Any]:
    """Expand configured sources into a deterministic, globally numbered shard catalog."""
    tasks: list[dict[str, Any]] = []
    sources = (*config.sources, *_sources_from_groups(config))
    ids = [source.id for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError(f"prepare source ids must be unique after group expansion; got {ids}")
    for source in sources:
        prefix = _source_prefix(config, source)
        if source.record is not None:
            indexed = (
                load_shard_index()["records"].get(source.record, {}).get(f"{source.archive}.zip")
            )
            if not isinstance(indexed, list):
                raise ValueError(
                    f"no vendored shard index for {source.record}/{source.archive}.zip; "
                    "regenerate prospect_shards.json"
                )
            shards = [str(row[0]) for row in indexed]
        else:
            if prefix is None:
                raise ValueError(f"source {source.id!r} needs source_prefix for local discovery")
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
                    "meta_uri": _uri_join(prefix, source.meta) if prefix else "",
                    "shard_uri": _cached_shard_uri(prefix, source, shard_uri) or shard_uri,
                    "cache_prefix": prefix or "",
                    "shard_name": shard_uri,
                    "shard_index": shard_index,
                    "archive": source.archive,
                    "instrument": source.instrument,
                    "record": source.record,
                    "record_id": source.record_id,
                    "archive_url": source.archive_url,
                    "meta_url": source.meta_url,
                    "config_fingerprint": config.fingerprint,
                    "curation": config.curation.canonical(),
                }
            )
    return {
        "version": _CATALOG_VERSION,
        "config_fingerprint": config.fingerprint,
        "output_prefix": config.output_prefix,
        "tasks": tasks,
    }


def ensure_catalog(config: PrepareConfig, force: bool = False) -> dict[str, Any]:
    uri = _uri_join(config.output_prefix, "catalog.json")
    if not force:
        existing = _load_json(uri)
        if (
            existing
            and existing.get("version") == _CATALOG_VERSION
            and existing.get("config_fingerprint") == config.fingerprint
        ):
            return existing
    catalog = discover_catalog(config)
    _write_json(uri, catalog)
    return catalog


def _task_output_prefix(output_prefix: str, task: dict[str, Any]) -> str:
    return _uri_join(
        output_prefix,
        f"shards/{task['source_id']}/{int(task['shard_index']):06d}",
    )


def _catalog_manifests(
    config: PrepareConfig,
    catalog: dict[str, Any],
    log: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load and validate completed task manifests without per-object HEAD requests."""
    fs, _, roots = fsspec.get_fs_token_paths(_uri_join(config.output_prefix, "shards"))
    if log is not None:
        log(f"[prepare] listing completed shard assets for {len(catalog['tasks']):,} task(s)")
    try:
        existing = set(fs.find(roots[0]))
    except FileNotFoundError:
        existing = set()

    assets: list[tuple[dict[str, Any], str, str]] = []
    for task in catalog["tasks"]:
        prefix = _task_output_prefix(config.output_prefix, task)
        assets.append(
            (
                task,
                fs._strip_protocol(_uri_join(prefix, "manifest.json")),
                fs._strip_protocol(_uri_join(prefix, "data.parquet")),
            )
        )
    manifest_paths = [
        path for _, path, data_path in assets if path in existing and data_path in existing
    ]
    if log is not None:
        log(f"[prepare] reading {len(manifest_paths):,} manifest(s) in a concurrent batch")
    loaded = _load_json_paths(fs, manifest_paths)

    manifests: list[dict[str, Any]] = []
    missing: list[str] = []
    for task, manifest_path, data_path in assets:
        manifest = loaded.get(manifest_path)
        expected_data_uri = _uri_join(
            _task_output_prefix(config.output_prefix, task), "data.parquet"
        )
        valid = (
            manifest is not None
            and manifest.get("version") == 1
            and data_path in existing
            and manifest.get("data_uri") == expected_data_uri
            and manifest.get("task", {}).get("config_fingerprint") == config.fingerprint
        )
        if valid:
            manifests.append(manifest)
        else:
            missing.append(f"{task['source_id']}/{int(task['shard_index']):06d}")
    return manifests, missing


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
    shard_uri, temporary_dir = _resolve_shard(task)
    try:
        meta_uri = _resolve_meta(task)
        raw_files = _shard_raw_files(shard_uri)
        emit(f"{len(raw_files)} raw file(s); filtering and decoding")
        rows = _rows_for_shard(
            meta_uri,
            shard_uri,
            raw_files,
            task["dataset"],
            task["instrument"],
            ProspectSchema(),
        )
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
    if not rows:
        raise ValueError(
            f"task {task['source_id']}/{task['shard_index']} produced no usable spectra"
        )
    from .curation import curate_prepared_frame, curation_input_frame

    curation_input = curation_input_frame(rows)
    curation = task.get("curation", {})
    curation_report = None
    if curation.get("enabled", False):
        policy = {key: value for key, value in curation.items() if key != "enabled"}
        analysis = curate_prepared_frame(curation_input, **policy)
        frame = analysis.selected
        curation_report = analysis.report
        emit(
            f"curation retained {frame.height:,}/{curation_input.height:,} spectra from "
            f"{analysis.report['selection']['qualifying_peptidoforms']:,} peptidoforms"
        )
    else:
        frame = curation_input.select(
            [
                pl.col(name).cast(dtype, strict=True)
                for name, dtype in PREPARED_SPECTRA_SCHEMA.items()
            ]
        )
    _write_parquet(frame, data_uri)
    emit(f"wrote {frame.height:,} spectra in {time.perf_counter() - started:.1f}s")
    manifest = {
        "version": 1,
        "task": task,
        "data_uri": data_uri,
        "rows": frame.height,
        "source_shard": Path(task["shard_uri"]).name,
    }
    if curation_report is not None:
        manifest["curation"] = curation_report
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
    return [
        prepare_task(config.output_prefix, task, force=force, log=log) for task in tasks[start:stop]
    ]


def balanced_partition_range(
    catalog: dict[str, Any], index: int, partitions: int
) -> tuple[int, int, int]:
    """Return a contiguous range balanced by vendored raw shard bytes.

    The output assets remain ordinal-addressed and worker-independent. This only chooses array
    boundaries; explicit ``--range`` continues to address the exact same global catalog.
    """
    tasks = catalog["tasks"]
    if not 0 < partitions <= len(tasks):
        raise ValueError(f"partitions must be between 1 and {len(tasks)}")
    if not 0 <= index < partitions:
        raise ValueError(f"partition index {index} outside 0..{partitions - 1}")
    shard_index = load_shard_index()["records"]
    weights: list[int] = []
    for task in tasks:
        rows = shard_index.get(str(task.get("record")), {}).get(f"{task.get('archive')}.zip", [])
        shard_ordinal = int(task.get("shard_index", -1))
        row = (
            rows[shard_ordinal]
            if isinstance(rows, list) and 0 <= shard_ordinal < len(rows)
            else None
        )
        # Raw Parquet bytes are a better proxy for decode work than task count. Local/custom
        # sources lack a vendored central-directory row and retain equal-task weighting.
        weights.append(max(1, int(row[2])) if isinstance(row, list) and len(row) >= 3 else 1)
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    boundaries = [0]
    for part in range(1, partitions):
        target = prefix[-1] * part / partitions
        low = boundaries[-1] + 1
        high = len(tasks) - (partitions - part)
        after = bisect_left(prefix, target, low, high + 1)
        candidates = [candidate for candidate in (after - 1, after) if low <= candidate <= high]
        boundaries.append(min(candidates, key=lambda candidate: abs(prefix[candidate] - target)))
    boundaries.append(len(tasks))
    start, stop = boundaries[index : index + 2]
    return start, stop, prefix[stop] - prefix[start]


def finalize_catalog(
    config: PrepareConfig, log: Callable[[str], None] | None = print
) -> dict[str, Any]:
    """Validate all shard manifests and write the worker-independent training manifest."""
    catalog = ensure_catalog(config)
    manifests, missing = _catalog_manifests(config, catalog, log=log)
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


def catalog_status(config: PrepareConfig, count_only: bool = False) -> dict[str, int]:
    catalog = ensure_catalog(config)
    if count_only:
        return {"complete": 0, "missing": len(catalog["tasks"]), "total": len(catalog["tasks"])}
    manifests, _ = _catalog_manifests(config, catalog)
    complete = len(manifests)
    return {
        "complete": complete,
        "missing": len(catalog["tasks"]) - complete,
        "total": len(catalog["tasks"]),
    }
