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

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ..data.meta_index import build_meta_index_from_frame
from ..data.prospect import ProspectSchema, decode_fragments


def _uri_join(prefix: str | Path, name: str) -> str:
    return f"{str(prefix).rstrip('/')}/{name.lstrip('/')}"


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


def prepare_source(
    source_prefix: str | Path,
    meta_filename: str,
    archive_stem: str,
    out_prefix: str | Path,
    dataset: str,
    instrument: str = "Lumos",
    max_shards: int | None = None,
    schema: ProspectSchema | None = None,
) -> dict[str, Any]:
    """Prepare one extracted PROSPECT archive and write a version-1 manifest."""
    schema = schema or ProspectSchema()
    meta_uri = _uri_join(source_prefix, meta_filename)
    shards = _list_shards(source_prefix, archive_stem)
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise ValueError(f"no extracted parquet shards found for {archive_stem!r}")

    chunk_rows: list[dict[str, Any]] = []
    chunk_uris: list[str] = []
    for index, shard_uri in enumerate(shards):
        raw_files = _shard_raw_files(shard_uri)
        rows = _rows_for_shard(meta_uri, shard_uri, raw_files, dataset, instrument, schema)
        if not rows:
            continue
        frame = pl.DataFrame(rows).with_columns(pl.col("ms2").cast(pl.List(pl.Float32)))
        chunk_uri = _uri_join(out_prefix, f"chunks/{dataset}/{index:06d}.parquet")
        _write_parquet(frame, chunk_uri)
        chunk_uris.append(chunk_uri)
        chunk_rows.append(
            {
                "uri": chunk_uri,
                "dataset": dataset,
                "rows": frame.height,
                "source_shard": Path(shard_uri).name,
            }
        )

    if not chunk_rows:
        raise ValueError(f"no usable spectra found while preparing {archive_stem!r}")
    winners_uri = _uri_join(out_prefix, "val_winners.parquet")
    _val_winners(chunk_uris, winners_uri)
    stats = _irt_stats(chunk_uris)
    split_rows = _split_rows(chunk_uris)
    split_datasets = _split_datasets(chunk_uris)
    manifest = {
        "version": 1,
        "datasets": {dataset: 1},
        "chunks": chunk_rows,
        "val_winners": [],
        "val_winners_uri": winners_uri,
        "irt_stats": list(stats),
        "split_rows": split_rows,
        "split_datasets": split_datasets,
        "source": {
            "prefix": str(source_prefix),
            "meta": meta_filename,
            "archive": f"{archive_stem}.zip",
        },
    }
    manifest_uri = _uri_join(out_prefix, "manifest.json")
    with fsspec.open(manifest_uri, "wt") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--meta", required=True, help="metadata filename at source prefix root")
    parser.add_argument("--archive-stem", required=True, help="extracted archive directory stem")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--instrument", default="Lumos")
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    manifest = prepare_source(
        source_prefix=args.source_prefix,
        meta_filename=args.meta,
        archive_stem=args.archive_stem,
        out_prefix=args.out_prefix,
        dataset=args.dataset,
        instrument=args.instrument,
        max_shards=args.max_shards,
    )
    print(
        f"prepared {len(manifest['chunks'])} chunk(s), "
        f"{sum(row['rows'] for row in manifest['chunks']):,} spectra -> "
        f"{_uri_join(args.out_prefix, 'manifest.json')}"
    )


if __name__ == "__main__":
    main()
