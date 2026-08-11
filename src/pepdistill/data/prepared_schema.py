"""Canonical schema contract for prepared real-spectrum Parquet assets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl


PREPARED_SPECTRA_SCHEMA = pl.Schema(
    {
        "spectrum_id": pl.UInt64,
        "dataset": pl.String,
        "raw_file": pl.String,
        "scan_number": pl.Int64,
        "sequence": pl.String,
        "mods": pl.String,
        "charge": pl.Int64,
        "split": pl.String,
        "irt": pl.Float64,
        "raw_rt": pl.Float64,
        "instrument": pl.String,
        "detector": pl.String,
        "fragmentation": pl.String,
        "energy": pl.Float64,
        "andromeda_score": pl.Float64,
        "ms2": pl.List(pl.Float32),
    }
)
VALIDATION_WINNER_SCHEMA = pl.Schema({"spectrum_id": pl.UInt64})


def prepared_frame(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """Construct a prepared spectra table without inferring any column type."""
    return pl.DataFrame(rows, schema=PREPARED_SPECTRA_SCHEMA, strict=True)


def _canonical_projection(schema: pl.Schema) -> list[pl.Expr]:
    return [pl.col(name).cast(dtype, strict=True).alias(name) for name, dtype in schema.items()]


def canonical_prepared_scan(sources: str | Sequence[str]) -> pl.LazyFrame:
    """Read prepared shards through an explicit ordered projection and strict casts."""
    return pl.scan_parquet(sources, extra_columns="raise", missing_columns="raise").select(
        _canonical_projection(PREPARED_SPECTRA_SCHEMA)
    )


def _read_canonical_parquet(source: Any, expected: pl.Schema) -> pl.DataFrame:
    """Validate physical columns, read explicitly, and canonicalize legacy ID types."""
    physical = pl.read_parquet_schema(source)
    if list(physical) != list(expected):
        raise ValueError(
            f"prepared Parquet columns differ from the contract: "
            f"expected={list(expected)}, actual={list(physical)}"
        )
    for name, dtype in physical.items():
        if dtype == expected[name]:
            continue
        # Existing v1 shards inferred this per file. Accept it only at the read boundary.
        if name == "spectrum_id" and dtype in {pl.Int64, pl.Int128}:
            continue
        raise ValueError(
            f"prepared Parquet column {name!r} has type {dtype}; expected {expected[name]}"
        )
    if hasattr(source, "seek"):
        source.seek(0)
    frame = pl.read_parquet(source, schema=physical)
    return frame.select(_canonical_projection(expected))


def read_prepared_parquet(source: Any) -> pl.DataFrame:
    return _read_canonical_parquet(source, PREPARED_SPECTRA_SCHEMA)


def read_validation_winners(source: Any) -> pl.DataFrame:
    return _read_canonical_parquet(source, VALIDATION_WINNER_SCHEMA)


def require_schema(frame: pl.DataFrame, expected: pl.Schema, label: str) -> None:
    if frame.schema != expected:
        raise ValueError(f"{label} schema differs from the contract: {frame.schema} != {expected}")


__all__ = [
    "PREPARED_SPECTRA_SCHEMA",
    "VALIDATION_WINNER_SCHEMA",
    "canonical_prepared_scan",
    "prepared_frame",
    "read_prepared_parquet",
    "read_validation_winners",
    "require_schema",
]
