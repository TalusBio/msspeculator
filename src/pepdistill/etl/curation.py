"""Measure candidate PSM-curation policies before deriving filtered training assets.

The PROSPECT metadata includes precursor intensity, but prepared v1 assets intentionally retain
only model inputs and targets. This module joins one prepared shard back to its source metadata
and evaluates an observed half-maximum window. It does not mutate or replace immutable prepared
assets; the output is evidence for choosing a corpus-wide policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..data.prepared_schema import read_prepared_parquet


CURATION_ANNOTATION_SCHEMA = pl.Schema(
    {
        "spectrum_id": pl.UInt64,
        "precursor_intensity": pl.Float64,
        "intensity_ratio": pl.Float64,
        "supports_half_max": pl.Boolean,
        "apex_window_start": pl.Float64,
        "apex_window_end": pl.Float64,
        "within_apex_window": pl.Boolean,
        "context_psms": pl.Int64,
        "context_window_psms": pl.Int64,
        "selection_rank": pl.Int64,
        "selected": pl.Boolean,
        "selection_reason": pl.String,
    }
)

_SPECTRUM_KEY = ["raw_file", "scan_number"]
_PEPTIDOFORM_KEY = ["raw_file", "sequence", "mods"]
_CONTEXT_KEY = _PEPTIDOFORM_KEY + ["charge", "detector", "fragmentation", "energy"]


@dataclass(frozen=True)
class CurationAnalysis:
    report: dict[str, Any]
    annotations: pl.DataFrame

    def write(self, report_path: str | Path, annotations_path: str | Path | None = None) -> None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.report, indent=2, sort_keys=True) + "\n")
        if annotations_path is not None:
            annotations_path = Path(annotations_path)
            annotations_path.parent.mkdir(parents=True, exist_ok=True)
            self.annotations.write_parquet(annotations_path, compression="zstd")


def _quantiles(series: pl.Series) -> dict[str, float | None]:
    return {
        f"p{int(q * 100):02d}": (
            float(value)
            if (value := series.quantile(q, interpolation="linear")) is not None
            else None
        )
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def _mean_leave_one_out_sa(frame: pl.DataFrame, predicate: str | None = None) -> float | None:
    """Mean SA to the other spectra of the same precursor/acquisition context.

    Leave-one-out avoids giving a spectrum credit merely because it contributed to its own
    consensus. Singleton contexts do not have an agreement measurement.
    """
    values: list[np.ndarray] = []
    for group in frame.partition_by(_CONTEXT_KEY, maintain_order=False):
        spectra = np.asarray(group["ms2"].to_list(), dtype=np.float64)
        norms = np.linalg.norm(spectra, axis=1)
        valid = norms > 0
        if int(valid.sum()) < 2:
            continue
        unit = np.zeros_like(spectra)
        unit[valid] = spectra[valid] / norms[valid, None]
        total = unit[valid].sum(axis=0)
        peers = total[None, :] - unit[valid]
        peer_norms = np.linalg.norm(peers, axis=1)
        measurable = peer_norms > 0
        cosine = (
            np.sum(unit[valid][measurable] * peers[measurable], axis=1) / peer_norms[measurable]
        )
        sa = 1.0 - 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)) / pi
        if predicate is not None:
            mask = group[predicate].fill_null(False).to_numpy().astype(bool)[valid][measurable]
            sa = sa[mask]
        if sa.size:
            values.append(sa)
    return float(np.concatenate(values).mean()) if values else None


def analyze_prepared_curation(
    prepared_path: str | Path,
    metadata_path: str | Path,
    *,
    half_max_fraction: float = 0.5,
    cap_per_context: int = 8,
) -> CurationAnalysis:
    """Analyze one prepared shard against source precursor-intensity metadata.

    A single FWHM-like interval is estimated per peptidoform and raw file, pooling all charge
    states and acquisition modes. Observations whose intensity supports the interval define its
    RT bounds; those same bounds are then applied to every mode, even when a row's own intensity
    is below half maximum. This is still sampled from PSM-triggered observations rather than a
    reconstructed continuous XIC. Selection is capped separately per supervised acquisition
    context, with a best-score fallback when that context has no in-window PSM.
    """
    if not 0.0 < half_max_fraction <= 1.0:
        raise ValueError("half_max_fraction must be in (0, 1]")
    if cap_per_context < 1:
        raise ValueError("cap_per_context must be positive")

    prepared = read_prepared_parquet(prepared_path)
    metadata_schema = pl.read_parquet_schema(metadata_path)
    required = {*_SPECTRUM_KEY, "precursor_intensity"}
    missing = sorted(required - set(metadata_schema))
    if missing:
        raise ValueError(f"source metadata is missing curation columns: {missing}")
    metadata = (
        pl.scan_parquet(metadata_path)
        .select(*_SPECTRUM_KEY, "precursor_intensity")
        .filter(pl.col("raw_file").is_in(prepared["raw_file"].unique().to_list()))
        # PROSPECT contains a few physically duplicated metadata records. The preparation index
        # likewise keeps the first row for a spectrum key, so mirror that deterministic rule.
        .unique(_SPECTRUM_KEY, keep="first", maintain_order=True)
        .collect(engine="streaming")
    )
    frame = prepared.join(metadata, on=_SPECTRUM_KEY, how="left", validate="1:1")
    frame = frame.with_columns(
        pl.col("precursor_intensity").max().over(_PEPTIDOFORM_KEY).alias("_apex_intensity"),
        pl.len().over(_CONTEXT_KEY).cast(pl.Int64).alias("_context_psms"),
    ).with_columns(
        (pl.col("precursor_intensity") / pl.col("_apex_intensity")).alias("_intensity_ratio")
    )
    frame = (
        frame.with_columns(
            (
                pl.col("_intensity_ratio").is_finite()
                & (pl.col("_intensity_ratio") >= half_max_fraction)
            )
            .fill_null(False)
            .alias("_supports_half_max")
        )
        .with_columns(
            pl.col("raw_rt")
            .filter(pl.col("_supports_half_max"))
            .min()
            .over(_PEPTIDOFORM_KEY)
            .alias("_apex_window_start"),
            pl.col("raw_rt")
            .filter(pl.col("_supports_half_max"))
            .max()
            .over(_PEPTIDOFORM_KEY)
            .alias("_apex_window_end"),
        )
        .with_columns(
            (
                pl.col("raw_rt").is_between(
                    pl.col("_apex_window_start"), pl.col("_apex_window_end"), closed="both"
                )
                & pl.col("_apex_window_start").is_finite()
                & pl.col("_apex_window_end").is_finite()
            )
            .fill_null(False)
            .alias("_within_apex_window")
        )
        .with_columns(
            pl.col("_within_apex_window")
            .sum()
            .over(_CONTEXT_KEY)
            .cast(pl.Int64)
            .alias("_context_window_psms")
        )
    )
    frame = frame.with_columns(
        pl.when(pl.col("_context_window_psms") > 0)
        .then(pl.col("_within_apex_window"))
        .otherwise(True)
        .alias("_eligible")
    ).sort(
        _CONTEXT_KEY + ["_eligible", "andromeda_score", "precursor_intensity", "spectrum_id"],
        descending=[False] * len(_CONTEXT_KEY) + [True, True, True, False],
        nulls_last=True,
    )
    frame = (
        frame.with_columns(
            pl.col("_eligible").cum_sum().over(_CONTEXT_KEY).cast(pl.Int64).alias("_selection_rank")
        )
        .with_columns(
            (pl.col("_eligible") & (pl.col("_selection_rank") <= cap_per_context)).alias(
                "_selected"
            )
        )
        .with_columns(
            pl.when(~pl.col("_selected"))
            .then(pl.lit("not_selected"))
            .when(pl.col("_context_window_psms") == 0)
            .then(pl.lit("context_best_score_fallback"))
            .otherwise(pl.lit("apex_window_top_score"))
            .alias("_selection_reason")
        )
    )

    peptidoform_groups = frame.group_by(_PEPTIDOFORM_KEY).agg(
        pl.len().cast(pl.Int64).alias("psms"),
        pl.col("_supports_half_max").sum().cast(pl.Int64).alias("half_max_support_psms"),
        pl.col("_within_apex_window").sum().cast(pl.Int64).alias("window_psms"),
    )
    context_groups = frame.group_by(_CONTEXT_KEY).agg(
        pl.len().cast(pl.Int64).alias("psms"),
        pl.col("_within_apex_window").sum().cast(pl.Int64).alias("window_psms"),
        pl.col("_selected").sum().cast(pl.Int64).alias("selected_psms"),
    )
    selected_rows = int(frame["_selected"].sum())
    support_rows = int(frame["_supports_half_max"].sum())
    window_rows = int(frame["_within_apex_window"].sum())
    report = {
        "policy": {
            "apex_scope": "raw_file+peptidoform (shared across charge and acquisition mode)",
            "context_scope": "peptidoform+charge+detector+fragmentation+energy",
            "half_max_fraction": half_max_fraction,
            "cap_per_context": cap_per_context,
            "fallback": "best Andromeda score per acquisition context",
            "caveat": "observed PSM-triggered intensities; not a continuous reconstructed XIC",
        },
        "input": {
            "rows": frame.height,
            "peptidoform_groups": peptidoform_groups.height,
            "context_groups": context_groups.height,
            "missing_precursor_intensity_rows": int(frame["precursor_intensity"].null_count()),
        },
        "replication": {
            "psms_per_peptidoform": _quantiles(peptidoform_groups["psms"]),
            "half_max_support_psms_per_peptidoform": _quantiles(
                peptidoform_groups["half_max_support_psms"]
            ),
            "window_psms_per_peptidoform": _quantiles(peptidoform_groups["window_psms"]),
            "window_psms_per_context": _quantiles(context_groups["window_psms"]),
        },
        "selection": {
            "half_max_support_rows": support_rows,
            "apex_window_rows": window_rows,
            "apex_window_fraction_of_rows": window_rows / frame.height if frame.height else 0.0,
            "selected_rows": selected_rows,
            "selected_fraction_of_rows": selected_rows / frame.height if frame.height else 0.0,
            "fallback_contexts": int((context_groups["window_psms"] == 0).sum()),
            "contexts_preserved": int((context_groups["selected_psms"] > 0).sum()),
        },
        "spectral_consistency": {
            "metric": "mean leave-one-out spectral angle to acquisition-context consensus",
            "all": _mean_leave_one_out_sa(frame),
            "within_apex_window": _mean_leave_one_out_sa(frame, "_within_apex_window"),
            "selected": _mean_leave_one_out_sa(frame, "_selected"),
        },
    }
    annotations = frame.select(
        pl.col("spectrum_id").cast(pl.UInt64),
        pl.col("precursor_intensity").cast(pl.Float64),
        pl.col("_intensity_ratio").cast(pl.Float64).alias("intensity_ratio"),
        pl.col("_supports_half_max").alias("supports_half_max"),
        pl.col("_apex_window_start").cast(pl.Float64).alias("apex_window_start"),
        pl.col("_apex_window_end").cast(pl.Float64).alias("apex_window_end"),
        pl.col("_within_apex_window").alias("within_apex_window"),
        pl.col("_context_psms").alias("context_psms"),
        pl.col("_context_window_psms").alias("context_window_psms"),
        pl.col("_selection_rank").alias("selection_rank"),
        pl.col("_selected").alias("selected"),
        pl.col("_selection_reason").alias("selection_reason"),
    ).select(
        [
            pl.col(name).cast(dtype, strict=True)
            for name, dtype in CURATION_ANNOTATION_SCHEMA.items()
        ]
    )
    if annotations.schema != CURATION_ANNOTATION_SCHEMA:
        raise RuntimeError("curation annotation schema construction drifted")
    return CurationAnalysis(report, annotations)


__all__ = [
    "CURATION_ANNOTATION_SCHEMA",
    "CurationAnalysis",
    "analyze_prepared_curation",
]
