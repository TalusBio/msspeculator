"""Run-shared chromatographic curation and context-level PSM deduplication."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..data.prepared_schema import PREPARED_SPECTRA_SCHEMA, read_prepared_parquet
from ..diagnostics import SA_HISTOGRAM_EDGES, sa_histogram
from .config import PrepareCuration

_NO_VALUES = np.empty(0, dtype=np.float64)


CURATION_INPUT_SCHEMA = pl.Schema(
    {**dict(PREPARED_SPECTRA_SCHEMA.items()), "precursor_intensity": pl.Float64}
)
CURATION_ANNOTATION_SCHEMA = pl.Schema(
    {
        "spectrum_id": pl.UInt64,
        "precursor_intensity": pl.Float64,
        "intensity_ratio": pl.Float64,
        "supports_half_max": pl.Boolean,
        "apex_rt": pl.Float64,
        "run_width_minutes": pl.Float64,
        "apex_window_start": pl.Float64,
        "apex_window_end": pl.Float64,
        "within_apex_window": pl.Boolean,
        "peptidoform_window_psms": pl.Int64,
        "context_window_psms": pl.Int64,
        "selection_rank": pl.Int64,
        "selected": pl.Boolean,
        "selection_reason": pl.String,
    }
)

_SPECTRUM_KEY = ["raw_file", "scan_number"]
_PEPTIDOFORM_KEY = ["raw_file", "sequence", "mods"]
_CONTEXT_KEY = _PEPTIDOFORM_KEY + [
    "charge",
    "detector",
    "fragmentation",
    "_energy_bucket",
]


@dataclass(frozen=True)
class CurationAnalysis:
    report: dict[str, Any]
    annotations: pl.DataFrame
    selected: pl.DataFrame

    def write(self, report_path: str | Path, annotations_path: str | Path | None = None) -> None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.report, indent=2, sort_keys=True) + "\n")
        if annotations_path is not None:
            annotations_path = Path(annotations_path)
            annotations_path.parent.mkdir(parents=True, exist_ok=True)
            self.annotations.write_parquet(annotations_path, compression="zstd")


def curation_input_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build the temporary intensity-bearing ETL table without schema inference."""
    return pl.DataFrame(rows, schema=CURATION_INPUT_SCHEMA, strict=True)


def _quantiles(series: pl.Series) -> dict[str, float | None]:
    if series.is_empty():
        return {f"p{int(q * 100):02d}": None for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)}
    return {
        f"p{int(q * 100):02d}": (
            float(value)
            if (value := series.quantile(q, interpolation="linear")) is not None
            else None
        )
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def _leave_one_out_sa(group: pl.DataFrame) -> np.ndarray:
    """Each replicate's SA against the consensus of the *other* replicates in its context.

    This is the achievable-accuracy ceiling: a model can at best predict the consensus of the
    replicates, so however well one replicate agrees with its peers bounds how well any model
    could score against it. The consensus is built only from the rows handed in, so a filtered
    subset is scored against itself -- measuring a subset against a consensus that still contained
    the rejected spectra scored *identical* spectra at 0.21.
    """
    if group.height < 2:
        return _NO_VALUES
    spectra = np.asarray(group["ms2"].to_list(), dtype=np.float64)
    norms = np.linalg.norm(spectra, axis=1)
    valid = norms > 0
    if int(valid.sum()) < 2:
        return _NO_VALUES
    unit = spectra[valid] / norms[valid, None]
    peers = unit.sum(axis=0)[None, :] - unit
    peer_norms = np.linalg.norm(peers, axis=1)
    measurable = peer_norms > 0
    cosine = np.sum(unit[measurable] * peers[measurable], axis=1) / peer_norms[measurable]
    return 1.0 - 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)) / pi


def _achievable_ceilings(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Ceilings for every row, the in-window rows, and the retained rows.

    Partitions once and evaluates all three subsets per context: this runs for every shard in
    production, so three separate passes over the same partitioning would triple the cost of a
    diagnostic that is always on.
    """
    collected: dict[str, list[np.ndarray]] = {"all": [], "within_apex_window": [], "selected": []}
    for group in frame.partition_by(_CONTEXT_KEY, maintain_order=False):
        for name, predicate in (
            ("all", None),
            ("within_apex_window", "_within_apex_window"),
            ("selected", "_selected"),
        ):
            subset = (
                group if predicate is None else group.filter(pl.col(predicate).fill_null(False))
            )
            values = _leave_one_out_sa(subset)
            if values.size:
                collected[name].append(values)
    out: dict[str, dict[str, Any]] = {}
    for name, chunks in collected.items():
        values = np.concatenate(chunks) if chunks else _NO_VALUES
        out[name] = {
            "mean": float(values.mean()) if values.size else None,
            "replicates_compared": int(values.size),
            "histogram": sa_histogram(values) if values.size else None,
        }
    return out


def curate_prepared_frame(
    frame: pl.DataFrame, policy: PrepareCuration = PrepareCuration()
) -> CurationAnalysis:
    """Apply one shared-width window per raw-file peptidoform, then keep top PSMs per context.

    The policy arrives as a :class:`PrepareCuration` rather than as loose keywords so its defaults
    and its bounds are defined exactly once. Restating them in this signature meant a new knob had
    to be added in four places, and nothing made the two copies agree.

    Half-height observations establish robust run-level width anchors. Every peptidoform in a
    raw file receives that run's median anchor width, centered on its own intensity apex. This
    avoids the severe observation-count bias of using each peptide's sampled min/max width.

    That single width is used *directly* as the acceptance window: a PSM counts when its RT lies
    within ``apex_rt +/- run_width_minutes / 2``. Because the width is estimated from sampled
    half-height points rather than an integrated peak, it is clamped to
    ``[min_run_width_minutes, max_run_width_minutes]`` -- the range a real chromatographic peak
    can occupy -- and each run records whether that clamp applied.

    Peptidoforms below ``min_in_window_psms`` are rejected; qualifying rows are ranked within a
    charge/acquisition context by score, apex proximity, precursor intensity, then stable ID.
    """
    if frame.schema != CURATION_INPUT_SCHEMA:
        raise ValueError(
            f"curation input schema differs: {frame.schema} != {CURATION_INPUT_SCHEMA}"
        )
    # Bounds are enforced by PrepareCuration.__post_init__, so an instance is already valid.
    half_max_fraction = policy.half_max_fraction
    min_in_window_psms = policy.min_in_window_psms
    max_psms_per_context = policy.max_psms_per_context
    width_anchor_min_psms = policy.width_anchor_min_psms
    energy_bucket_width = policy.energy_bucket_width
    min_run_width_minutes = policy.min_run_width_minutes
    max_run_width_minutes = policy.max_run_width_minutes

    frame = frame.with_columns(
        pl.when(pl.col("energy").is_finite())
        .then((pl.col("energy") / energy_bucket_width).round().cast(pl.Int64))
        .otherwise(pl.lit(None, dtype=pl.Int64))
        .alias("_energy_bucket"),
        pl.col("precursor_intensity").max().over(_PEPTIDOFORM_KEY).alias("_apex_intensity"),
    ).with_columns(
        (pl.col("precursor_intensity") / pl.col("_apex_intensity")).alias("_intensity_ratio")
    )
    frame = frame.with_columns(
        (pl.col("_intensity_ratio").is_finite() & (pl.col("_intensity_ratio") >= half_max_fraction))
        .fill_null(False)
        .alias("_supports_half_max")
    )
    peptidoforms = (
        frame.group_by(_PEPTIDOFORM_KEY)
        .agg(
            pl.len().cast(pl.Int64).alias("psms"),
            pl.col("_supports_half_max").sum().cast(pl.Int64).alias("half_max_support_psms"),
            pl.col("raw_rt")
            .filter(pl.col("_supports_half_max"))
            .sort_by(
                pl.col("precursor_intensity").filter(pl.col("_supports_half_max")), descending=True
            )
            .first()
            .alias("apex_rt"),
            pl.col("raw_rt").filter(pl.col("_supports_half_max")).min().alias("observed_start"),
            pl.col("raw_rt").filter(pl.col("_supports_half_max")).max().alias("observed_end"),
        )
        .with_columns(
            (pl.col("observed_end") - pl.col("observed_start")).alias("observed_width_minutes")
        )
    )
    positive_widths = peptidoforms.filter(pl.col("observed_width_minutes") > 0)
    anchors = positive_widths.filter(pl.col("half_max_support_psms") >= width_anchor_min_psms)
    run_widths = anchors.group_by("raw_file").agg(
        pl.col("observed_width_minutes").median().alias("run_width_minutes"),
        pl.len().cast(pl.Int64).alias("width_anchor_peptidoforms"),
    )
    # Small/custom runs can lack eight-repeat anchors. Fall back to the run median of every
    # measurable positive width, but never invent a width when the source has no evidence.
    fallbacks = positive_widths.group_by("raw_file").agg(
        pl.col("observed_width_minutes").median().alias("fallback_width_minutes")
    )
    raw_files = frame.select("raw_file").unique()
    run_widths = (
        raw_files.join(run_widths, on="raw_file", how="left")
        .join(fallbacks, on="raw_file", how="left")
        .with_columns(
            pl.coalesce("run_width_minutes", "fallback_width_minutes").alias("run_width_minutes"),
            pl.col("width_anchor_peptidoforms").fill_null(0),
        )
        .drop("fallback_width_minutes")
        .with_columns(
            # The estimate is the RT span between *sampled* half-height points, so it can be
            # implausible in either direction: a fraction of a second when a run has no anchors,
            # or minutes when a peptidoform elutes more than once. Clamp into the range a real
            # peak can occupy and record which runs were clamped -- a clamped width is a stated
            # prior about chromatography, not a measurement from this run. A run whose width is
            # not estimable at all stays null and selects nothing; there is no invented width.
            (
                (pl.col("run_width_minutes") < min_run_width_minutes)
                | (pl.col("run_width_minutes") > max_run_width_minutes)
            )
            .fill_null(False)
            .alias("width_clamped"),
            pl.col("run_width_minutes")
            .clip(min_run_width_minutes, max_run_width_minutes)
            .alias("run_width_minutes"),
        )
    )
    peptidoforms = peptidoforms.join(
        run_widths, on="raw_file", how="left", validate="m:1"
    ).with_columns(
        (pl.col("apex_rt") - pl.col("run_width_minutes") / 2).alias("apex_window_start"),
        (pl.col("apex_rt") + pl.col("run_width_minutes") / 2).alias("apex_window_end"),
    )
    frame = frame.join(
        peptidoforms.select(
            _PEPTIDOFORM_KEY
            + ["apex_rt", "run_width_minutes", "apex_window_start", "apex_window_end"]
        ),
        on=_PEPTIDOFORM_KEY,
        how="left",
        validate="m:1",
    ).with_columns(
        (
            pl.col("raw_rt").is_between(
                pl.col("apex_window_start"), pl.col("apex_window_end"), closed="both"
            )
            & pl.col("apex_rt").is_finite()
        )
        .fill_null(False)
        .alias("_within_apex_window"),
        (pl.col("raw_rt") - pl.col("apex_rt")).abs().alias("_apex_distance"),
    )
    frame = frame.with_columns(
        pl.col("_within_apex_window")
        .sum()
        .over(_PEPTIDOFORM_KEY)
        .cast(pl.Int64)
        .alias("_peptidoform_window_psms"),
        pl.col("_within_apex_window")
        .sum()
        .over(_CONTEXT_KEY)
        .cast(pl.Int64)
        .alias("_context_window_psms"),
    ).with_columns(
        (
            pl.col("_within_apex_window")
            & (pl.col("_peptidoform_window_psms") >= min_in_window_psms)
        ).alias("_eligible")
    )
    # Polars orders NaN above every finite value, so an unscored or intensity-less PSM would
    # otherwise outrank real evidence. Demote non-finite rank keys to null and sort nulls last.
    rank_keys = [pl.col(name) for name in _CONTEXT_KEY] + [
        pl.col("_eligible"),
        pl.col("andromeda_score").fill_nan(None),
        pl.col("_apex_distance").fill_nan(None),
        pl.col("precursor_intensity").fill_nan(None),
        pl.col("spectrum_id"),
    ]
    frame = frame.sort(
        rank_keys,
        descending=[False] * len(_CONTEXT_KEY) + [True, True, False, True, False],
        nulls_last=True,
    ).with_columns(
        pl.col("_eligible").cum_sum().over(_CONTEXT_KEY).cast(pl.Int64).alias("_selection_rank")
    )
    frame = frame.with_columns(
        (pl.col("_eligible") & (pl.col("_selection_rank") <= max_psms_per_context)).alias(
            "_selected"
        )
    ).with_columns(
        pl.when(pl.col("_selected"))
        .then(pl.lit("top_context_psm"))
        .when(pl.col("_peptidoform_window_psms") < min_in_window_psms)
        .then(pl.lit("insufficient_in_window_replication"))
        .when(~pl.col("_within_apex_window"))
        .then(pl.lit("outside_shared_window"))
        .otherwise(pl.lit("context_cap"))
        .alias("_selection_reason")
    )

    peptidoform_groups = frame.group_by(_PEPTIDOFORM_KEY).agg(
        pl.len().cast(pl.Int64).alias("psms"),
        pl.col("_supports_half_max").sum().cast(pl.Int64).alias("half_max_support_psms"),
        pl.col("_within_apex_window").sum().cast(pl.Int64).alias("window_psms"),
        pl.col("_selected").sum().cast(pl.Int64).alias("selected_psms"),
    )
    context_groups = frame.group_by(_CONTEXT_KEY).agg(
        pl.col("_within_apex_window").sum().cast(pl.Int64).alias("window_psms"),
        pl.col("_selected").sum().cast(pl.Int64).alias("selected_psms"),
    )
    selected_rows = int(frame["_selected"].sum())
    window_rows = int(frame["_within_apex_window"].sum())
    qualifying = int((peptidoform_groups["window_psms"] >= min_in_window_psms).sum())
    report: dict[str, Any] = {
        "policy": {
            "apex_scope": "raw_file+peptidoform (shared across charge and acquisition mode)",
            "width_scope": (
                "robust median per raw file, clamped to "
                f"[{min_run_width_minutes}, {max_run_width_minutes}] minutes and used directly "
                "as the acceptance window (apex +/- width/2)"
            ),
            "context_scope": "peptidoform+charge+detector+fragmentation+energy_bucket",
            "half_max_fraction": half_max_fraction,
            "min_in_window_psms": min_in_window_psms,
            "max_psms_per_context": max_psms_per_context,
            "width_anchor_min_psms": width_anchor_min_psms,
            "energy_bucket_width": energy_bucket_width,
            "min_run_width_minutes": min_run_width_minutes,
            "max_run_width_minutes": max_run_width_minutes,
        },
        "input": {
            "rows": frame.height,
            "peptidoform_groups": peptidoform_groups.height,
            "context_groups": context_groups.height,
            # Sources without the column arrive as NaN (see meta_index), joins as null; both
            # are unusable evidence and must be reported rather than silently ranked.
            "missing_precursor_intensity_rows": int(
                (~frame["precursor_intensity"].is_finite().fill_null(False)).sum()
            ),
        },
        "chromatography": {
            "run_widths": {
                row["raw_file"]: {
                    "width_minutes": row["run_width_minutes"],
                    "anchor_peptidoforms": row["width_anchor_peptidoforms"],
                    "width_clamped": row["width_clamped"],
                }
                for row in run_widths.to_dicts()
            },
            "clamped_runs": int(run_widths["width_clamped"].sum()),
            "runs": run_widths.height,
            "observed_anchor_width_minutes": _quantiles(anchors["observed_width_minutes"]),
        },
        "replication": {
            "psms_per_peptidoform": _quantiles(peptidoform_groups["psms"]),
            "window_psms_per_peptidoform": _quantiles(peptidoform_groups["window_psms"]),
            "selected_psms_per_peptidoform": _quantiles(peptidoform_groups["selected_psms"]),
        },
        "selection": {
            "apex_window_rows": window_rows,
            "qualifying_peptidoforms": qualifying,
            "rejected_peptidoforms": peptidoform_groups.height - qualifying,
            "selected_rows": selected_rows,
            "selected_fraction_of_rows": selected_rows / frame.height if frame.height else 0.0,
            "selected_contexts": int((context_groups["selected_psms"] > 0).sum()),
        },
    }
    report["achievable_ceiling"] = {
        "metric": (
            "leave-one-out spectral angle of each replicate against the consensus of the other "
            "replicates in its acquisition context; an upper bound on what any model can score"
        ),
        "histogram_bin_edges": list(SA_HISTOGRAM_EDGES),
        **_achievable_ceilings(frame),
    }

    annotations = frame.select(
        pl.col("spectrum_id").cast(pl.UInt64),
        pl.col("precursor_intensity").cast(pl.Float64),
        pl.col("_intensity_ratio").cast(pl.Float64).alias("intensity_ratio"),
        pl.col("_supports_half_max").alias("supports_half_max"),
        pl.col("apex_rt").cast(pl.Float64),
        pl.col("run_width_minutes").cast(pl.Float64),
        pl.col("apex_window_start").cast(pl.Float64),
        pl.col("apex_window_end").cast(pl.Float64),
        pl.col("_within_apex_window").alias("within_apex_window"),
        pl.col("_peptidoform_window_psms").alias("peptidoform_window_psms"),
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
    selected = frame.filter(pl.col("_selected")).select(
        [pl.col(name).cast(dtype, strict=True) for name, dtype in PREPARED_SPECTRA_SCHEMA.items()]
    )
    return CurationAnalysis(report, annotations, selected)


def analyze_prepared_curation(
    prepared_path: str | Path,
    metadata_path: str | Path,
    **policy: Any,
) -> CurationAnalysis:
    """Join one prepared shard to source intensity metadata and evaluate the production policy.

    Keeps loose keywords because it is the interactive entry point, where naming one knob and
    taking defaults for the rest is the common case. Building the policy here also rejects an
    invalid one before any shard is read.
    """
    resolved = PrepareCuration(**policy)
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
        .unique(_SPECTRUM_KEY, keep="first", maintain_order=True)
        .collect(engine="streaming")
    )
    frame = prepared.join(metadata, on=_SPECTRUM_KEY, how="left", validate="1:1").select(
        [pl.col(name).cast(dtype, strict=True) for name, dtype in CURATION_INPUT_SCHEMA.items()]
    )
    return curate_prepared_frame(frame, resolved)


__all__ = [
    "CURATION_ANNOTATION_SCHEMA",
    "CURATION_INPUT_SCHEMA",
    "CurationAnalysis",
    "analyze_prepared_curation",
    "curate_prepared_frame",
    "curation_input_frame",
]
