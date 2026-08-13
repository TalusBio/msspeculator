import json

import polars as pl
import pytest

from pepdistill.data.prepared_schema import prepared_frame
from pepdistill.etl.curation import CURATION_ANNOTATION_SCHEMA, analyze_prepared_curation


def _row(
    spectrum_id: int,
    scan: int,
    sequence: str,
    score: float,
    ms2: list[float],
    *,
    raw_rt: float | None = None,
    detector: str = "FTMS",
) -> dict:
    return {
        "spectrum_id": spectrum_id,
        "dataset": "ptm",
        "raw_file": "run",
        "scan_number": scan,
        "sequence": sequence,
        "mods": "1:Phospho",
        "charge": 2,
        "split": "train",
        "irt": 10.0,
        "raw_rt": float(scan) if raw_rt is None else raw_rt,
        "instrument": "Lumos",
        "detector": detector,
        "fragmentation": "HCD",
        "energy": 30.0,
        "andromeda_score": score,
        "ms2": ms2,
    }


def test_curation_uses_half_max_cap_and_missing_intensity_fallback(tmp_path):
    prepared = prepared_frame(
        [
            _row(1, 1, "PEPTIDE", 10.0, [1.0, 0.0]),
            _row(2, 2, "PEPTIDE", 20.0, [0.9, 0.1]),
            _row(3, 3, "PEPTIDE", 30.0, [0.0, 1.0]),
            # A different mode with low own intensity but RT inside the shared peptide window.
            _row(4, 4, "PEPTIDE", 15.0, [0.8, 0.2], raw_rt=1.5, detector="ITMS"),
            _row(5, 5, "OTHER", 5.0, [1.0, 0.0]),
            _row(6, 6, "OTHER", 15.0, [0.8, 0.2]),
        ]
    )
    metadata = pl.DataFrame(
        {
            "raw_file": ["run"] * 6,
            "scan_number": [1, 2, 3, 4, 5, 6],
            "precursor_intensity": [10.0, 6.0, 2.0, 1.0, None, None],
        },
        schema={
            "raw_file": pl.String,
            "scan_number": pl.Int64,
            "precursor_intensity": pl.Float64,
        },
    )
    prepared_path = tmp_path / "prepared.parquet"
    metadata_path = tmp_path / "metadata.parquet"
    prepared.write_parquet(prepared_path)
    metadata.write_parquet(metadata_path)

    analysis = analyze_prepared_curation(prepared_path, metadata_path, cap_per_context=1)
    assert analysis.annotations.schema == CURATION_ANNOTATION_SCHEMA
    selected = analysis.annotations.filter(pl.col("selected")).sort("spectrum_id")
    # Of the two half-max PEPTIDE rows, higher score wins. OTHER has no intensity and falls back.
    assert selected["spectrum_id"].to_list() == [2, 4, 6]
    assert selected["selection_reason"].to_list() == [
        "apex_window_top_score",
        "apex_window_top_score",
        "context_best_score_fallback",
    ]
    assert analysis.annotations.filter(pl.col("spectrum_id") == 4)["supports_half_max"][0] is False
    assert analysis.annotations.filter(pl.col("spectrum_id") == 4)["within_apex_window"][0] is True
    assert analysis.report["selection"]["half_max_support_rows"] == 2
    assert analysis.report["selection"]["apex_window_rows"] == 3
    assert analysis.report["selection"]["selected_rows"] == 3
    assert analysis.report["selection"]["fallback_contexts"] == 1
    assert analysis.report["selection"]["contexts_preserved"] == 3
    assert analysis.report["spectral_consistency"]["all"] is not None
    assert analysis.report["replication"]["selected_psms_per_peptidoform"]["p50"] == 1.5
    assert analysis.report["chromatography"]["window_width_minutes"]["p50"] == 1.0
    assert analysis.report["chromatography"]["missing_windows"] == 1

    report_path = tmp_path / "report.json"
    annotations_path = tmp_path / "annotations.parquet"
    analysis.write(report_path, annotations_path)
    assert json.loads(report_path.read_text())["policy"]["cap_per_context"] == 1
    assert pl.read_parquet(annotations_path).schema == CURATION_ANNOTATION_SCHEMA


@pytest.mark.parametrize(
    ("half_max_fraction", "cap"),
    [(0.0, 8), (1.1, 8), (0.5, 0)],
)
def test_curation_rejects_invalid_policy(tmp_path, half_max_fraction, cap):
    with pytest.raises(ValueError):
        analyze_prepared_curation(
            tmp_path / "unused-prepared.parquet",
            tmp_path / "unused-metadata.parquet",
            half_max_fraction=half_max_fraction,
            cap_per_context=cap,
        )
