import json

import polars as pl
import pytest

from pepdistill.data.prepared_schema import PREPARED_SPECTRA_SCHEMA, prepared_frame
from pepdistill.etl.curation import CURATION_ANNOTATION_SCHEMA, analyze_prepared_curation


def _row(
    spectrum_id: int,
    scan: int,
    proforma: str,
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
        "proforma": proforma,
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


def test_curation_uses_shared_window_replication_filter_and_context_cap(tmp_path):
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

    analysis = analyze_prepared_curation(
        prepared_path,
        metadata_path,
        min_in_window_psms=2,
        max_psms_per_context=1,
        width_anchor_min_psms=2,
        # These fixtures space replicates minutes apart to keep the arithmetic readable, so the
        # plausible-width clamp is widened out of the way; clamping has its own test above.
        max_run_width_minutes=60.0,
    )
    assert analysis.annotations.schema == CURATION_ANNOTATION_SCHEMA
    assert analysis.selected.schema == PREPARED_SPECTRA_SCHEMA
    selected = analysis.annotations.filter(pl.col("selected")).sort("spectrum_id")
    # One shared PEPTIDE window includes both acquisition modes; the intensity-less OTHER
    # peptidoform is rejected rather than receiving a fabricated chromatographic window.
    assert selected["spectrum_id"].to_list() == [1, 4]
    assert selected["selection_reason"].to_list() == ["top_context_psm", "top_context_psm"]
    assert analysis.annotations.filter(pl.col("spectrum_id") == 4)["supports_half_max"][0] is False
    assert analysis.annotations.filter(pl.col("spectrum_id") == 4)["within_apex_window"][0] is True
    rejected = analysis.annotations.filter(pl.col("spectrum_id") == 6)
    assert rejected["selection_reason"][0] == "insufficient_in_window_replication"
    assert analysis.report["selection"]["apex_window_rows"] == 2
    assert analysis.report["selection"]["selected_rows"] == 2
    assert analysis.report["selection"]["qualifying_peptidoforms"] == 1
    assert analysis.report["selection"]["rejected_peptidoforms"] == 1
    assert analysis.report["achievable_ceiling"]["all"]["mean"] is not None
    assert analysis.report["replication"]["selected_psms_per_peptidoform"]["p50"] == 1.0
    assert analysis.report["chromatography"]["run_widths"]["run"]["width_minutes"] == 1.0

    report_path = tmp_path / "report.json"
    annotations_path = tmp_path / "annotations.parquet"
    analysis.write(report_path, annotations_path)
    assert json.loads(report_path.read_text())["policy"]["max_psms_per_context"] == 1
    assert pl.read_parquet(annotations_path).schema == CURATION_ANNOTATION_SCHEMA


def test_achievable_ceiling_scores_each_subset_against_its_own_consensus(tmp_path):
    """The reported ceiling must describe the retained subset, not the subset versus everything.

    Building the consensus from every spectrum in the context and only masking which rows get
    measured scores a policy against the observations it discarded, so the three numbers cannot
    be compared to each other -- which is what made `selected` land below `within_apex_window`.
    Here the two selected spectra agree perfectly with each other and disagree with the rejected
    ones, so a subset-local consensus must report 1.0.
    """
    prepared = prepared_frame(
        [
            _row(1, 1, "PEPTIDE", 10.0, [0.0, 1.0], raw_rt=1.0),
            _row(2, 2, "PEPTIDE", 90.0, [1.0, 0.0], raw_rt=2.0),
            _row(3, 3, "PEPTIDE", 100.0, [1.0, 0.0], raw_rt=3.0),
            _row(4, 4, "PEPTIDE", 20.0, [0.0, 1.0], raw_rt=4.0),
            _row(5, 5, "PEPTIDE", 30.0, [0.0, 1.0], raw_rt=5.0),
        ]
    )
    metadata = pl.DataFrame(
        {
            "raw_file": ["run"] * 5,
            "scan_number": [1, 2, 3, 4, 5],
            "precursor_intensity": [6.0, 9.0, 10.0, 9.0, 6.0],
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

    analysis = analyze_prepared_curation(
        prepared_path,
        metadata_path,
        min_in_window_psms=2,
        max_psms_per_context=2,
        width_anchor_min_psms=2,
        # These fixtures space replicates minutes apart to keep the arithmetic readable, so the
        # plausible-width clamp is widened out of the way; clamping has its own test above.
        max_run_width_minutes=60.0,
    )
    assert analysis.report["selection"]["selected_rows"] == 2
    ceiling = analysis.report["achievable_ceiling"]
    assert ceiling["selected"]["mean"] == pytest.approx(1.0)
    assert ceiling["selected"]["mean"] > ceiling["within_apex_window"]["mean"]
    # The distribution travels with the mean so the ceiling can be drawn, not just annotated,
    # and it shares the grid the teacher yardstick and student validation use.
    histogram = ceiling["selected"]["histogram"]
    assert histogram["counted"] == histogram["total"] == ceiling["selected"]["replicates_compared"]
    assert len(histogram["counts"]) == len(ceiling["histogram_bin_edges"]) - 1
    assert histogram["counts"][-1] == histogram["total"]  # all mass in the top bin at SA 1.0


def test_curation_ranks_unscored_psms_last(tmp_path):
    """A NaN Andromeda score must never outrank a real one.

    Polars orders NaN above every finite value, so a descending sort on a raw score column
    hands the context slot to an unscored PSM. The observed corpus does contain rows without a
    usable score, so this is a silent quality regression, not a hypothetical one.
    """
    prepared = prepared_frame(
        [
            _row(1, 1, "PEPTIDE", float("nan"), [1.0, 0.0], raw_rt=1.0),
            _row(2, 2, "PEPTIDE", 10.0, [0.9, 0.1], raw_rt=2.0),
            _row(3, 3, "PEPTIDE", 5.0, [0.8, 0.2], raw_rt=3.0),
        ]
    )
    metadata = pl.DataFrame(
        {
            "raw_file": ["run"] * 3,
            "scan_number": [1, 2, 3],
            "precursor_intensity": [10.0, 8.0, 6.0],
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

    analysis = analyze_prepared_curation(
        prepared_path,
        metadata_path,
        min_in_window_psms=2,
        max_psms_per_context=1,
        width_anchor_min_psms=2,
        # These fixtures space replicates minutes apart to keep the arithmetic readable, so the
        # plausible-width clamp is widened out of the way; clamping has its own test above.
        max_run_width_minutes=60.0,
    )
    # Apex is scan 1 (highest intensity); the shared window spans scans 1-2, so both compete
    # for the single context slot. Scan 2 must win on score despite sitting further from apex.
    selected = analysis.annotations.filter(pl.col("selected"))
    assert selected["spectrum_id"].to_list() == [2]


def test_curation_clamps_an_implausible_run_width(tmp_path):
    """An implausible per-run width must be clamped into chromatographic range, and flagged.

    Reproduces the real failure found across the full corpus: with no anchors the fallback median
    can be a fraction of a second, which admits nothing and rejects a shard full of usable PSMs.
    The width doubles as the acceptance window (apex +/- width/2), so clamping it is what makes
    the window plausible; each run records whether the clamp applied.
    """
    prepared = prepared_frame(
        [
            _row(1, 1, "PEPTIDE", 10.0, [1.0, 0.0], raw_rt=10.000),
            _row(2, 2, "PEPTIDE", 20.0, [0.9, 0.1], raw_rt=10.002),
            _row(3, 3, "OTHERPEP", 10.0, [1.0, 0.0], raw_rt=20.000),
            _row(4, 4, "OTHERPEP", 20.0, [0.9, 0.1], raw_rt=20.002),
        ]
    )
    metadata = pl.DataFrame(
        {
            "raw_file": ["run"] * 4,
            "scan_number": [1, 2, 3, 4],
            "precursor_intensity": [10.0, 9.0, 10.0, 9.0],
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

    analysis = analyze_prepared_curation(
        prepared_path,
        metadata_path,
        min_in_window_psms=2,
        max_psms_per_context=2,
        width_anchor_min_psms=8,  # no peptidoform reaches this, so the fallback decides
    )
    run = analysis.report["chromatography"]["run_widths"]["run"]
    assert run["anchor_peptidoforms"] == 0
    assert run["width_clamped"] is True
    # 0.002 min is below the 3 s floor, so it is raised to it.
    assert run["width_minutes"] == pytest.approx(0.05)
    assert analysis.report["chromatography"]["clamped_runs"] == 1
    # Both replicates of both peptidoforms now sit inside the clamped window and are retained.
    assert analysis.report["selection"]["qualifying_peptidoforms"] == 2
    assert analysis.report["selection"]["selected_rows"] == 4

    # Widening the plausible range disables the clamp and reproduces the collapse the corpus run
    # exhibited, so the recovery is attributable to the clamp and nothing else in the policy.
    unclamped = analyze_prepared_curation(
        prepared_path,
        metadata_path,
        min_in_window_psms=2,
        max_psms_per_context=2,
        width_anchor_min_psms=8,
        min_run_width_minutes=0.0,
    )
    assert unclamped.report["chromatography"]["run_widths"]["run"]["width_minutes"] == (
        pytest.approx(0.002)
    )
    assert unclamped.report["selection"]["selected_rows"] == 0

    # An over-wide estimate is pulled down to the ceiling, and the ceiling cannot sit below the
    # floor -- an inverted range is a configuration error, not something to silently reorder.
    wide = analyze_prepared_curation(
        prepared_path,
        metadata_path,
        min_in_window_psms=2,
        max_psms_per_context=2,
        width_anchor_min_psms=8,
        min_run_width_minutes=0.0,
        max_run_width_minutes=0.001,
    )
    assert wide.report["chromatography"]["run_widths"]["run"]["width_minutes"] == (
        pytest.approx(0.001)
    )
    assert wide.report["chromatography"]["clamped_runs"] == 1
    with pytest.raises(ValueError, match="max_run_width_minutes"):
        analyze_prepared_curation(
            prepared_path, metadata_path, min_run_width_minutes=0.5, max_run_width_minutes=0.1
        )


@pytest.mark.parametrize(
    ("half_max_fraction", "minimum", "maximum", "anchor"),
    [(0.0, 4, 2, 8), (1.1, 4, 2, 8), (0.5, 0, 2, 8), (0.5, 4, 0, 8), (0.5, 4, 2, 1)],
)
def test_curation_rejects_invalid_policy(tmp_path, half_max_fraction, minimum, maximum, anchor):
    with pytest.raises(ValueError):
        analyze_prepared_curation(
            tmp_path / "unused-prepared.parquet",
            tmp_path / "unused-metadata.parquet",
            half_max_fraction=half_max_fraction,
            min_in_window_psms=minimum,
            max_psms_per_context=maximum,
            width_anchor_min_psms=anchor,
        )
