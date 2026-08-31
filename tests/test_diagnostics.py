import numpy as np
import pytest

from msspeculator.diagnostics import (
    EmbeddingConnection,
    IRT_STANDARDS,
    LabeledEmbedding,
    PcaBasis,
    RtObservation,
    SpectrumComparison,
    normalized_spectral_angle,
    irt_regression_metrics,
    plot_irt_scatter,
    plot_labeled_embedding_pca,
    plot_spectrum_butterflies,
)


def test_fixed_pca_basis_roundtrip_shape_and_variance():
    values = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 1.0], [6.0, 1.0]])
    basis = PcaBasis.fit(values)
    coordinates = basis.transform(values)
    assert coordinates.shape == (4, 2)
    assert basis.explained_variance_ratio.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(coordinates.mean(axis=0), 0.0, atol=1e-12)


def test_spectrum_plot_writes_png(tmp_path):
    butterfly_path = plot_spectrum_butterflies(
        [
            SpectrumComparison(
                proforma_sequence="PEPTIDEK",
                charge=2,
                fragment_mz=np.asarray([[100.0, 200.0]]),
                student_intensity=np.asarray([[1.0, 0.2]]),
                reference_intensity=np.asarray([[0.8, 0.3]]),
                reference_name="experiment",
            )
        ],
        tmp_path / "butterfly.png",
    )
    assert butterfly_path.stat().st_size > 0


def test_labeled_embedding_and_irt_plots_write_pngs(tmp_path):
    embedding_path, _ = plot_labeled_embedding_pca(
        [
            LabeledEmbedding("A", "residue", np.asarray([1.0, 0.0, 0.2])),
            LabeledEmbedding("B", "residue", np.asarray([0.0, 1.0, 0.1])),
            LabeledEmbedding("B:mass", "mass", np.asarray([0.1, 0.9, 0.1])),
        ],
        tmp_path / "labeled.png",
        title="tokens",
        connections=[EmbeddingConnection("B", "B:mass")],
    )
    irt_path = plot_irt_scatter(
        [
            RtObservation("PEPTIDEK", 0.0, 1.0),
            RtObservation("SAMPLER", 10.0, 9.0),
        ],
        tmp_path / "irt.png",
    )
    assert embedding_path.stat().st_size > 0
    assert irt_path.stat().st_size > 0


def test_canonical_irt_panel_is_ordered_and_complete():
    assert len(IRT_STANDARDS) == 11
    assert IRT_STANDARDS[0].sequence == "LGGNEQVTR"
    assert IRT_STANDARDS[0].irt == pytest.approx(-24.916114)
    assert IRT_STANDARDS[-1].irt == pytest.approx(100.00282166666665)


def test_normalized_spectral_angle_matches_identical_and_orthogonal():
    assert normalized_spectral_angle(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])) == 1.0
    assert normalized_spectral_angle(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == 0.0


def test_irt_regression_metrics_are_reusable_without_plotting():
    metrics = irt_regression_metrics([RtObservation("A", 0.0, 1.0), RtObservation("B", 10.0, 9.0)])
    assert metrics.slope == pytest.approx(0.8)
    assert metrics.intercept == pytest.approx(1.0)
    assert metrics.r_squared == pytest.approx(1.0)
    assert metrics.mae == pytest.approx(1.0)


def test_spectral_angle_series_recovers_its_mean_from_counts():
    from msspeculator.diagnostics import SA_HISTOGRAM_BINS, SpectralAngleSeries, sa_histogram

    values = np.concatenate([np.full(300, 0.92), np.full(100, 0.41)])
    series = SpectralAngleSeries("student", sa_histogram(values)["counts"])
    assert series.total() == 400
    # Binned, so the recovered mean is accurate to within one bin width rather than exactly.
    assert series.mean() == pytest.approx(float(values.mean()), abs=1.0 / SA_HISTOGRAM_BINS)
    assert SpectralAngleSeries("empty", [0] * SA_HISTOGRAM_BINS).mean() is None

    # Interpolated within the containing bin, so also good to within one bin width.
    assert series.median() == pytest.approx(float(np.median(values)), abs=1.0 / SA_HISTOGRAM_BINS)
    assert SpectralAngleSeries("empty", [0] * SA_HISTOGRAM_BINS).median() is None

    # A skewed distribution is where the two statistics have to disagree; recovering both from one
    # histogram is the only reason to show them together.
    skewed = np.clip(1.0 - np.abs(np.random.default_rng(1).normal(0.0, 0.15, 20_000)), 0.0, 1.0)
    piled = SpectralAngleSeries("skewed", sa_histogram(skewed)["counts"])
    assert piled.median() > piled.mean()
    assert piled.median() == pytest.approx(float(np.median(skewed)), abs=1.0 / SA_HISTOGRAM_BINS)

    # Interpolation is what keeps every near-perfect dataset off one shared bin-center value.
    lopsided = SpectralAngleSeries("lopsided", [0] * (SA_HISTOGRAM_BINS - 1) + [1000])
    assert lopsided.median() == pytest.approx(0.99, abs=1e-9)


def test_spectral_angle_violins_render_three_series_per_dataset(tmp_path):
    """The panel the corpus, the teacher and the student are all binned for.

    All three arrive as counts on the same grid, so the figure is the only place they meet; this
    checks the renderer accepts that shape and tolerates a series that is missing for a dataset.
    """
    pytest.importorskip("matplotlib")
    from msspeculator.diagnostics import (
        SA_HISTOGRAM_BINS,
        SpectralAngleSeries,
        plot_spectral_angle_violins,
        sa_histogram,
    )

    rng = np.random.default_rng(0)

    def series(label: str, center: float, n: int) -> SpectralAngleSeries:
        values = np.clip(rng.normal(center, 0.08, n), 0.0, 1.0)
        return SpectralAngleSeries(label, sa_histogram(values)["counts"])

    groups = [
        (
            "prospect_tum_hla",
            [
                series("student", 0.78, 4000),
                series("teacher", 0.66, 4000),
                series("ceiling", 0.95, 900),
            ],
        ),
        (
            # A dataset the teacher cannot be asked about still has a student and a ceiling.
            "tmt_tum_hla",
            [
                series("student", 0.60, 2000),
                SpectralAngleSeries("teacher", [0] * SA_HISTOGRAM_BINS),
                series("ceiling", 0.93, 500),
            ],
        ),
    ]
    path = plot_spectral_angle_violins(groups, tmp_path / "violins.png")
    assert path.exists() and path.stat().st_size > 5_000

    with pytest.raises(ValueError, match="at least one group"):
        plot_spectral_angle_violins([], tmp_path / "empty.png")

    # A corpus-sized panel wraps instead of growing one unreadable axis. Rows make the figure
    # taller and no wider than a single full row, which is the whole point of wrapping.
    many = [(f"dataset_{index:02d}", groups[index % 2][1]) for index in range(25)]
    wrapped = plot_spectral_angle_violins(many, tmp_path / "wrapped.png", groups_per_row=8)
    single = plot_spectral_angle_violins(many[:8], tmp_path / "single.png", groups_per_row=8)
    assert wrapped.exists()
    from PIL import Image

    with Image.open(wrapped) as figure, Image.open(single) as one_row:
        assert figure.width == one_row.width
        assert figure.height > 2.5 * one_row.height  # 25 groups over 8 columns is four rows

    with pytest.raises(ValueError, match="groups_per_row must be positive"):
        plot_spectral_angle_violins(groups, tmp_path / "bad.png", groups_per_row=0)


def test_reference_distributions_load_and_panel_uses_them(tmp_path):
    """The training panel reads whichever reference lines a corpus publishes."""
    pytest.importorskip("matplotlib")
    import json

    from msspeculator.diagnostics import SA_HISTOGRAM_BINS, SpectralAngleSeries, sa_histogram
    from msspeculator.training_diagnostics import load_reference_distributions

    rng = np.random.default_rng(1)
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()

    def counts(center: float, n: int) -> list[int]:
        return sa_histogram(np.clip(rng.normal(center, 0.07, n), 0.0, 1.0))["counts"]

    (diagnostics / "teacher-yardstick.json").write_text(
        json.dumps(
            {"per_dataset": {"ptm": {"spectral_angle_histogram": {"counts": counts(0.66, 900)}}}}
        )
    )
    # Both subsets are published and the loader must take the in-window one: retention caps a
    # context at two PSMs, so the retained subset's leave-one-out score is pairwise agreement
    # between two noisy replicates and sits below what a model can actually reach.
    (diagnostics / "curation-summary.json").write_text(
        json.dumps(
            {
                "achievable_ceiling": {
                    "per_source": {
                        "ptm": {
                            "within_apex_window": counts(0.94, 400),
                            "selected": counts(0.70, 200),
                        }
                    }
                }
            }
        )
    )

    references = load_reference_distributions(str(tmp_path))
    assert sorted(references["ptm"]) == ["ceiling", "teacher"]
    ceiling = SpectralAngleSeries("ceiling", references["ptm"]["ceiling"])
    assert ceiling.total() == 400 and ceiling.mean() == pytest.approx(0.94, abs=0.02)

    # A prefix publishing neither report must not abort a training run; but it must not be
    # silent either, or a missing panel is indistinguishable from an unconfigured one.
    with pytest.warns(RuntimeWarning, match="publishes no"):
        assert load_reference_distributions(str(tmp_path / "absent")) == {}

    # Build the real renderer so the prefix is loaded through its own constructor.
    from msspeculator.training_diagnostics import TrainingDiagnosticRenderer

    renderer = TrainingDiagnosticRenderer(
        tmp_path / "out", butterflies=2, reference_prefix=str(tmp_path)
    )
    assert sorted(renderer.reference_distributions["ptm"]) == ["ceiling", "teacher"]

    student = {"ptm": counts(0.80, 700), "unknown_dataset": counts(0.5, 10)}
    path = renderer.spectral_angle_panel(student, tmp_path / "panel.png")
    assert path is not None and path.exists()
    # A dataset with no published reference is skipped rather than drawn against nothing.
    assert (
        renderer.spectral_angle_panel(
            {"unknown_dataset": [0] * SA_HISTOGRAM_BINS}, tmp_path / "x.png"
        )
        is None
    )
