import numpy as np
import pytest

from pepdistill.diagnostics import (
    DiagnosticReferencePanel,
    EmbeddingConnection,
    IRT_STANDARDS,
    LabeledEmbedding,
    PcaBasis,
    ReferenceSpectrum,
    RtObservation,
    SpectrumComparison,
    normalized_spectral_angle,
    plot_embedding_pca,
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


def test_reference_panel_roundtrips_and_reports_teacher_yardstick(tmp_path):
    experimental = np.asarray([[1.0, 0.2], [0.0, 0.5]], dtype=np.float32)
    teacher = experimental.copy()
    panel = DiagnosticReferencePanel(
        (
            ReferenceSpectrum(
                dataset="pool",
                sequence="PEPTIDEK",
                serialized_mods="",
                charge=2,
                fragment_mz=np.asarray([[100.0, 200.0], [300.0, 400.0]]),
                experimental_intensity=experimental,
                teacher_intensity=teacher,
            ),
        )
    )
    path = tmp_path / "panel.npz"
    panel.save(path)
    loaded = DiagnosticReferencePanel.load(path)

    assert loaded.spectra[0].sequence == "PEPTIDEK"
    np.testing.assert_array_equal(loaded.spectra[0].teacher_intensity, teacher)
    assert loaded.teacher_yardstick() == {"pool": pytest.approx(1.0)}


def test_plot_prototypes_write_pngs(tmp_path):
    pca_path = plot_embedding_pca(
        np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
        lengths=[8, 10],
        modified=[False, True],
        path=tmp_path / "pca.png",
        title="diagnostic",
        explained_variance_ratio=[0.8, 0.2],
    )
    butterfly_path = plot_spectrum_butterflies(
        [
            SpectrumComparison(
                modified_sequence="PEPTIDEK",
                charge=2,
                fragment_mz=np.asarray([[100.0, 200.0]]),
                student_intensity=np.asarray([[1.0, 0.2]]),
                reference_intensity=np.asarray([[0.8, 0.3]]),
                reference_name="experiment",
            )
        ],
        tmp_path / "butterfly.png",
    )
    assert pca_path.stat().st_size > 0
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
