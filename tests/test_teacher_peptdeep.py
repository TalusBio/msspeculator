"""Real AlphaPeptDeep teacher tests. Skipped unless peptdeep is installed."""

import pytest

from pepdistill.chem import Peptide
from pepdistill.data.precursors import Precursor

peptdeep = pytest.importorskip("peptdeep")


@pytest.fixture(scope="module")
def teacher():
    from pepdistill.teacher import get_teacher

    return get_teacher("alphapeptdeep", device="cpu")


def test_labels_align_with_input_order(teacher):
    # Varied lengths so predict_all reorders by nAA — the label list must come back
    # in the ORIGINAL order, with rows == length - 1 for each precursor.
    seqs = ["SAMPLERPEPTIDEK", "ACDK", "AAAAAAAAAAAAR", "SAMPLER", "MYPEPTIDEKACDEFGHIK"]
    precs = [Precursor(Peptide(s), 2, "train") for s in seqs]
    labels = teacher.predict(precs)

    assert len(labels) == len(precs)
    for prec, lab in zip(precs, labels):
        assert lab.ms2.shape[0] == prec.peptide.length - 1
        assert lab.ms2.shape[1] == 4
        assert 0.0 <= lab.ms2.max() <= 1.0 + 1e-6


def test_modified_peptide_gets_labels(teacher):
    prec = Precursor(Peptide("ACDEMK", ((1, "Carbamidomethyl@C"), (4, "Oxidation@M"))), 2, "train")
    (lab,) = teacher.predict([prec])
    assert lab.ms2.shape[0] == 5
    assert lab.rt == lab.rt  # not NaN
