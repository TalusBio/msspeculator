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


def test_teacher_construction_does_not_emit_mask_modloss_warning():
    import warnings

    from pepdistill.teacher.peptdeep_teacher import PeptDeepTeacher

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PeptDeepTeacher(device="cpu")
    assert not any("mask_modloss is deprecated" in str(w.message) for w in caught)


def test_modified_peptide_gets_labels(teacher):
    prec = Precursor(Peptide("ACDEMK", ((1, "Carbamidomethyl@C"), (4, "Oxidation@M"))), 2, "train")
    (lab,) = teacher.predict([prec])
    assert lab.ms2.shape[0] == 5
    assert lab.rt == lab.rt  # not NaN


def test_peptdeep_frame_refuses_mods_it_cannot_represent():
    """peptdeep identifies mods by name and has its own terminal-site convention.

    Both cases used to raise a bare TypeError from deep inside a string join. Now they name
    the peptide and the reason. Terminal sites are refused rather than guessed: placing a real
    modification on the wrong residue would yield a confident, plausible, wrong spectrum.
    """
    import pytest

    from pepdistill.chem import Peptide
    from pepdistill.teacher.peptdeep_teacher import _mod_name, _mod_site

    mass_only = Peptide("PEPTIDE", ((2, 42.010565),))
    with pytest.raises(ValueError, match="mass-only"):
        _mod_name(mass_only, mass_only.mods[0][1])

    nterm = Peptide("PEPTIDE", (("n", "TMT6plex"),))
    with pytest.raises(NotImplementedError, match="terminal"):
        _mod_site(nterm, nterm.mods[0][0])

    # Residue sites still convert, 1-based.
    named = Peptide("ACDEK", ((1, "Carbamidomethyl@C"),))
    assert _mod_name(named, named.mods[0][1]) == "Carbamidomethyl@C"
    assert _mod_site(named, named.mods[0][0]) == 2
