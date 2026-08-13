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
    prec = Precursor(Peptide("ACDEMK", ((1, "UNIMOD:4"), (4, "UNIMOD:35"))), 2, "train")
    (lab,) = teacher.predict([prec])
    assert lab.ms2.shape[0] == 5
    assert lab.rt == lab.rt  # not NaN


def test_alphabase_modification_conventions_we_would_map_onto():
    """Pin the upstream facts any bare-name -> alphabase-name mapping would rest on.

    Our prepared rows carry a mixed vocabulary: common mods are already stored alphabase-style
    ("UNIMOD:4") while PROSPECT mods are bare UNIMOD titles ("UNIMOD:737"). Mapping the
    latter is only safe while these upstream properties hold, and each one silently produces a
    plausible wrong spectrum if it changes, so assert them rather than assume them.
    """
    from alphabase.constants.modification import MOD_DF, MOD_MASS, calc_modification_mass
    from peptdeep.model.featurize import MOD_TO_FEATURE

    # 1. Every named modification is composition-encodable; there is no unsupported class.
    assert set(MOD_MASS) == set(MOD_TO_FEATURE)

    # 2. Site convention: 0 is peptide N-term, -1 is C-term, and 1..nAA are 1-based residues.
    tmt = MOD_MASS["TMT6plex@Any_N-term"]
    assert calc_modification_mass(5, ["TMT6plex@Any_N-term"], [0])[0] == pytest.approx(tmt)
    assert calc_modification_mass(5, ["TMT6plex@K"], [-1])[-1] == pytest.approx(tmt)
    phospho = MOD_MASS["Phospho@S"]
    assert calc_modification_mass(5, ["Phospho@S"], [3])[2] == pytest.approx(phospho)

    # 3. Names are not always "Name@Residue": residue-anchored terminal mods exist only in a
    # compound form, and each pyro-Glu accession we carry resolves to exactly one entry.
    assert list(MOD_DF[MOD_DF.unimod_id == 27].mod_name) == ["Glu->pyro-Glu@E^Any_N-term"]
    assert list(MOD_DF[MOD_DF.unimod_id == 28].mod_name) == ["Gln->pyro-Glu@Q^Any_N-term"]

    # 4. Already-qualified names must be passed through, never suffixed a second time.
    assert "Carbamidomethyl@C" in MOD_MASS
    assert "Carbamidomethyl@C@C" not in MOD_MASS


def test_peptdeep_predicts_terminal_and_residue_mods_with_exact_masses(teacher):
    """peptdeep accepts the terminal convention end-to-end, and applies both mods for real.

    Masses are checked because a dropped or misplaced modification still yields a full,
    confident spectrum -- the failure mode our teacher wrapper refuses to risk.
    """
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "sequence": "DNTAEWDHK",
                "mods": "TMT6plex@Any_N-term;Phospho@T",
                "mod_sites": "0;3",
                "charge": 2,
                "nce": 30.0,
                "instrument": "Lumos",
            },
            {
                "sequence": "PEPTIDEK",
                "mods": "",
                "mod_sites": "",
                "charge": 2,
                "nce": 30.0,
                "instrument": "Lumos",
            },
        ]
    )
    result = teacher._mgr.predict_all(frame, predict_items=["ms2"], multiprocessing=False)
    predicted = result["precursor_df"]
    # predict_all sorts by length, so look the rows up by sequence rather than by position.
    by_sequence = dict(zip(predicted["sequence"], predicted["precursor_mz"]))
    assert by_sequence["PEPTIDEK"] == pytest.approx(464.7347, abs=1e-3)
    assert by_sequence["DNTAEWDHK"] == pytest.approx(712.8059, abs=1e-3)
    assert np.isfinite(result["fragment_intensity_df"].to_numpy()).all()


def test_peptdeep_frame_refuses_mods_it_cannot_represent():
    """A mass-only delta cannot be named, and an unknown name is not silently substituted."""
    import pytest

    from pepdistill.chem import Peptide
    from pepdistill.teacher.peptdeep_teacher import _alphabase_mod, _modification_title

    mass_only = Peptide("PEPTIDE", ((2, 42.010565),))
    with pytest.raises(ValueError, match="mass-only"):
        _modification_title(mass_only, mass_only.mods[0][1])

    # A legacy name is refused for what it is, rather than searched for: identity is an accession
    # everywhere past ingest, so a name here means something upstream skipped the boundary.
    named = Peptide("PEPTIDE", ((2, "NotARealModification"),))
    with pytest.raises(ValueError, match="not a UNIMOD accession"):
        _alphabase_mod(named, *named.mods[0])

    # An accession the vendored table does know, but alphabase does not register for that residue,
    # is refused after the candidate search rather than relocated.
    unregistered = Peptide("PEPTIDE", ((2, "UNIMOD:21"),))
    with pytest.raises(ValueError, match="does not resolve"):
        _alphabase_mod(unregistered, *unregistered.mods[0])


def test_alphabase_mod_resolution_covers_our_mixed_vocabulary():
    """Our names are a mix of alphabase-style aliases and bare UNIMOD titles; both must map.

    Site and name are resolved together because they are coupled: our residue index 0 becomes
    alphabase site 1 for a side-chain mod but site 0 for a residue-anchored terminal mod.
    """
    from pepdistill.chem import Peptide
    from pepdistill.teacher.peptdeep_teacher import _alphabase_mod

    # Already alphabase-style (frozen aliases and the digest path) pass through untouched.
    qualified = Peptide("ACDEMK", ((1, "UNIMOD:4"), (4, "UNIMOD:35")))
    assert _alphabase_mod(qualified, 1, "UNIMOD:4") == ("Carbamidomethyl@C", 2)
    assert _alphabase_mod(qualified, 4, "UNIMOD:35") == ("Oxidation@M", 5)

    # Bare titles gain the residue suffix; the site stays 1-based.
    phospho = Peptide("SAMPLER", ((0, "UNIMOD:21"),))
    assert _alphabase_mod(phospho, 0, "UNIMOD:21") == ("Phospho@S", 1)

    # Terminal markers become positional names filed at site 0 / -1.
    tmt = Peptide("DNTAEWDHK", (("n", "UNIMOD:737"), (8, "UNIMOD:737")))
    assert _alphabase_mod(tmt, "n", "UNIMOD:737") == ("TMT6plex@Any_N-term", 0)
    assert _alphabase_mod(tmt, 8, "UNIMOD:737") == ("TMT6plex@K", 9)

    # A residue-suffixed alias found on a TERMINUS must not be filed on residue 1. Our site is
    # the authority on placement, so the terminal form is required or the mod is refused --
    # `parse_modseq("[UNIMOD:35]METIDEK")` really does yield ("n", "UNIMOD:35"), and filing it
    # at residue 1 returned a confident, plausible, wrong spectrum.
    import pytest as _pytest

    nterm_cam = Peptide("CDEMK", (("n", "UNIMOD:4"),))
    assert _alphabase_mod(nterm_cam, "n", "UNIMOD:4") == ("Carbamidomethyl@Any_N-term", 0)
    nterm_ox = Peptide("METIDEK", (("n", "UNIMOD:35"),))
    with _pytest.raises(ValueError, match="does not resolve"):
        # alphabase registers no N-terminal Oxidation, so refuse rather than relocate it.
        _alphabase_mod(nterm_ox, "n", "UNIMOD:35")

    # An accession on an unusual-but-real residue now resolves, where a name did not. This used to
    # raise "refusing to relocate", because our name declared `@C` and the site was a lysine -- but
    # that check was enforcing our own spelling, not chemistry: carbamidomethyl-lysine exists and
    # alphabase registers it. An accession declares no residue, so there is nothing to contradict,
    # and alphabase's table is the authority on whether the pairing is real.
    assert _alphabase_mod(Peptide("AKDEMK", ((1, "UNIMOD:4"),)), 1, "UNIMOD:4") == (
        "Carbamidomethyl@K",
        2,
    )

    # Residue-anchored terminal mods exist only in compound form and are filed at site 0,
    # even though our representation stores them on residue index 0.
    pyro_e = Peptide("EPTIDEK", ((0, "UNIMOD:27"),))
    assert _alphabase_mod(pyro_e, 0, "UNIMOD:27") == ("Glu->pyro-Glu@E^Any_N-term", 0)
    pyro_q = Peptide("QPTIDEK", ((0, "UNIMOD:28"),))
    assert _alphabase_mod(pyro_q, 0, "UNIMOD:28") == ("Gln->pyro-Glu@Q^Any_N-term", 0)


def test_accessions_are_translated_at_the_peptdeep_boundary():
    """Our canonical identity is a UNIMOD accession; peptdeep needs a name. Translate once, here.

    Prepared shards carry accessions, so without this every modified spectrum raised: `UNIMOD:21`
    became the candidate `UNIMOD:21@S`, which is in no alphabase table. Resolved through the
    vendored UNIMOD table so there is no second mapping to drift, and an accession the table does
    not know is refused rather than guessed at.
    """
    from pepdistill.chem import Peptide
    from pepdistill.teacher.peptdeep_teacher import _alphabase_mod

    for proforma, expected in (
        ("PEPS[UNIMOD:21]IDEK", [("Phospho@S", 4)]),
        ("[UNIMOD:737]-PEPTIDEK", [("TMT6plex@Any_N-term", 0)]),
        ("AC[UNIMOD:4]DEM[UNIMOD:35]K", [("Carbamidomethyl@C", 2), ("Oxidation@M", 5)]),
        ("PEPTIDEK[UNIMOD:1]", [("Acetyl@K", 8)]),
        ("PEPTIDEK[UNIMOD:121]", [("GG@K", 8)]),
    ):
        peptide = Peptide.from_string(proforma)
        resolved = [_alphabase_mod(peptide, site, spec) for site, spec in peptide.mods]
        assert resolved == expected, proforma

    # A mass-only modification has no name to translate to, and is refused rather than invented.
    with pytest.raises(ValueError, match="mass-only modification"):
        mass_only = Peptide.from_string("PEP[+15.5]TIDEK")
        [_alphabase_mod(mass_only, site, spec) for site, spec in mass_only.mods]
