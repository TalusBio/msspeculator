"""Composition channels: what collate puts in mod_comp / mod_mass and the two masks."""

import pytest

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor

TMT_MASS = 229.1629321
PHOSPHO_MASS = 79.9663312


def test_unmodified_peptide_has_no_mod_signal():
    b = collate([Precursor(Peptide("SAMPLER"), 2, "t")])
    assert not b.mod_present.any()
    assert not b.mod_has_composition.any()
    assert float(b.mod_comp.abs().max()) == 0.0
    assert float(b.mod_mass.abs().max()) == 0.0


def test_named_mod_populates_comp_and_mass_at_its_column():
    b = collate([Precursor(Peptide("ACDEK", ((1, "Carbamidomethyl@C"),)), 2, "t")])
    col = 1 + 1  # N-term token at 0, residue i at 1+i
    assert bool(b.mod_present[0, col]) and bool(b.mod_has_composition[0, col])
    # Carbamidomethyl is C2H3NO; ELEMENTS order is C, H, N, O, S, P.
    assert b.mod_comp[0, col].tolist() == [2.0, 3.0, 1.0, 1.0, 0.0, 0.0]
    assert abs(float(b.mod_mass[0, col]) - 57.021463723) < 1e-4
    assert not b.mod_present[0, :col].any() and not b.mod_present[0, col + 1 :].any()


def test_mass_is_unscaled_daltons():
    b = collate([Precursor(Peptide("PEPTIDEK", (("n", "TMT6plex"),)), 2, "t")])
    assert abs(float(b.mod_mass[0, 0]) - TMT_MASS) < 1e-3


def test_tmt_comp_and_mass_diverge_as_designed():
    b = collate([Precursor(Peptide("PEPTIDEK", (("n", "TMT6plex"),)), 2, "t")])
    assert b.mod_comp[0, 0].tolist() == [12.0, 20.0, 2.0, 2.0, 0.0, 0.0]
    assert abs(float(b.mod_mass[0, 0]) - TMT_MASS) < 1e-3  # not the element view's 224.15


def test_nterm_and_residue_zero_occupy_different_columns():
    p = Peptide("KPEPTIDE", (("n", "TMT6plex"), (0, "TMT6plex")))
    b = collate([Precursor(p, 2, "t")])
    assert bool(b.mod_present[0, 0]) and bool(b.mod_present[0, 1])
    assert abs(float(b.mod_mass[0, 0]) - TMT_MASS) < 1e-3
    assert abs(float(b.mod_mass[0, 1]) - TMT_MASS) < 1e-3


def test_cterm_mod_lands_on_the_last_column():
    p = Peptide("PEK", (("c", "Phospho"),))
    b = collate([Precursor(p, 2, "t")])
    col = 1 + 3  # [N] P E K [C] -> C-term token at index 4
    assert bool(b.mod_present[0, col])
    assert abs(float(b.mod_mass[0, col]) - PHOSPHO_MASS) < 1e-4


def test_mass_only_mod_is_present_but_not_named():
    b = collate([Precursor(Peptide("PEPTIDE", ((2, 42.010565),)), 2, "t")])
    col = 1 + 2
    assert bool(b.mod_present[0, col])
    assert not bool(b.mod_has_composition[0, col])
    assert float(b.mod_comp[0, col].abs().max()) == 0.0
    assert abs(float(b.mod_mass[0, col]) - 42.010565) < 1e-5


def test_two_named_mods_on_one_column_accumulate_comp_and_mass():
    p = Peptide("CPEPTIDE", ((0, "Carbamidomethyl@C"), (0, "Oxidation@M")))
    b = collate([Precursor(p, 2, "t")])
    col = 1
    assert bool(b.mod_has_composition[0, col]) and bool(b.mod_present[0, col])
    # Carbamidomethyl C2H3NO + Oxidation O, in ELEMENTS order C, H, N, O, S, P.
    assert b.mod_comp[0, col].tolist() == [2.0, 3.0, 1.0, 2.0, 0.0, 0.0]
    assert abs(float(b.mod_mass[0, col]) - (57.021464 + 15.994915)) < 1e-4


def test_two_mass_only_mods_on_one_column_sum():
    p = Peptide("CPEPTIDE", ((0, 57.021464), (0, 15.994915)))
    b = collate([Precursor(p, 2, "t")])
    col = 1
    assert bool(b.mod_present[0, col]) and not bool(b.mod_has_composition[0, col])
    assert float(b.mod_comp[0, col].abs().max()) == 0.0
    assert abs(float(b.mod_mass[0, col]) - (57.021464 + 15.994915)) < 1e-4


def test_named_plus_mass_only_on_one_column_is_refused():
    """`mod_has_composition` is one boolean per column, so the column routes wholly through comp_enc or
    wholly through mass_enc. A site holding one of each would drop a channel from the model
    input while still moving mono_mass and every fragment m/z — refuse instead."""
    p = Peptide("PEPCIDER", ((3, "Carbamidomethyl@C"), (3, 15.994915)))
    with pytest.raises(Exception) as e:
        collate([Precursor(p, 2, "t")])
    # Diagnostics quote the same ProForma descriptor as emission, so the named mod appears as
    # its accession rather than as the internal alias it was supplied with.
    assert "UNIMOD:4" in str(e.value) and "15.994915" in str(e.value)
    # The mass path must agree, or a library row would carry m/z for an unencodable molecule.
    with pytest.raises(Exception):
        p.mono_mass()


def test_nterm_named_plus_residue_zero_mass_only_stays_legal():
    """The refusal is per SITE, not per residue index: these are two different columns even
    though residue_masses folds an N-terminal delta onto residue 0."""
    p = Peptide("KPEPTIDE", (("n", "TMT6plex"), (0, 15.994915)))
    b = collate([Precursor(p, 2, "t")])
    assert bool(b.mod_has_composition[0, 0]) and not bool(b.mod_has_composition[0, 1])
    assert abs(float(b.mod_mass[0, 1]) - 15.994915) < 1e-5
    assert p.mono_mass() > 0


def test_mod_scale_is_retired():
    import pepdistill_rs

    assert not hasattr(pepdistill_rs, "MOD_SCALE")
