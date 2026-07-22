import math

from pepdistill.chem import (
    H2O,
    PROTON,
    ION_TYPES,
    Peptide,
    fragment_mz,
    fragment_mz_matrix,
    ms2_target_shape,
)


def test_peptide_monoisotopic_mass():
    # PEPTIDE monoisotopic neutral mass is 799.35997 Da.
    pep = Peptide("PEPTIDE")
    assert math.isclose(pep.mono_mass(), 799.35997, abs_tol=0.01)


def test_precursor_mz_charge2():
    pep = Peptide("PEPTIDE")
    expected = (799.35997 + 2 * PROTON) / 2
    assert math.isclose(pep.precursor_mz(2), expected, abs_tol=0.01)


def test_y1_of_c_terminal_lysine():
    rm = Peptide("SAMPLEK").residue_masses()
    # y1 = K residue + water + proton
    y1 = fragment_mz(rm, "y", 1, 1)
    assert math.isclose(y1, 128.094963 + H2O + PROTON, abs_tol=0.001)


def test_b_plus_y_complementarity():
    pep = Peptide("SAMPLER")
    rm = pep.residue_masses()
    n = pep.length
    # b_i(+H) + y_(n-i)(+H) = M + 2*proton  (singly charged neutral complementarity)
    for i in range(1, n):
        b = fragment_mz(rm, "b", i, 1)
        y = fragment_mz(rm, "y", n - i, 1)
        assert math.isclose(b + y, pep.mono_mass() + 2 * PROTON, abs_tol=0.001)


def test_fixed_mod_shifts_mass():
    base = Peptide("ACDEK")
    mod = Peptide("ACDEK", ((1, "Carbamidomethyl@C"),))
    assert math.isclose(mod.mono_mass() - base.mono_mass(), 57.021463723, abs_tol=1e-6)


def test_modified_sequence_rendering():
    pep = Peptide("ACDEMK", ((1, "Carbamidomethyl@C"), (4, "Oxidation@M")))
    assert pep.modified_sequence() == "AC[Carbamidomethyl@C]DEM[Oxidation@M]K"


def test_fragment_matrix_shape():
    pep = Peptide("SAMPLER")
    mat = fragment_mz_matrix(pep)
    assert (len(mat), len(mat[0])) == ms2_target_shape(pep.length)
    assert len(mat[0]) == len(ION_TYPES)
