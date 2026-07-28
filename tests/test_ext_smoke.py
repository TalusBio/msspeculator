import math
import pytest

rs = pytest.importorskip("pepdistill_rs")


def test_peptide_via_ext():
    p = rs.Peptide("PEPTIDE")
    assert math.isclose(p.mono_mass(), 799.35997, abs_tol=0.01)
    assert math.isclose(p.precursor_mz(2), (799.35997 + 2 * rs.PROTON) / 2, abs_tol=0.01)
    q = rs.Peptide("ACDEMK", [(1, "Carbamidomethyl@C"), (4, "Oxidation@M")])
    assert q.modified_sequence() == "AC[Carbamidomethyl@C]DEM[Oxidation@M]K"
    # hashable + value equality (canonical mods)
    assert hash(q) == hash(rs.Peptide("ACDEMK", [(4, "Oxidation@M"), (1, "Carbamidomethyl@C")]))
    assert q == rs.Peptide("ACDEMK", [(4, "Oxidation@M"), (1, "Carbamidomethyl@C")])


def test_constants_and_ion_types():
    assert rs.PAD_IDX == 26 and rs.N_TOKENS == 29
    assert rs.ION_TYPES == [("b", 1), ("y", 1), ("b", 2), ("y", 2)]
    assert math.isclose(rs.MOD_DELTA["Carbamidomethyl@C"], 57.021463723, abs_tol=1e-6)
