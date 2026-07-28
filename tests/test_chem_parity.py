"""Transitional: the Rust ext must match the (still-present) Python chem reference.
Deleted in the final task once Python chem is gone."""
import math
import pytest

import pepdistill.chem as pychem  # OLD python implementation, still present here

rs = pytest.importorskip("pepdistill_rs")

CASES = [
    ("PEPTIDE", []),
    ("SAMPLER", []),
    ("ACDEMK", [(1, "Carbamidomethyl@C"), (4, "Oxidation@M")]),
    ("MKLVR", [(0, "Oxidation@M")]),
]


@pytest.mark.parametrize("seq,mods", CASES)
def test_peptide_parity(seq, mods):
    pp = pychem.Peptide(seq, tuple(mods))
    rp = rs.Peptide(seq, mods)
    assert math.isclose(pp.mono_mass(), rp.mono_mass(), abs_tol=1e-9)
    for z in (1, 2, 3):
        assert math.isclose(pp.precursor_mz(z), rp.precursor_mz(z), abs_tol=1e-9)
    assert pp.modified_sequence() == rp.modified_sequence()
    assert list(pp.residue_masses()) == pytest.approx(list(rp.residue_masses()), abs=1e-9)


def test_constants_parity():
    assert rs.ION_TYPES == [(i, z) for i, z in pychem.ION_TYPES]
    for name, delta in pychem.MOD_DELTA.items():
        assert math.isclose(rs.MOD_DELTA[name], delta, abs_tol=1e-9)
