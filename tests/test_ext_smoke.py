import math
import numpy as np
import pytest

rs = pytest.importorskip("pepdistill_rs")


def test_peptide_via_ext():
    p = rs.Peptide("PEPTIDE")
    assert math.isclose(p.mono_mass(), 799.35997, abs_tol=0.01)
    assert math.isclose(p.precursor_mz(2), (799.35997 + 2 * rs.PROTON) / 2, abs_tol=0.01)
    q = rs.Peptide("ACDEMK", [(1, "Carbamidomethyl@C"), (4, "Oxidation@M")])
    # Internal alias names are accepted on input but never emitted: a ProForma descriptor must
    # parse back, and "[Carbamidomethyl@C]" does not.
    assert q.modified_sequence() == "AC[UNIMOD:4]DEM[UNIMOD:35]K"
    # hashable + value equality (canonical mods)
    assert hash(q) == hash(rs.Peptide("ACDEMK", [(4, "Oxidation@M"), (1, "Carbamidomethyl@C")]))
    assert q == rs.Peptide("ACDEMK", [(4, "Oxidation@M"), (1, "Carbamidomethyl@C")])


def test_collate_prepared_matches_object_collate():
    # The prepared column is canonical ProForma, so the collator and the object path must agree
    # about the same peptide expressed two ways.
    proforma = ["AC[UNIMOD:4]DEM[UNIMOD:35]K", "[UNIMOD:737]-PEP[+15.5]TID"]
    charges = [2, 3]
    prepared = rs.collate_prepared(proforma, charges)
    objects = rs.collate(
        [
            rs.Peptide("ACDEMK", [(1, "Carbamidomethyl@C"), (4, "Oxidation@M")]),
            rs.Peptide("PEPTID", [("n", "TMT6plex"), (2, 15.5)]),
        ],
        charges,
    )
    assert prepared.keys() == objects.keys()
    for key in prepared:
        np.testing.assert_array_equal(prepared[key], objects[key])


def test_constants_and_ion_types():
    assert rs.PAD_IDX == 26 and rs.N_TOKENS == 29
    assert rs.ION_TYPES == [("b", 1), ("y", 1), ("b", 2), ("y", 2)]
    assert math.isclose(rs.MOD_DELTA["Carbamidomethyl@C"], 57.021463723, abs_tol=1e-6)
