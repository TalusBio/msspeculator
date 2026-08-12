from pepdistill.chem import Peptide
from pepdistill.proforma import proforma_sequence


def test_proforma_uses_accessions_and_terminal_syntax():
    peptide = Peptide(
        "ACDMK",
        (
            ("n", "TMT6plex"),
            (1, "Carbamidomethyl@C"),
            (3, "Oxidation@M"),
            ("c", "Amidated"),
        ),
    )
    assert proforma_sequence(peptide) == ("[UNIMOD:737]-AC[UNIMOD:4]DM[UNIMOD:35]K-[UNIMOD:2]")


def test_proforma_preserves_mass_only_modifications():
    peptide = Peptide("PEPTIDE", ((2, 42.010565),))
    assert proforma_sequence(peptide) == "PEP[+42.010565]TIDE"
