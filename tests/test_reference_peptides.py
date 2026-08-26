import csv
import re
from pathlib import Path

import pytest

from msspeculator.diagnostics import IRT_STANDARDS


PRTC_PATH = Path(__file__).parents[1] / "data" / "reference_peptides" / "prtc.tsv"
IRT_PATH = Path(__file__).parents[1] / "data" / "reference_peptides" / "biognosys_irt.tsv"
IRT_TRANSITIONS_PATH = (
    Path(__file__).parents[1] / "data" / "reference_peptides" / "biognosys_irt_transitions.tsv"
)


def test_prtc_transcription_is_complete_and_proforma_labeled():
    with PRTC_PATH.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    assert [int(row["peptide_number"]) for row in rows] == list(range(1, 16))
    assert rows[0]["sequence"] == "SSAAPPPPPR"
    assert rows[-1]["sequence"] == "LSSEAPALFQFDLK"
    for row in rows:
        accession = "267" if row["sequence"].endswith("R") else "259"
        assert row["proforma_sequence"] == f"{row['sequence']}[UNIMOD:{accession}]"
        assert re.fullmatch(r"[A-Z]+\[UNIMOD:(259|267)\]", row["proforma_sequence"])


def test_prtc_reported_mass_and_charge_two_mz_are_consistent():
    proton_mass = 1.007276466621
    with PRTC_PATH.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    for row in rows:
        expected_mz = (float(row["neutral_monoisotopic_mass"]) + 2 * proton_mass) / 2
        assert float(row["observed_mz_z2"]) == pytest.approx(expected_mz, abs=0.0002)


def test_biognosys_irt_standards_and_transitions_agree():
    with IRT_PATH.open(newline="") as stream:
        standards = list(csv.DictReader(stream, delimiter="\t"))
    with IRT_TRANSITIONS_PATH.open(newline="") as stream:
        transitions = list(csv.DictReader(stream, delimiter="\t"))

    assert len(standards) == 11
    assert len(transitions) == 33
    assert {row["proforma_sequence"] for row in standards} == {
        row["nominal sequence"] for row in transitions
    }
    assert all(int(row["precursor charge"]) == 2 for row in transitions)
    assert float(standards[0]["irt"]) == pytest.approx(-24.916114)
    assert float(standards[-1]["irt"]) == pytest.approx(100.00282166666665)
    for expected, row in zip(IRT_STANDARDS, standards, strict=True):
        assert expected.sequence == row["proforma_sequence"]
        assert expected.charge == int(row["precursor_charge"])
        assert expected.irt == pytest.approx(float(row["irt"]))
