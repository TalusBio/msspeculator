"""Chemistry-preserving residue-substitution augmentation."""

import pytest
import torch

from pepdistill.chem import RESIDUE_COMP, RESIDUE_MASS, Peptide
from pepdistill.data.augmentation import substitute_residues
from pepdistill.data.encode import AA_OFFSET, collate
from pepdistill.data.precursors import Precursor


def _batch(sequence: str, mods=()):
    return collate([Precursor(Peptide(sequence, mods), 2, "train")])


def _amino_acid(token: int) -> str:
    return chr(token + AA_OFFSET)


def test_substitution_preserves_effective_composition_and_mass():
    original = _batch("ACDEFGHIK")
    torch.manual_seed(7)
    augmented = substitute_residues(original, 1.0)

    changed = (augmented.tokens != original.tokens).nonzero(as_tuple=False)
    assert changed.shape == (1, 2)  # at most, and with p=1, exactly, one site per peptide
    row, column = changed[0].tolist()
    old = _amino_acid(int(original.tokens[row, column]))
    new = _amino_acid(int(augmented.tokens[row, column]))
    expected_comp = torch.tensor(RESIDUE_COMP[old], dtype=torch.float32)
    effective_comp = (
        torch.tensor(RESIDUE_COMP[new], dtype=torch.float32) + augmented.mod_comp[row, column]
    )
    assert torch.equal(effective_comp, expected_comp)
    assert float(RESIDUE_MASS[new] + augmented.mod_mass[row, column]) == pytest.approx(
        RESIDUE_MASS[old], abs=1e-5
    )
    assert bool(augmented.mod_present[row, column])
    assert bool(augmented.mod_has_composition[row, column])
    assert not original.mod_present.any(), "augmentation must not mutate the source batch"


def test_composition_routed_modification_accumulates_with_compensating_delta():
    original = _batch("C", ((0, "UNIMOD:4"),))
    torch.manual_seed(3)
    augmented = substitute_residues(original, 1.0)
    new = _amino_acid(int(augmented.tokens[0, 1]))

    effective = torch.tensor(RESIDUE_COMP[new], dtype=torch.float32) + augmented.mod_comp[0, 1]
    original_effective = (
        torch.tensor(RESIDUE_COMP["C"], dtype=torch.float32) + original.mod_comp[0, 1]
    )
    assert torch.equal(effective, original_effective)
    assert float(RESIDUE_MASS[new] + augmented.mod_mass[0, 1]) == pytest.approx(
        RESIDUE_MASS["C"] + float(original.mod_mass[0, 1]), abs=1e-5
    )


def test_mass_only_sites_are_not_eligible():
    original = _batch("A", ((0, 1.2345),))
    augmented = substitute_residues(original, 1.0)
    assert augmented is original


def test_isomeric_substitution_without_a_delta_is_excluded():
    original = _batch("I")
    torch.manual_seed(5)
    augmented = substitute_residues(original, 1.0)
    replacement = _amino_acid(int(augmented.tokens[0, 1]))
    assert replacement not in {"I", "L"}
    assert bool(augmented.mod_has_composition[0, 1])
    assert augmented.mod_comp[0, 1].ne(0).any()


def test_probability_zero_is_a_noop_and_invalid_probability_errors():
    original = _batch("PEPTIDE")
    assert substitute_residues(original, 0.0) is original
    with pytest.raises(ValueError, match="between 0 and 1"):
        substitute_residues(original, 1.01)


def test_every_selected_peptide_gets_only_one_substitution():
    original = collate(
        [Precursor(Peptide(sequence), 2, "train") for sequence in ("PEPTIDE", "SAMPLER", "ACDK")]
    )
    torch.manual_seed(11)
    augmented = substitute_residues(original, 1.0)
    assert (augmented.tokens != original.tokens).sum(dim=1).tolist() == [1, 1, 1]
