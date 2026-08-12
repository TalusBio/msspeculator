"""Chemistry-preserving training augmentation in the model-input representation."""

from __future__ import annotations

from functools import lru_cache

import torch

from ..chem import RESIDUE_COMP, RESIDUE_MASS
from .encode import AA_OFFSET, N_TOKENS, Batch

_AMINO_ACIDS = tuple(sorted(RESIDUE_MASS))
_STANDARD_TOKENS = torch.tensor(
    [ord(amino_acid) - AA_OFFSET for amino_acid in _AMINO_ACIDS], dtype=torch.long
)
_TOKEN_TO_STANDARD = torch.full((N_TOKENS,), -1, dtype=torch.long)
_TOKEN_TO_STANDARD[_STANDARD_TOKENS] = torch.arange(len(_AMINO_ACIDS))
_COMPOSITIONS = torch.tensor(
    [RESIDUE_COMP[amino_acid] for amino_acid in _AMINO_ACIDS], dtype=torch.float32
)
_MASSES = torch.tensor(
    [RESIDUE_MASS[amino_acid] for amino_acid in _AMINO_ACIDS], dtype=torch.float32
)


@lru_cache(maxsize=None)
def _device_tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Copy the tiny chemistry lookups once per accelerator, not once per training batch."""
    return (
        _TOKEN_TO_STANDARD.to(device),
        _STANDARD_TOKENS.to(device),
        _COMPOSITIONS.to(device),
        _MASSES.to(device),
    )


def substitute_residues(batch: Batch, probability: float) -> Batch:
    """Mutate at most one residue per selected peptide without changing its chemistry.

    Each peptide is selected independently with ``probability``. One eligible residue is
    replaced by a uniformly sampled different standard amino acid, while the exact
    ``original - replacement`` elemental composition and mass are added as a modification at
    that site. Targets therefore remain valid: precursor and every fragment retain the same
    composition and mass, but the model must learn that residue token plus delta describes the
    original chemistry.

    A site carrying a mass-only modification is not eligible because the model deliberately
    refuses to mix mass-only and compositional modifications in one column. Existing named
    modifications are compositional and safely accumulate. Zero-composition substitutions such
    as I/L are excluded: without a compensating delta they would incorrectly teach direct token
    invariance rather than residue-plus-modification equivalence.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("residue substitution probability must be between 0 and 1")
    if probability == 0.0 or batch.tokens.numel() == 0:
        return batch

    device = batch.tokens.device
    token_to_standard, standard_tokens, compositions, masses = _device_tables(device)

    standard_index = token_to_standard[batch.tokens]
    eligible = (standard_index >= 0) & ~(batch.mod_present & ~batch.mod_named)
    selected_rows = (torch.rand(batch.tokens.shape[0], device=device) < probability) & eligible.any(
        dim=1
    )
    rows = selected_rows.nonzero(as_tuple=False).squeeze(1)
    if rows.numel() == 0:
        return batch

    # Independent random priorities choose one eligible site per selected peptide without a
    # Python loop; ineligible sites sort behind every eligible [0,1) priority.
    eligible_rows = eligible[rows]
    priorities = torch.rand(eligible_rows.shape, device=device).masked_fill(~eligible_rows, 2.0)
    columns = priorities.argmin(dim=1)
    original_index = standard_index[rows, columns]
    # Choose uniformly among residues with a different elemental composition. In particular,
    # this excludes I<->L: a zero-delta token swap would carry no compensating modification.
    replacement_priorities = torch.rand(
        (rows.numel(), len(_AMINO_ACIDS)), device=device
    )
    same_composition = (
        compositions.unsqueeze(0) == compositions[original_index].unsqueeze(1)
    ).all(dim=2)
    replacement_index = replacement_priorities.masked_fill(same_composition, 2.0).argmin(dim=1)

    tokens = batch.tokens.clone()
    mod_comp = batch.mod_comp.clone()
    mod_mass = batch.mod_mass.clone()
    mod_present = batch.mod_present.clone()
    mod_named = batch.mod_named.clone()

    tokens[rows, columns] = standard_tokens[replacement_index]
    composition_delta = compositions[original_index] - compositions[replacement_index]
    mass_delta = masses[original_index] - masses[replacement_index]
    mod_comp[rows, columns] += composition_delta
    mod_mass[rows, columns] += mass_delta
    mod_present[rows, columns] = True
    mod_named[rows, columns] = True

    return Batch(
        tokens=tokens,
        mod_comp=mod_comp,
        mod_mass=mod_mass,
        mod_present=mod_present,
        mod_named=mod_named,
        charge=batch.charge,
        lengths=batch.lengths,
        pad_mask=batch.pad_mask,
        frag_mask=batch.frag_mask,
    )


__all__ = ["substitute_residues"]
