"""Chemistry is single-sourced in Rust (pepdistill_rs / rust/core). This module only
re-exports — it defines no constants or logic (so nothing here can drift)."""

from pepdistill_rs import (  # noqa: F401
    H2O,
    ION_TYPES,
    MOD_DELTA,
    PROTON,
    RESIDUE_COMP,
    RESIDUE_MASS,
    Peptide,
    fragment_mz,
    fragment_mz_matrix,
    ms2_target_shape,
    unimod_accession,
    unimod_name,
)

__all__ = [
    "H2O",
    "ION_TYPES",
    "MOD_DELTA",
    "PROTON",
    "RESIDUE_COMP",
    "RESIDUE_MASS",
    "Peptide",
    "fragment_mz",
    "fragment_mz_matrix",
    "ms2_target_shape",
    "unimod_accession",
    "unimod_name",
]
