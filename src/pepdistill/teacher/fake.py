"""A deterministic, dependency-free teacher for tests and demos.

Emits smooth pseudo-labels derived from residue masses so a student can actually
learn the mapping, without needing peptdeep/torch weights installed.
"""

from __future__ import annotations

import numpy as np

from ..chem import ION_TYPES, Peptide
from ..data.precursors import Precursor
from .base import PrecursorLabels, Teacher

# Kyte-Doolittle hydropathy, used to fake an RT that correlates with sequence.
_HYDRO = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


class FakeTeacher(Teacher):
    name = "fake"

    def predict(self, precursors: list[Precursor]) -> list[PrecursorLabels]:
        return [self._one(p) for p in precursors]

    def _one(self, prec: Precursor) -> PrecursorLabels:
        pep: Peptide = prec.peptide
        rm = np.array(pep.residue_masses(), dtype=np.float64)
        n = pep.length

        ms2 = np.zeros((n - 1, len(ION_TYPES)), dtype=np.float32)
        for i in range(n - 1):
            b_mass = rm[: i + 1].sum()
            y_mass = rm[i + 1 :].sum()
            for j, (ion, z) in enumerate(ION_TYPES):
                base = b_mass if ion == "b" else y_mass
                # Deterministic, bounded, charge-dependent intensity.
                val = 0.5 * (1 + np.sin(base / (37.0 * z) + prec.charge))
                ms2[i, j] = val
        peak = ms2.max()
        if peak > 0:
            ms2 /= peak

        rt = float(sum(_HYDRO.get(a, 0.0) for a in pep.sequence) + 0.02 * pep.mono_mass())
        ccs = float(0.9 * pep.mono_mass() / prec.charge + 30.0 * prec.charge)
        return PrecursorLabels(ms2=ms2, rt=rt, ccs=ccs)
