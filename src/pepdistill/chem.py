"""Monoisotopic mass and m/z arithmetic for peptides and fragment ions.

Pure numeric code, no torch. Everything here is deterministic and unit-testable.
Modifications are represented as ``(zero_based_site, delta_mass)`` pairs applied on
top of the bare residue masses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PROTON = 1.007_276_466_879
H2O = 18.010_564_684_25

# Monoisotopic residue masses (Da), i.e. amino acid minus one water.
RESIDUE_MASS: dict[str, float] = {
    "G": 57.021_463_723,
    "A": 71.037_113_787,
    "S": 87.032_028_409,
    "P": 97.052_763_851,
    "V": 99.068_413_915,
    "T": 101.047_678_473,
    "C": 103.009_184_477,
    "L": 113.084_064_015,
    "I": 113.084_064_015,
    "N": 114.042_927_446,
    "D": 115.026_943_031,
    "Q": 128.058_577_540,
    "K": 128.094_963_016,
    "E": 129.042_593_095,
    "M": 131.040_484_605,
    "H": 137.058_911_861,
    "F": 147.068_413_915,
    "R": 156.101_111_050,
    "Y": 163.063_328_575,
    "W": 186.079_312_952,
    # U/O/X/B/Z not supported; digestion filters them out.
}

# Named modification deltas (Da). Extend as needed.
# Our own modification table (name -> monoisotopic delta). This is pepdistill's chemistry
# representation, independent of how any external source (peptdeep, PROSPECT/UNIMOD) encodes
# mods — adapters translate INTO these names. The student uses a single mod-mass channel, so
# adding a curated mod here is all it takes to represent it.
MOD_DELTA: dict[str, float] = {
    "Carbamidomethyl@C": 57.021_463_723,
    "Oxidation@M": 15.994_914_622,
    "Phospho": 79.966_331_2,
    "TMT6plex": 229.162_932_1,
}


@dataclass(frozen=True, slots=True)
class Peptide:
    """A modified peptide.

    ``mods`` maps zero-based residue index -> modification name. Names must exist in
    :data:`MOD_DELTA`. The object is hashable so it can key label caches.
    """

    sequence: str
    mods: tuple[tuple[int, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Normalize mods to a sorted tuple for canonical hashing/equality.
        object.__setattr__(self, "mods", tuple(sorted(self.mods)))

    @property
    def length(self) -> int:
        return len(self.sequence)

    def residue_masses(self) -> list[float]:
        masses = [RESIDUE_MASS[a] for a in self.sequence]
        for site, name in self.mods:
            masses[site] += MOD_DELTA[name]
        return masses

    def mono_mass(self) -> float:
        return sum(self.residue_masses()) + H2O

    def precursor_mz(self, charge: int) -> float:
        return (self.mono_mass() + charge * PROTON) / charge

    def modified_sequence(self) -> str:
        """Human/engine-readable form, e.g. ``AC[Carbamidomethyl@C]DEM[Oxidation@M]K``."""
        by_site: dict[int, str] = {s: n for s, n in self.mods}
        out: list[str] = []
        for i, aa in enumerate(self.sequence):
            out.append(aa)
            if i in by_site:
                out.append(f"[{by_site[i]}]")
        return "".join(out)


# Fragment ion types supported by the student's MS2 head. Order is the tensor column
# order; keep stable — checkpoints depend on it.
ION_TYPES: tuple[tuple[str, int], ...] = (
    ("b", 1),
    ("y", 1),
    ("b", 2),
    ("y", 2),
)


def fragment_mz(residue_masses: list[float], ion: str, ordinal: int, charge: int) -> float:
    """m/z of a single fragment ion.

    ``ordinal`` is 1-based (b1/y1 = first fragment). ``residue_masses`` is the
    per-position mass list from :meth:`Peptide.residue_masses`.
    """
    n = len(residue_masses)
    if ion == "b":
        neutral = sum(residue_masses[:ordinal])
    elif ion == "y":
        neutral = sum(residue_masses[n - ordinal :]) + H2O
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown ion type {ion!r}")
    return (neutral + charge * PROTON) / charge


def fragment_mz_matrix(pep: Peptide) -> list[list[float]]:
    """Full (n_positions, n_ion_types) m/z matrix for a peptide.

    Row ``i`` (0-based) is fragmentation site after residue ``i+1``: b-ions use
    ordinal ``i+1``, y-ions use ordinal ``L-1-i``. Matches :func:`ms2_target_shape`.
    """
    rm = pep.residue_masses()
    n = pep.length
    matrix: list[list[float]] = []
    for i in range(n - 1):
        row: list[float] = []
        for ion, z in ION_TYPES:
            ordinal = (i + 1) if ion == "b" else (n - 1 - i)
            row.append(fragment_mz(rm, ion, ordinal, z))
        matrix.append(row)
    return matrix


def ms2_target_shape(length: int) -> tuple[int, int]:
    """(n_fragment_positions, n_ion_types) for a peptide of the given length."""
    return (length - 1, len(ION_TYPES))
