"""Configuration dataclasses for the FASTA -> precursor data generator."""

from __future__ import annotations

from dataclasses import dataclass


# Enzyme cleavage rules: cut C-terminal to any residue in ``cleave_after``, unless the
# next residue is in ``restrict_before``. Extend the registry to add enzymes.
@dataclass(frozen=True, slots=True)
class Enzyme:
    name: str
    cleave_after: frozenset[str]
    restrict_before: frozenset[str] = frozenset()


ENZYMES: dict[str, Enzyme] = {
    "trypsin": Enzyme("trypsin", frozenset("KR"), frozenset("P")),
    "trypsin/p": Enzyme("trypsin/p", frozenset("KR")),
    "lysc": Enzyme("lysc", frozenset("K")),
    # Non-specific ("unspecific") digestion: a cut is allowed after every residue, so
    # cleave_protein enumerates all substrings within the length window. Use with a tight
    # min/max length (e.g. 8-11 for HLA class I) and large missed_cleavages, since with
    # every position a site the missed-cleavage count is what actually bounds peptide span.
    "unspecific": Enzyme("unspecific", frozenset("GASPVTCLINDQKEMHFRYW")),
}


@dataclass(frozen=True, slots=True)
class DigestConfig:
    """Everything controlling which precursors come out of a FASTA."""

    enzyme: str = "trypsin"
    missed_cleavages: int = 2
    min_length: int = 7
    max_length: int = 30
    min_charge: int = 2
    max_charge: int = 4
    # Fixed mods always applied to every matching residue.
    fixed_mods: tuple[str, ...] = ("Carbamidomethyl@C",)
    # Variable mods enumerated as mod-forms up to ``max_variable_mods`` per peptide.
    variable_mods: tuple[str, ...] = ("Oxidation@M",)
    max_variable_mods: int = 1

    def enzyme_rule(self) -> Enzyme:
        try:
            return ENZYMES[self.enzyme]
        except KeyError as exc:
            raise ValueError(f"unknown enzyme {self.enzyme!r}; known: {sorted(ENZYMES)}") from exc

    @property
    def charges(self) -> range:
        return range(self.min_charge, self.max_charge + 1)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Deterministic hash split. Fractions must sum to 1.0."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    salt: str = "pepdistill-v1"

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
