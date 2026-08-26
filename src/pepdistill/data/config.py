"""Configuration dataclasses for the FASTA -> precursor data generator."""

from __future__ import annotations

from dataclasses import dataclass

from .mod_rules import ModRule, parse_rule


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
    # Fixed rules, applied to every matching site.
    fixed_mods: tuple[str, ...] = ("C[UNIMOD:4]",)
    # Variable rules with a per-site probability: each matching residue independently takes the
    # modification at that rate. The defaults are the canonical PTMs measured in PROSPECT, minus
    # TMT, which the teacher scores at 0.29-0.34 and so cannot usefully supervise.
    variable_mods: tuple[tuple[str, float], ...] = (
        ("M[UNIMOD:35]", 0.1),
        ("STY[UNIMOD:21]", 0.001),
        ("K[UNIMOD:1]", 0.001),
        ("K[UNIMOD:121]", 0.001),
    )
    max_variable_mods: int = 1

    def __post_init__(self) -> None:
        """Reject an unusable rule here, where the config is read, rather than deep in a run.

        The grammar answers both questions a rule has to: which sites it applies to (the residue
        set) and what it places there (the accession). Neither was checked on the pretrain path
        before, and neither failure was prompt; a site-agnostic name raised ``IndexError`` from a
        string split, and an unknown one built precursors quite happily, because ``Peptide``
        resolves mass lazily, then failed once the teacher asked for a mass well into a run.
        """
        for rule in self.fixed_mods:
            parse_rule(rule)
        for rule, probability in self.variable_mods:
            parse_rule(rule)
            if not 0.0 < probability <= 1.0:
                raise ValueError(
                    f"variable_mods rule {rule!r} has probability {probability!r}; it must be in "
                    "(0, 1]. Remove the rule rather than setting it to zero."
                )
        if self.max_variable_mods < 0:
            raise ValueError("max_variable_mods must not be negative")

    def fixed_rules(self) -> tuple[ModRule, ...]:
        """Parsed fixed rules. Parsed on demand, so callers do this once per batch, not per row."""
        return tuple(parse_rule(rule) for rule in self.fixed_mods)

    def variable_rules(self) -> tuple[tuple[ModRule, float], ...]:
        return tuple((parse_rule(rule), probability) for rule, probability in self.variable_mods)

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
