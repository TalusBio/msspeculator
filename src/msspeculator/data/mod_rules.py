"""Modification rules: which residues a modification applies to, and how often.

A rule is written in the same ProForma grammar a peptide uses, with a residue set in front of the
modification, so a rule cannot express something a peptide could not carry:

    "C[UNIMOD:4]"       carbamidomethyl on every cysteine
    "STY[UNIMOD:21]"    phospho on serine, threonine or tyrosine
    "[UNIMOD:737]-"     TMT on the peptide N-terminus
    "-[UNIMOD:21]"      on the C-terminus

Fixed rules apply to every matching site. Variable rules carry a per-site probability, so
``{"STY[UNIMOD:21]": 0.001}`` means each S, T and Y independently takes a phosphate one time in a
thousand. That is a sampling knob for the pretrain stream, where a peptide is seen once per pass
and the point is to cover modforms at a realistic rate. Exhaustive enumeration for library
generation ignores the probability and takes every combination up to the per-peptide cap, because
a library that omitted a modform would simply fail to identify it.

Parsing and validation are the Rust grammar's, so an unknown accession or an unparseable rule fails
here rather than at first use, and there is no second vocabulary to keep in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

import msspeculator_rs as _rs


@dataclass(frozen=True, slots=True)
class ModRule:
    """One parsed rule: the sites it targets and the modification it places there."""

    #: Residue letters, or ``"n"``/``"c"`` for a terminus.
    targets: tuple[str, ...]
    #: Canonical modification identity, e.g. ``"UNIMOD:21"``.
    spec: str

    @property
    def is_terminal(self) -> bool:
        return self.targets in (("n",), ("c",))


def parse_rule(rule: str) -> ModRule:
    """Parse one rule, raising on anything the grammar or the UNIMOD table rejects.

    >>> parse_rule("STY[UNIMOD:21]")
    ModRule(targets=('S', 'T', 'Y'), spec='UNIMOD:21')
    >>> parse_rule("[UNIMOD:737]-")
    ModRule(targets=('n',), spec='UNIMOD:737')
    >>> parse_rule("STY[Phospho]")
    Traceback (most recent call last):
    ValueError: invalid modification rule "STY[Phospho]"...
    """
    targets, spec = _rs.parse_modification_rule(rule)
    return ModRule(targets=tuple(targets), spec=spec)


def rule_sites(sequence: str, rule: ModRule) -> list[int | str]:
    """Every site in ``sequence`` that ``rule`` targets.

    >>> rule_sites("PEPSTIDEK", parse_rule("STY[UNIMOD:21]"))
    [3, 4]
    >>> rule_sites("PEPSTIDEK", parse_rule("[UNIMOD:737]-"))
    ['n']
    """
    if rule.is_terminal:
        return [rule.targets[0]]
    wanted = set(rule.targets)
    return [index for index, residue in enumerate(sequence) if residue in wanted]


def fixed_sites(sequence: str, rules: tuple[ModRule, ...]) -> list[tuple[int | str, str]]:
    """Every ``(site, spec)`` a fixed rule places on ``sequence``.

    >>> fixed_sites("ACDEMCK", (parse_rule("C[UNIMOD:4]"),))
    [(1, 'UNIMOD:4'), (5, 'UNIMOD:4')]
    """
    return [(site, rule.spec) for rule in rules for site in rule_sites(sequence, rule)]


def sampled_sites(
    sequence: str,
    rules: tuple[tuple[ModRule, float], ...],
    rng: np.random.Generator,
    max_mods: int,
) -> list[tuple[int | str, str]]:
    """Draw variable modifications, each candidate site independently at its rule's probability.

    Independent per site rather than per peptide, so a peptide with ten serines is ten times as
    likely to carry a phosphate as one with a single serine; which is what a per-amino-acid rate
    means. Draws are then truncated to ``max_mods``; truncation keeps a random subset rather than
    the first few, since candidate sites are generated in sequence order and keeping the first
    would bias every modification towards the N-terminus.
    """
    if max_mods <= 0:
        return []
    drawn: list[tuple[int | str, str]] = []
    for rule, probability in rules:
        sites = rule_sites(sequence, rule)
        if not sites:
            continue
        hits = rng.random(len(sites)) < probability
        drawn.extend((site, rule.spec) for site, hit in zip(sites, hits) if hit)
    if len(drawn) <= max_mods:
        return drawn
    keep = rng.choice(len(drawn), size=max_mods, replace=False)
    return [drawn[index] for index in sorted(keep)]


def enumerate_modforms(
    sequence: str, rules: tuple[ModRule, ...], max_mods: int
) -> list[list[tuple[int | str, str]]]:
    """Every combination of variable modifications up to ``max_mods``, including the bare form.

    Probabilities are deliberately absent: this is the library-generation path, where omitting a
    modform means failing to identify it.

    >>> forms = enumerate_modforms("MPEM", (parse_rule("M[UNIMOD:35]"),), 1)
    >>> forms
    [[], [(0, 'UNIMOD:35')], [(3, 'UNIMOD:35')]]
    """
    candidates = [(site, rule.spec) for rule in rules for site in rule_sites(sequence, rule)]
    forms: list[list[tuple[int | str, str]]] = [[]]
    for count in range(1, min(max_mods, len(candidates)) + 1):
        forms.extend(list(chosen) for chosen in combinations(candidates, count))
    return forms


__all__ = [
    "ModRule",
    "parse_rule",
    "rule_sites",
    "fixed_sites",
    "sampled_sites",
    "enumerate_modforms",
]
