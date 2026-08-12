"""Canonical ProForma rendering for peptide labels and exported diagnostics."""

from __future__ import annotations

from .chem import Peptide, unimod_accession


def _mod_tag(spec: str | float) -> str:
    if isinstance(spec, str):
        accession = unimod_accession(spec)
        if accession is None:
            raise ValueError(f"named modification {spec!r} has no vendored UNIMOD accession")
        return f"[UNIMOD:{accession}]"
    return f"[{float(spec):+.9g}]"


def proforma_sequence(peptide: Peptide) -> str:
    """Render a model peptide using CV accessions and ProForma terminal placement."""
    n_terminal: list[str] = []
    c_terminal: list[str] = []
    by_residue: dict[int, list[str]] = {}
    for site, spec in peptide.mods:
        tag = _mod_tag(spec)
        if site == "n":
            n_terminal.append(tag)
        elif site == "c":
            c_terminal.append(tag)
        else:
            by_residue.setdefault(int(site), []).append(tag)

    body = "".join(
        residue + "".join(by_residue.get(index, ()))
        for index, residue in enumerate(peptide.sequence)
    )
    prefix = "".join(n_terminal) + ("-" if n_terminal else "")
    suffix = ("-" if c_terminal else "") + "".join(c_terminal)
    return prefix + body + suffix


__all__ = ["proforma_sequence"]
