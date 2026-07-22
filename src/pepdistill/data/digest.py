"""FASTA parsing and in-silico enzymatic digestion."""

from __future__ import annotations

from collections.abc import Iterator, Iterable
from pathlib import Path

from .config import DigestConfig, Enzyme

_VALID_AA = set("GASPVTCLINDQKEMHFRYW")


def parse_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` pairs from a FASTA file."""
    header: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip().upper())
    if header is not None:
        yield header, "".join(chunks)


def _cleavage_sites(protein: str, enzyme: Enzyme) -> list[int]:
    """Indices (0-based, after this residue) where the enzyme cuts."""
    sites = [0]
    for i, aa in enumerate(protein):
        if aa in enzyme.cleave_after:
            nxt = protein[i + 1] if i + 1 < len(protein) else ""
            if nxt in enzyme.restrict_before:
                continue
            sites.append(i + 1)
    if sites[-1] != len(protein):
        sites.append(len(protein))
    return sites


def cleave_protein(protein: str, cfg: DigestConfig) -> Iterator[str]:
    """Peptides from one protein sequence honoring missed cleavages + length."""
    enzyme = cfg.enzyme_rule()
    sites = _cleavage_sites(protein, enzyme)
    n = len(sites) - 1
    for start_idx in range(n):
        for mc in range(cfg.missed_cleavages + 1):
            end_idx = start_idx + mc + 1
            if end_idx > n:
                break
            pep = protein[sites[start_idx] : sites[end_idx]]
            if cfg.min_length <= len(pep) <= cfg.max_length and _VALID_AA.issuperset(pep):
                yield pep


def digest_fasta(path: str | Path, cfg: DigestConfig) -> list[str]:
    """All unique peptides from a FASTA, sorted for determinism."""
    return digest_records(parse_fasta(path), cfg)


def digest_records(records: Iterable[tuple[str, str]], cfg: DigestConfig) -> list[str]:
    peptides: set[str] = set()
    for _header, seq in records:
        peptides.update(cleave_protein(seq, cfg))
    return sorted(peptides)
