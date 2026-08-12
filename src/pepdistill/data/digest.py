"""FASTA parsing and in-silico enzymatic digestion."""

from __future__ import annotations

import os
import re
import time
import urllib.request
from collections.abc import Callable, Iterator, Iterable
from pathlib import Path

from .config import DigestConfig, Enzyme

_VALID_AA = set("GASPVTCLINDQKEMHFRYW")
_UNIPROT_PROTEOME = re.compile(r"UP\d{9}")


def resolve_fasta(
    source: str | Path,
    *,
    cache_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Resolve a local path or ``uniprot:UP...`` proteome into a local FASTA.

    UniProt references are downloaded atomically once and then reused. ``PEPDISTILL_CACHE_DIR``
    overrides the cache root; otherwise the XDG cache directory (or ``~/.cache``) is used.
    """
    value = str(source)
    if not value.startswith("uniprot:"):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"FASTA does not exist: {path}")
        return path

    accession = value.removeprefix("uniprot:")
    if not _UNIPROT_PROTEOME.fullmatch(accession):
        raise ValueError(
            f"invalid UniProt proteome reference {value!r}; expected uniprot:UP followed by "
            "nine digits"
        )
    if cache_dir is None:
        configured = os.environ.get("PEPDISTILL_CACHE_DIR")
        if configured:
            root = Path(configured)
        else:
            root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pepdistill"
    else:
        root = Path(cache_dir)
    target = root / "fasta" / f"{accession}.fasta"
    if target.is_file() and target.stat().st_size > 0:
        if log is not None:
            log(f"[fasta] using cached {accession}: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".fasta.part")
    url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=proteome%3A{accession}"
    if log is not None:
        log(f"[fasta] downloading {accession} from UniProt -> {target}")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"UniProt returned an empty FASTA for {accession}")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if log is not None:
        log(
            f"[fasta] cached {accession}: {target} "
            f"({target.stat().st_size / (1024 * 1024):.1f} MiB in "
            f"{time.perf_counter() - started:.1f}s)"
        )
    return target


def parse_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` pairs from a FASTA file."""
    header: str | None = None
    chunks: list[str] = []
    with resolve_fasta(path).open() as fh:
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


def _check_length_window_reachable(cfg: DigestConfig) -> None:
    """Refuse a digest whose length window no peptide can enter.

    With ``unspecific`` every residue is a cleavage site, so a peptide spans at most
    ``missed_cleavages + 1`` residues. Pair that with the default ``min_length`` and the digest
    yields nothing — silently, because an empty result looks exactly like a proteome with no
    matching peptides. Observed: ``DigestConfig(enzyme="unspecific")`` returned 0 peptides from
    a 4,403-protein proteome.
    """
    enzyme = cfg.enzyme_rule()
    if len(enzyme.cleave_after) < 20:
        return  # a specific enzyme's peptide length is set by site spacing, not the MC count
    max_span = cfg.missed_cleavages + 1
    if max_span < cfg.min_length:
        raise ValueError(
            f"enzyme {cfg.enzyme!r} cleaves after every residue, so a peptide spans at most "
            f"missed_cleavages + 1 = {max_span} residues — but min_length is {cfg.min_length}, "
            f"so no peptide can qualify. Raise missed_cleavages to at least "
            f"{cfg.min_length - 1}, or lower min_length."
        )


def digest_records(records: Iterable[tuple[str, str]], cfg: DigestConfig) -> list[str]:
    _check_length_window_reachable(cfg)
    peptides: set[str] = set()
    n_records = 0
    for _header, seq in records:
        n_records += 1
        peptides.update(cleave_protein(seq, cfg))
    if n_records and not peptides:
        raise ValueError(
            f"digest produced 0 peptides from {n_records} protein(s) with enzyme "
            f"{cfg.enzyme!r}, missed_cleavages={cfg.missed_cleavages}, length window "
            f"[{cfg.min_length}, {cfg.max_length}]. An empty digest is a configuration error, "
            "not a result — check the length window against what this enzyme can produce."
        )
    return sorted(peptides)
