"""FASTA digestion, precursor enumeration, deterministic splitting, tensor encoding."""

from .config import DigestConfig, SplitConfig
from .digest import cleave_protein, digest_fasta, digest_records, parse_fasta
from .encode import Batch, collate
from .precursors import (
    Precursor,
    enumerate_precursors,
    frame_to_precursors,
    precursors_to_frame,
)
from .split import assign_split

__all__ = [
    "DigestConfig",
    "SplitConfig",
    "parse_fasta",
    "cleave_protein",
    "digest_fasta",
    "digest_records",
    "enumerate_precursors",
    "precursors_to_frame",
    "frame_to_precursors",
    "Precursor",
    "assign_split",
    "collate",
    "Batch",
]
