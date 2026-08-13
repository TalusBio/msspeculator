"""FASTA digestion, precursor enumeration, deterministic splitting.

Tensor encoding is deliberately *not* re-exported here. ``encode`` imports torch, so re-exporting
it made importing anything from this package -- ``config``, ``digest``, ``split`` -- pull the whole
torch stack in with it. That put torch on the preparation ETL's import path, which never touches a
tensor. Import :mod:`pepdistill.data.encode` directly, as every caller already does.
"""

from .config import DigestConfig, SplitConfig
from .digest import cleave_protein, digest_fasta, digest_records, parse_fasta
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
]
