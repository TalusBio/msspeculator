"""Online distillation: label batches with the teacher live, in the training loop.

No cache — every batch is fresh sequences the teacher scores on the fly. Throughput is
capped by the teacher (fine for an overnight run). A curriculum warms up on random
peptides, then switches to real FASTA digests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..data.config import DigestConfig
from ..data.sources import (
    fasta_peptide_stream,
    precursors_from_sequences,
    random_peptide_stream,
)
from ..teacher.base import Teacher
from .dataset import DistillDataset, LabeledBatch, collate_with_labels


def _take(seq_iter: Iterator[str], n: int) -> list[str]:
    return [next(seq_iter) for _ in range(n)]


def curriculum_batches(
    teacher: Teacher,
    cfg: DigestConfig,
    rng: np.random.Generator,
    batch_size: int,
    total_batches: int,
    warmup_batches: int,
    fasta: str | Path | None,
) -> Iterator[LabeledBatch]:
    """Yield teacher-labeled batches: random peptides for warmup, then FASTA digests."""
    rand_seq = random_peptide_stream(rng, cfg.min_length, cfg.max_length)
    if fasta is not None:
        fasta_seq: Iterator[str] = fasta_peptide_stream(fasta, cfg, rng, loop=True)
    else:
        fasta_seq = rand_seq  # random-only curriculum

    for step in range(total_batches):
        src = rand_seq if step < warmup_batches else fasta_seq
        seqs = _take(src, batch_size)
        precs = precursors_from_sequences(seqs, cfg, rng)
        labels = teacher.predict(precs)
        yield collate_with_labels(precs, labels)


def estimate_norm(
    teacher: Teacher, cfg: DigestConfig, rng: np.random.Generator, n: int = 2000
) -> tuple[float, float, float, float]:
    """Estimate rt/ccs mean+std from a random sample so the student can standardize."""
    seqs = _take(random_peptide_stream(rng, cfg.min_length, cfg.max_length), n)
    precs = precursors_from_sequences(seqs, cfg, rng)
    labels = teacher.predict(precs)
    return DistillDataset(precs, labels).rt_ccs_stats()


def build_val_set(
    teacher: Teacher,
    cfg: DigestConfig,
    rng: np.random.Generator,
    n: int,
    fasta: str | Path | None,
) -> DistillDataset:
    """A fixed held-out set for periodic eval (from FASTA if given, else random)."""
    src = (
        fasta_peptide_stream(fasta, cfg, rng, loop=True)
        if fasta is not None
        else random_peptide_stream(rng, cfg.min_length, cfg.max_length)
    )
    seqs = _take(src, n)
    precs = precursors_from_sequences(seqs, cfg, rng)
    return DistillDataset(precs, teacher.predict(precs))
