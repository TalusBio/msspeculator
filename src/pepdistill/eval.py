"""Validation-set reduction for reporting.

Real spectral libraries observe the same precursor many times (repeated scans, multiple
runs). Averaging a val metric over every observation over-weights abundant peptides, so the
number tracks sampling depth rather than model quality. We instead report on ONE
representative per precursor per dataset — the best-quality observation — so every library
entry counts once.

"Best" defaults to the highest total MS2 intensity (the most fully sampled spectrum). The
grouping key is (dataset, modified_sequence, charge): a mod-form at charge 2 and charge 3 are
distinct library entries and both survive. This is deliberately coarse — finer generalization
folds (leave-one-run-out, held-out modifications) need a different split key, not this.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np

from .data.precursors import Precursor
from .teacher.base import PrecursorLabels


def precursor_key(p: Precursor, dataset: Hashable = None) -> tuple:
    """Library-entry identity: (dataset, modified_sequence, charge)."""
    return (dataset, p.peptide.modified_sequence(), p.charge)


def ms2_intensity(label: PrecursorLabels) -> float:
    """Default quality score: total fragment intensity (most fully sampled spectrum)."""
    return float(np.nansum(label.ms2))


def best_per_key(quality: Sequence[float], keys: Sequence[Hashable]) -> list[int]:
    """Indices keeping the single argmax-quality item per key, in first-seen key order."""
    best: dict[Hashable, int] = {}
    for i, (q, k) in enumerate(zip(quality, keys)):
        j = best.get(k)
        if j is None or q > quality[j]:
            best[k] = i
    return list(best.values())


def best_examples(
    precursors: Sequence[Precursor],
    labels: Sequence[PrecursorLabels],
    *extra: Sequence[Any],
    dataset: Hashable | Sequence[Hashable] = None,
    quality: Callable[[PrecursorLabels], float] = ms2_intensity,
) -> tuple[list, ...]:
    """Keep the best example per (dataset, modified_sequence, charge).

    ``dataset`` is one label for all examples, or a per-example sequence. Any ``extra``
    parallel sequences (raw_rt, source ids, ...) are sliced with the same indices, so the
    return is ``(precursors, labels, *extra)`` reduced consistently.
    """
    ds = dataset if isinstance(dataset, Sequence) and not isinstance(dataset, str) else [dataset] * len(precursors)
    keys = [precursor_key(p, d) for p, d in zip(precursors, ds)]
    q = [quality(lab) for lab in labels]
    idx = best_per_key(q, keys)
    cols = (precursors, labels, *extra)
    return tuple([col[i] for i in idx] for col in cols)
