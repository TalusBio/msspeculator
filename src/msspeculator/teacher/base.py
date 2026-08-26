"""Teacher interface + label containers shared by every teacher implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..chem import ION_TYPES
from ..data.precursors import Precursor

ION_COLUMNS: tuple[str, ...] = tuple(f"{ion}_z{z}" for ion, z in ION_TYPES)


@dataclass(slots=True)
class PrecursorLabels:
    """Teacher soft labels for one precursor.

    ``ms2`` has shape ``(length - 1, len(ION_TYPES))``. Teacher labels are normalized so the
    max is 1.0. Real PROSPECT labels are base-peak normalized per spectrum *before* the
    b/y + charge<=2 + no-neutral-loss filter, which drops the base peak for about a third of
    spectra, so their max is often below 1.0. Both MS2 losses (``ms2_cosine_loss``,
    ``spectral_angle``) are scale-invariant, so this difference does not reach training,
    but do not compare magnitudes across the two sources.
    ``rt`` and ``ccs`` are scalars in the teacher's native units.
    """

    ms2: np.ndarray
    rt: float
    ccs: float


class Teacher(ABC):
    """Produces soft labels for a list of precursors."""

    name: str = "teacher"

    @abstractmethod
    def predict(self, precursors: list[Precursor], nces=None) -> list[PrecursorLabels]:
        """Label precursors. ``nces`` (optional) is a per-precursor collision energy override
        for a sweep; ``None`` uses the teacher's fixed NCE."""
        ...


def labels_to_frames(
    precursors: list[Precursor], labels: list[PrecursorLabels]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Serialize labels to (precursor-level, fragment-level) frames for caching."""
    if len(precursors) != len(labels):
        raise ValueError("precursors and labels length mismatch")

    prec_rows = []
    frag_rows = []
    for idx, lab in enumerate(labels):
        prec_rows.append({"prec_idx": idx, "rt": lab.rt, "ccs": lab.ccs})
        for pos in range(lab.ms2.shape[0]):
            row = {"prec_idx": idx, "position": pos}
            for col, val in zip(ION_COLUMNS, lab.ms2[pos]):
                row[col] = float(val)
            frag_rows.append(row)
    return pd.DataFrame(prec_rows), pd.DataFrame(frag_rows)


def labels_from_frames(prec_df: pd.DataFrame, frag_df: pd.DataFrame) -> list[PrecursorLabels]:
    """Inverse of :func:`labels_to_frames`."""
    ms2_by_prec: dict[int, list[np.ndarray]] = {}
    for row in frag_df.sort_values(["prec_idx", "position"]).itertuples(index=False):
        vec = np.array([getattr(row, c) for c in ION_COLUMNS], dtype=np.float32)
        ms2_by_prec.setdefault(int(row.prec_idx), []).append(vec)

    out: list[PrecursorLabels] = []
    for row in prec_df.sort_values("prec_idx").itertuples(index=False):
        idx = int(row.prec_idx)
        stack = ms2_by_prec.get(idx, [])
        ms2 = np.stack(stack) if stack else np.zeros((0, len(ION_COLUMNS)), dtype=np.float32)
        out.append(PrecursorLabels(ms2=ms2, rt=float(row.rt), ccs=float(row.ccs)))
    return out
