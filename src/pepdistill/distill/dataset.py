"""Pairs precursors with teacher labels and collates them into training batches."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..data.encode import Batch, collate, frag_offset
from ..data.precursors import Precursor
from ..teacher.base import PrecursorLabels


@dataclass(slots=True)
class MSFactors:
    """Per-example MS acquisition factors that feed MSContextEncoder. ``energy`` may be None
    (collision energy omitted -> that term contributes zero; nothing is fabricated)."""

    instrument_id: torch.Tensor
    detector_id: torch.Tensor
    fragmentation_id: torch.Tensor
    energy: torch.Tensor | None

    def to(self, device) -> "MSFactors":
        e = None if self.energy is None else self.energy.to(device)
        return MSFactors(
            self.instrument_id.to(device),
            self.detector_id.to(device),
            self.fragmentation_id.to(device),
            e,
        )


@dataclass(slots=True)
class LabeledBatch:
    inputs: Batch
    ms2_target: torch.Tensor  # (B, L-1, n_ion)
    rt_target: torch.Tensor  # (B,)
    ccs_target: torch.Tensor  # (B,)
    ms_factors: MSFactors | None = None

    def to(self, device: torch.device | str) -> "LabeledBatch":
        return LabeledBatch(
            self.inputs.to(device),
            self.ms2_target.to(device),
            self.rt_target.to(device),
            self.ccs_target.to(device),
            None if self.ms_factors is None else self.ms_factors.to(device),
        )


def iter_batch_indices(
    n: int, batch_size: int, shuffle: bool, generator: torch.Generator
) -> Iterator[list[int]]:
    """Yield index chunks over ``range(n)`` — shuffled (via ``generator``) or in order. The one
    place the epoch's example order and chunking live, shared by every in-memory dataset."""
    order = torch.randperm(n, generator=generator).tolist() if shuffle else list(range(n))
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


class DistillDataset:
    """In-memory list of (precursor, labels). Iterate with :meth:`batches`."""

    def __init__(self, precursors: list[Precursor], labels: list[PrecursorLabels]) -> None:
        if len(precursors) != len(labels):
            raise ValueError("precursors and labels must be the same length")
        self.precursors = precursors
        self.labels = labels

    def __len__(self) -> int:
        return len(self.precursors)

    def rt_ccs_stats(self) -> tuple[float, float, float, float]:
        rt = np.array([lab.rt for lab in self.labels], dtype=np.float64)
        ccs = np.array([lab.ccs for lab in self.labels], dtype=np.float64)
        return (
            float(rt.mean()),
            float(rt.std() or 1.0),
            float(ccs.mean()),
            float(ccs.std() or 1.0),
        )

    def batches(self, batch_size: int, shuffle: bool, generator: torch.Generator):
        for idx in iter_batch_indices(len(self), batch_size, shuffle, generator):
            yield collate_with_labels(
                [self.precursors[i] for i in idx], [self.labels[i] for i in idx]
            )


class BatchIterable(IterableDataset):
    """Wrap any object exposing ``batches(batch_size, shuffle, generator)`` as an
    ``IterableDataset``. ``__iter__`` runs once per epoch and re-seeds the generator, so a
    ``DataLoader(batch_size=None)`` reshuffles each epoch while passing each ready-made batch
    straight through — batching/collation already happened inside ``batches``.
    """

    def __init__(self, source, batch_size: int, shuffle: bool, seed: int) -> None:
        self.source = source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        yield from self.source.batches(self.batch_size, self.shuffle, gen)


def collate_with_labels(precursors: list[Precursor], labels: list[PrecursorLabels]) -> LabeledBatch:
    inputs: Batch = collate(precursors)
    b, frag_len = inputs.frag_mask.shape
    n_ion = labels[0].ms2.shape[1] if labels else 0

    off = frag_offset()
    ms2 = torch.zeros(b, frag_len, n_ion, dtype=torch.float32)
    for i, lab in enumerate(labels):
        k = lab.ms2.shape[0]  # = residues - 1
        # Place the k label rows at the fragment-site indices matching Batch.frag_mask.
        ms2[i, off : off + k] = torch.from_numpy(lab.ms2.astype(np.float32))

    rt = torch.tensor([lab.rt for lab in labels], dtype=torch.float32)
    ccs = torch.tensor([lab.ccs for lab in labels], dtype=torch.float32)
    return LabeledBatch(inputs, ms2, rt, ccs)
