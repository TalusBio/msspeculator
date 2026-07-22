"""Val reduction: one best-quality example per (dataset, modified_sequence, charge)."""

import numpy as np
import pytest

from pepdistill.chem import Peptide
from pepdistill.data.precursors import Precursor
from pepdistill.eval import best_examples, best_per_key, precursor_key
from pepdistill.teacher.base import PrecursorLabels


def _lab(total):
    ms2 = np.zeros((3, 4), dtype=np.float32)
    ms2[0, 0] = total
    return PrecursorLabels(ms2=ms2, rt=10.0, ccs=float("nan"))


def test_best_per_key_keeps_argmax():
    keys = ["a", "b", "a", "b", "a"]
    q = [0.1, 0.5, 0.9, 0.2, 0.3]
    idx = best_per_key(q, keys)
    assert idx == [2, 1]  # a -> index 2 (0.9), b -> index 1 (0.5); first-seen key order


def test_precursor_key_splits_charge_and_dataset():
    p2 = Precursor(Peptide("PEPTIDEK"), 2, "val")
    p3 = Precursor(Peptide("PEPTIDEK"), 3, "val")
    assert precursor_key(p2) != precursor_key(p3)  # charge distinguishes
    assert precursor_key(p2, "poolA") != precursor_key(p2, "poolB")


def test_best_examples_dedups_and_slices_extras():
    # Same precursor observed 3x with different intensities; a second distinct precursor.
    pep = Peptide("PEPTIDEK")
    precs = [Precursor(pep, 2, "val")] * 3 + [Precursor(Peptide("ACDEFGHK"), 2, "val")]
    labels = [_lab(0.2), _lab(0.9), _lab(0.5), _lab(0.4)]
    raw_rt = [11.0, 12.0, 13.0, 20.0]
    src = [0, 1, 0, 1]

    precs2, labels2, rrt2, src2 = best_examples(precs, labels, raw_rt, src)
    assert len(precs2) == 2  # two distinct library entries
    # PEPTIDEK kept its best observation (intensity 0.9 -> raw_rt 12.0, src 1).
    peptidek = next(i for i, p in enumerate(precs2) if p.peptide.sequence == "PEPTIDEK")
    assert rrt2[peptidek] == 12.0
    assert src2[peptidek] == 1
    assert float(np.nansum(labels2[peptidek].ms2)) == pytest.approx(0.9)


def test_best_examples_per_dataset_keeps_both():
    pep = Peptide("PEPTIDEK")
    precs = [Precursor(pep, 2, "val"), Precursor(pep, 2, "val")]
    labels = [_lab(0.3), _lab(0.7)]
    ds = ["poolA", "poolB"]
    precs2, _ = best_examples(precs, labels, dataset=ds)
    assert len(precs2) == 2  # same precursor, different dataset -> both survive
