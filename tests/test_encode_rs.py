"""Parity: Rust collate must match the pure-Python reference exactly, both termini modes.

Skipped unless the ``pepdistill_rs`` crate is built (``maturin develop`` in ``rust/``).
This is the contract guard — if the vocab constants drift between Python and Rust, or the
mask/offset logic diverges, these asserts fail.
"""

import pytest
import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate, set_termini, use_termini
from pepdistill.data.encode_rs import HAVE_RS
from pepdistill.data.precursors import Precursor

pytestmark = pytest.mark.skipif(not HAVE_RS, reason="pepdistill_rs not built")


def _precs():
    return [
        Precursor(Peptide("SAMPLER"), 2, "train"),
        Precursor(
            Peptide("ACDEMKPEPTIDE", ((1, "Carbamidomethyl@C"), (4, "Oxidation@M"))), 3, "train"
        ),
        Precursor(Peptide("MKLV", ((0, "Oxidation@M"),)), 1, "train"),
        Precursor(Peptide("WYFGHIKLMNPQRSTVWYAC"), 4, "train"),
    ]


@pytest.mark.parametrize("termini", [False, True])
def test_collate_parity(termini):
    from pepdistill.data.encode_rs import collate_rs

    prev = use_termini()
    set_termini(termini)
    try:
        ref = collate(_precs())
        got = collate_rs(_precs())
        for field in ("tokens", "mod_delta", "charge", "lengths", "pad_mask", "frag_mask"):
            r = getattr(ref, field)
            g = getattr(got, field)
            assert r.shape == g.shape, f"{field} shape {tuple(r.shape)} != {tuple(g.shape)}"
            assert r.dtype == g.dtype, f"{field} dtype {r.dtype} != {g.dtype}"
            if r.dtype == torch.float32:
                assert torch.allclose(r, g, atol=1e-6), field
            else:
                assert torch.equal(r, g), field
    finally:
        set_termini(prev)
