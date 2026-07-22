"""Acquisition-context conditioning: no-op init, head isolation, param-efficient fine-tune."""

import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.models.context import ContextBook
from pepdistill.models.registry import build_student
from pepdistill.models.student import StudentConfig, StudentModel


def _batch(seqs=("SAMPLER", "PEPTIDEK", "ACDEFGHIK")):
    return collate([Precursor(Peptide(s), 2 + i % 3, "t") for i, s in enumerate(seqs)])


def test_zero_context_is_base_model():
    """ctx=None and ctx=0 must both reproduce the base prediction exactly (zero bias init)."""
    m = build_student("small").eval()
    b = _batch()
    z = torch.zeros(3, m.cfg.context_dim)
    with torch.no_grad():
        base = m(b)
        zeroed = m(b, ctx_acq=z, ctx_lc=z)
    for k in ("ms2", "rt", "ccs"):
        assert torch.allclose(base[k], zeroed[k], atol=1e-6), k


def test_context_head_isolation():
    """ctx_lc moves RT only; ctx_acq moves MS2+CCS only (RT untouched)."""
    m = build_student("small").eval()
    b = _batch()
    g = torch.Generator().manual_seed(0)
    r = torch.randn(3, m.cfg.context_dim, generator=g)
    with torch.no_grad():
        base = m(b)
        lc = m(b, ctx_lc=r)
        acq = m(b, ctx_acq=r)

    # chromatography context -> RT changes, fragmentation/mobility untouched.
    assert not torch.allclose(lc["rt"], base["rt"])
    assert torch.allclose(lc["ms2"], base["ms2"], atol=1e-6)
    assert torch.allclose(lc["ccs"], base["ccs"], atol=1e-6)
    # acquisition context -> MS2 and CCS change, RT untouched.
    assert not torch.allclose(acq["ms2"], base["ms2"])
    assert not torch.allclose(acq["ccs"], base["ccs"])
    assert torch.allclose(acq["rt"], base["rt"], atol=1e-6)


def test_param_efficient_finetune():
    """Freeze the backbone; fit ONLY a ContextBook row and RT must move toward a target."""
    torch.manual_seed(0)
    m = StudentModel(StudentConfig(d_model=48, n_layers=1, n_heads=2, dropout=0.0))
    m.set_norm(30.0, 10.0, 400.0, 50.0)
    for p in m.parameters():  # backbone + heads + projections frozen
        p.requires_grad_(False)

    book = ContextBook(n_acq=1, n_lc=1, context_dim=m.cfg.context_dim)
    opt = torch.optim.Adam(book.parameters(), lr=0.1)
    b = _batch()
    ids = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        target_rt = m(b)["rt"] + 2.0  # want RT shifted well off the base (standardized units)

    losses = []
    for _ in range(200):
        ctx_acq, ctx_lc = book(ids, ids)
        out = m(b, ctx_acq=ctx_acq, ctx_lc=ctx_lc)
        loss = ((out["rt"] - target_rt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    # The tiny context vector alone (16 numbers) pulls RT most of the way to the target.
    assert losses[-1] < losses[0] * 0.1, (losses[0], losses[-1])
    # Only the context learned; the backbone is untouched.
    assert m.token_emb.weight.grad is None


def test_context_book_zero_init_is_noop():
    book = ContextBook(2, 2, 8)
    ids = torch.tensor([0, 1])
    acq, lc = book(ids, ids)
    assert torch.count_nonzero(acq) == 0 and torch.count_nonzero(lc) == 0
