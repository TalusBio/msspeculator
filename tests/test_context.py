"""Acquisition-context conditioning: no-op init, head isolation, param-efficient fine-tune."""

import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.models.context import ContextBook, ContextEncoder
from pepdistill.models.registry import build_student, load_checkpoint, load_context, save_checkpoint
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


def test_context_encoder_zero_init_is_base():
    """Fresh CE encoder emits ctx_acq=0 for any collision energy -> exact base MS2."""
    enc = ContextEncoder(context_dim=16)
    ce = torch.tensor([20.0, 30.0, 40.0])
    assert torch.count_nonzero(enc(ce)) == 0

    m = build_student("small").eval()
    b = _batch()
    with torch.no_grad():
        base = m(b)
        conditioned = m(b, ctx_acq=enc(torch.full((3,), 25.0)))
    assert torch.allclose(base["ms2"], conditioned["ms2"], atol=1e-6)


def test_checkpoint_persists_context(tmp_path):
    """save/load round-trips the CE encoder + per-run book so the artifact is complete."""
    m = build_student("small")
    enc = ContextEncoder(context_dim=m.cfg.context_dim, ce_center=30.0, ce_scale=10.0)
    torch.nn.init.normal_(enc.proj.weight, std=0.3)
    book = ContextBook(2, 2, m.cfg.context_dim)
    torch.nn.init.normal_(book.lc.weight, std=0.3)
    src = {"runA": 0, "runB": 1}

    path = tmp_path / "m.ckpt"
    save_checkpoint(m, path, encoder=enc, book=book, source_index=src)
    assert load_checkpoint(path).cfg.d_model == m.cfg.d_model  # model still loads

    ctx = load_context(path)
    assert ctx is not None and ctx.source_index == src
    assert ctx.encoder.ce_center == 30.0
    ce = torch.tensor([22.0, 31.0])
    assert torch.allclose(ctx.encoder(ce), enc(ce))  # reproduces ctx_acq exactly
    assert torch.allclose(ctx.book.lc.weight, book.lc.weight)


def test_checkpoint_without_context_is_none(tmp_path):
    path = tmp_path / "m.ckpt"
    save_checkpoint(build_student("tiny"), path)
    assert load_context(path) is None


def test_context_aware_predict_changes_ms2_not_rt():
    """TorchRunner with a ctx_acq shifts MS2 (and CCS) but leaves RT (context-free) alone."""
    from pepdistill.predict.fast import TorchRunner, _bucket_arrays

    m = build_student("small").eval()
    m.set_norm(30.0, 10.0, 400.0, 50.0)
    enc = ContextEncoder(context_dim=m.cfg.context_dim)
    torch.nn.init.normal_(enc.proj.weight, std=0.5)  # nonzero -> real ctx_acq
    ctx_vec = enc(torch.tensor([25.0])).detach().numpy()[0]

    precs = [Precursor(Peptide("PEPTIDEK"), 2, "t"), Precursor(Peptide("ACDEFGHK"), 2, "t")]
    tok, md, ch, _ = _bucket_arrays(precs, 8)
    ms2_base, rt_base, _ = TorchRunner(m).run(tok, md, ch)
    ms2_ctx, rt_ctx, _ = TorchRunner(m, ctx_acq=ctx_vec).run(tok, md, ch)

    assert not (ms2_base == ms2_ctx).all()  # CE context moved MS2
    assert (rt_base == rt_ctx).all()  # RT is context-free (no ctx_lc)


def test_context_encoder_learns_ce_dependence():
    """After a step of training, ctx_acq depends on collision energy and moves MS2."""
    torch.manual_seed(0)
    enc = ContextEncoder(context_dim=16)
    torch.nn.init.normal_(enc.proj.weight, std=0.5)  # simulate a trained (nonzero) encoder
    lo, hi = enc(torch.tensor([22.0])), enc(torch.tensor([31.0]))
    assert not torch.allclose(lo, hi)  # different CE -> different ctx_acq

    m = build_student("small").eval()
    b = _batch()
    with torch.no_grad():
        out_lo = m(b, ctx_acq=enc(torch.full((3,), 22.0)))
        out_hi = m(b, ctx_acq=enc(torch.full((3,), 31.0)))
    assert not torch.allclose(out_lo["ms2"], out_hi["ms2"])  # CE changes MS2
    assert torch.allclose(out_lo["rt"], out_hi["rt"], atol=1e-6)  # not RT
