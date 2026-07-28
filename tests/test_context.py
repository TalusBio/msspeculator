"""Acquisition-context conditioning: no-op init, head isolation, param-efficient fine-tune."""

import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.models.context import (
    DEFAULT_FRAGMENTATIONS,
    DEFAULT_INSTRUMENTS,
    ChromRunbook,
    ContextBook,
    ContextEncoder,
    MSContextEncoder,
)
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


def test_context_encoder_factors():
    """Analyzer/fragmentation factors shift ctx_acq; unknown -> id 0 (zero row) is a no-op."""
    enc = ContextEncoder(context_dim=8)
    assert enc.analyzer_id("FTMS") == 1 and enc.analyzer_id("NOPE") == 0
    assert enc.frag_id("HCD") == 1 and enc.frag_id("NOPE") == 0

    torch.nn.init.normal_(enc.ana_emb.weight, std=0.5)
    torch.nn.init.normal_(enc.frag_emb.weight, std=0.5)
    with torch.no_grad():  # keep the "unknown" row (0) zero
        enc.ana_emb.weight[0].zero_()
        enc.frag_emb.weight[0].zero_()

    ce = torch.tensor([30.0])
    base = enc(ce)  # CE only
    known = enc(ce, torch.tensor([enc.analyzer_id("FTMS")]), torch.tensor([enc.frag_id("HCD")]))
    unknown = enc(ce, torch.tensor([enc.analyzer_id("x")]), torch.tensor([enc.frag_id("y")]))
    assert not torch.allclose(base, known)  # known factors move ctx_acq
    assert torch.allclose(base, unknown)  # unknown -> row 0 (zero) -> no-op


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


def test_ms_context_blank_is_zero():
    enc = MSContextEncoder(context_dim=8)
    z = torch.zeros(4, dtype=torch.long)
    # all-unknown ids + no energy -> exact zero (the neutral "blank")
    out = enc(z, z, z, energy=None)
    assert out.shape == (4, 8)
    assert torch.allclose(out, torch.zeros(4, 8))


def test_ms_context_ids_and_unknown_fallback():
    enc = MSContextEncoder(context_dim=8)
    assert enc.instrument_id(DEFAULT_INSTRUMENTS[1]) == 1
    assert enc.detector_id("nonsense") == 0  # unknown -> 0
    assert enc.fragmentation_id(DEFAULT_FRAGMENTATIONS[1]) == 1


def test_ms_context_energy_is_wired_after_training_step():
    torch.manual_seed(0)
    enc = MSContextEncoder(context_dim=8)
    # force the (zero-init) energy path to become nonzero, then confirm energy changes output
    for p in enc.energy_mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    z = torch.zeros(4, dtype=torch.long)
    lo = enc(z, z, z, energy=torch.full((4,), 20.0))
    hi = enc(z, z, z, energy=torch.full((4,), 40.0))
    assert not torch.allclose(lo, hi)


def test_chrom_runbook_neutral_row_zero():
    book = ChromRunbook(n_datasets=3, context_dim=8)
    out = book(torch.tensor([0, 0]))  # index 0 = neutral/iRT
    assert out.shape == (2, 8)
    assert torch.allclose(out, torch.zeros(2, 8))  # zero-init


def test_chrom_runbook_rows_learn_independently():
    book = ChromRunbook(n_datasets=3, context_dim=8)
    torch.nn.init.normal_(book.emb.weight)  # simulate a trained book
    a = book(torch.tensor([1]))
    b = book(torch.tensor([2]))
    assert not torch.allclose(a, b)
    assert book.neutral(2, torch.device("cpu")).shape == (2, 8)
