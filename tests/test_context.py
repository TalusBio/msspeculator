"""Context conditioning: ms_context/chrom_context head routing, MSContextEncoder, ChromRunbook."""

import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.models.context import (
    DEFAULT_FRAGMENTATIONS,
    DEFAULT_INSTRUMENTS,
    ChromRunbook,
    ContextBook,
    MSContextEncoder,
)
from pepdistill.models.registry import (
    ContextBundle,
    build_student,
    load_context,
    save_checkpoint,
)


def _batch():
    return collate([Precursor(Peptide("PEPTIDEK"), 2, "t"), Precursor(Peptide("ACDEFGHK"), 3, "t")])


def test_context_roundtrip(tmp_path):
    m = build_student("small")
    enc = MSContextEncoder(context_dim=m.cfg.context_dim)
    book = ChromRunbook(n_datasets=2, context_dim=m.cfg.context_dim)
    torch.nn.init.normal_(enc.frag_emb.weight)
    torch.nn.init.normal_(book.emb.weight)
    p = tmp_path / "m.ckpt"
    save_checkpoint(m, p, encoder=enc, runbook=book, dataset_index={"dsA": 1, "dsB": 2})

    b: ContextBundle = load_context(p)
    assert b.dataset_index == {"dsA": 1, "dsB": 2}
    z = torch.zeros(1, dtype=torch.long)
    assert torch.allclose(b.encoder(z, z, z, None), enc(z, z, z, None), atol=1e-6)
    assert torch.allclose(b.runbook(torch.tensor([1])), book(torch.tensor([1])), atol=1e-6)


def test_zero_ms_context_is_base_ms2():
    m = build_student("small").eval()
    b = _batch()
    base = m.forward(b)
    ctx = torch.zeros(2, m.cfg.context_dim)
    with_zero = m.forward(b, ms_context=ctx, chrom_context=ctx)
    assert torch.allclose(base["ms2"], with_zero["ms2"], atol=1e-6)
    assert torch.allclose(base["rt"], with_zero["rt"], atol=1e-6)


def test_ms_context_moves_ms2_not_ccs():
    torch.manual_seed(0)
    m = build_student("small").eval()
    for lin in (m.ms_to_frag,):  # de-neutralize the MS->frag projection
        torch.nn.init.normal_(lin.weight, std=0.3)
    b = _batch()
    ctx = torch.randn(2, m.cfg.context_dim)
    base, cond = m.forward(b), m.forward(b, ms_context=ctx)
    assert not torch.allclose(base["ms2"], cond["ms2"])  # MS context reaches fragments
    assert torch.allclose(base["ccs"], cond["ccs"], atol=1e-6)  # ...but never CCS


def test_context_book_zero_init_is_noop():
    book = ContextBook(2, 2, 8)
    ids = torch.tensor([0, 1])
    acq, lc = book(ids, ids)
    assert torch.count_nonzero(acq) == 0 and torch.count_nonzero(lc) == 0


def test_checkpoint_without_context_is_none(tmp_path):
    path = tmp_path / "m.ckpt"
    save_checkpoint(build_student("tiny"), path)
    assert load_context(path) is None


def test_context_aware_predict_changes_ms2_not_rt():
    """TorchRunner with a ms_context shifts MS2 but leaves RT and CCS (context-free) alone."""
    from pepdistill.predict.fast import TorchRunner, _bucket_arrays

    m = build_student("small").eval()
    m.set_norm(30.0, 10.0, 400.0, 50.0)
    enc = MSContextEncoder(context_dim=m.cfg.context_dim)
    torch.nn.init.normal_(enc.frag_emb.weight, std=0.5)  # nonzero -> real ms_context
    ms_vec = (
        enc(
            torch.tensor([enc.instrument_id("Lumos")]),
            torch.tensor([enc.detector_id("FTMS")]),
            torch.tensor([enc.fragmentation_id("HCD")]),
            torch.tensor([25.0]),
        )
        .detach()
        .numpy()[0]
    )

    precs = [Precursor(Peptide("PEPTIDEK"), 2, "t"), Precursor(Peptide("ACDEFGHK"), 2, "t")]
    tok, md, ch, _ = _bucket_arrays(precs, 8)
    ms2_base, rt_base, ccs_base = TorchRunner(m).run(tok, md, ch)
    ms2_ctx, rt_ctx, ccs_ctx = TorchRunner(m, ms_context=ms_vec).run(tok, md, ch)

    assert not (ms2_base == ms2_ctx).all()  # MS context moved MS2
    assert (rt_base == rt_ctx).all()  # RT is context-free (no chrom context)
    assert (ccs_base == ccs_ctx).all()  # CCS is peptide+charge only, no acquisition context


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
