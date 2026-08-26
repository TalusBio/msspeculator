"""Context conditioning: ms_context/chrom_context head routing, MSContextEncoder, ChromRunbook."""

import torch
import fsspec

from msspeculator.chem import Peptide
from msspeculator.data.encode import collate
from msspeculator.data.precursors import Precursor
from msspeculator.models.context import (
    DEFAULT_FRAGMENTATIONS,
    DEFAULT_INSTRUMENTS,
    ChromRunbook,
    MSContextEncoder,
)
from msspeculator.models.registry import (
    ContextBundle,
    build_student,
    load_checkpoint,
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


def test_a_new_source_cannot_renumber_trained_rows(tmp_path):
    """The corpus growing a source must not move a dataset that already learned a row.

    The prepared manifest numbers datasets by sorted position, so adding a source that sorts first
    renumbers everything after it. While the name->row map lived apart from the weights, that
    renumbering silently handed each trained row to a different dataset: no error, just RT context
    predicted from the wrong chromatography.
    """
    from msspeculator.distill.pipeline import _runbook_for_datasets

    model = build_student("small")
    book = ChromRunbook(n_datasets=2, context_dim=model.cfg.context_dim)
    book.ensure(["multi_ptm_ps", "prospect_tum_hla"])
    assert book.names == {"multi_ptm_ps": 1, "prospect_tum_hla": 2}
    with torch.no_grad():
        book.emb.weight[book.row("multi_ptm_ps")].fill_(0.25)
        book.emb.weight[book.row("prospect_tum_hla")].fill_(0.75)

    # A library whose name sorts before both of them: under sorted numbering it would take row 1.
    grown = _runbook_for_datasets(
        book, ["evosep_library", "multi_ptm_ps", "prospect_tum_hla"], model.cfg.context_dim
    )
    assert grown.names == {"multi_ptm_ps": 1, "prospect_tum_hla": 2, "evosep_library": 3}
    assert torch.allclose(grown.emb.weight[grown.row("multi_ptm_ps")], torch.tensor(0.25))
    assert torch.allclose(grown.emb.weight[grown.row("prospect_tum_hla")], torch.tensor(0.75))
    # The new row starts neutral, so the library contributes nothing until it trains.
    assert torch.count_nonzero(grown.emb.weight[grown.row("evosep_library")]) == 0

    # And the map survives the round trip that the runtime resolves names through.
    path = tmp_path / "grown.ckpt"
    save_checkpoint(model, path, runbook=grown)
    assert load_context(path).runbook.names == grown.names


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


def test_checkpoint_without_context_is_none(tmp_path):
    path = tmp_path / "m.ckpt"
    save_checkpoint(build_student("flash"), path)
    assert load_context(path) is None


def test_checkpoint_loads_from_fsspec_uri(tmp_path):
    local = tmp_path / "m.ckpt"
    model = build_student("flash")
    encoder = MSContextEncoder(context_dim=model.cfg.context_dim)
    save_checkpoint(model, local, encoder=encoder)
    uri = "memory://msspeculator-tests/m.ckpt"
    with local.open("rb") as src, fsspec.open(uri, "wb") as dst:
        dst.write(src.read())
    loaded = load_checkpoint(uri)
    context = load_context(uri)
    assert loaded.cfg == model.cfg
    assert context is not None and context.encoder is not None


def test_context_aware_predict_changes_ms2_not_rt():
    """TorchRunner with a ms_context shifts MS2 but leaves RT and CCS (context-free) alone."""
    from msspeculator.predict.fast import TorchRunner, _bucket_arrays

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
    tok, mc, mm, mp, mn, ch, _ = _bucket_arrays(precs, 8)
    ms2_base, rt_base, ccs_base = TorchRunner(m).run(tok, mc, mm, mp, mn, ch)
    ms2_ctx, rt_ctx, ccs_ctx = TorchRunner(m, ms_context=ms_vec).run(tok, mc, mm, mp, mn, ch)

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


def test_ms_context_unknown_rows_are_fixed_neutral_even_for_old_checkpoint_weights():
    enc = MSContextEncoder(context_dim=8)
    with torch.no_grad():
        enc.inst_emb.weight[0].fill_(1.0)
        enc.det_emb.weight[0].fill_(2.0)
        enc.frag_emb.weight[0].fill_(3.0)
    unknown = torch.zeros(1, dtype=torch.long)
    assert torch.equal(enc(unknown, unknown, unknown, None), torch.zeros(1, 8))
    assert enc.inst_emb.padding_idx == enc.det_emb.padding_idx == enc.frag_emb.padding_idx == 0


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

    # Loading an old checkpoint cannot make the reserved neutral row non-neutral.
    with torch.no_grad():
        book.emb.weight[0].fill_(2.0)
        book.log_scale.weight[0].fill_(3.0)
        book.shift.weight[0].fill_(4.0)
    neutral = torch.zeros(2, dtype=torch.long)
    scale, shift = book.affine(neutral)
    assert torch.equal(book(neutral), torch.zeros(2, 8))
    assert torch.equal(scale, torch.ones(2))
    assert torch.equal(shift, torch.zeros(2))


def test_chrom_runbook_rows_learn_independently():
    book = ChromRunbook(n_datasets=3, context_dim=8)
    torch.nn.init.normal_(book.emb.weight)  # simulate a trained book
    a = book(torch.tensor([1]))
    b = book(torch.tensor([2]))
    assert not torch.allclose(a, b)
    assert book.neutral(2, torch.device("cpu")).shape == (2, 8)


def test_runbook_affine_is_identity_at_init():
    """Zero-init and the neutral row must both reproduce base RT EXACTLY, not approximately."""
    import torch

    from msspeculator.chem import Peptide
    from msspeculator.data.encode import collate
    from msspeculator.data.precursors import Precursor
    from msspeculator.models.context import ChromRunbook
    from msspeculator.models.registry import build_student

    m = build_student("small").eval()
    rb = ChromRunbook(2, m.cfg.context_dim)
    batch = collate([Precursor(Peptide("SAMPLER"), 2, "t")])

    for did in (0, 1, 2):  # 0 = neutral row, others untrained
        ids = torch.tensor([did])
        scale, shift = rb.affine(ids)
        assert float(scale) == 1.0 and float(shift) == 0.0
        with torch.no_grad():
            base = m.forward_context(batch)["rt"]
            cond = m.forward_context(batch, chrom_context=rb(ids), chrom_affine=rb.affine(ids))[
                "rt"
            ]
        assert torch.equal(base, cond), f"row {did} is not exactly identity"


def test_runbook_affine_recovers_a_known_scale_and_shift():
    """Fitting the runbook alone must recover an affine the data was generated with.

    This is the whole point of the mechanism: an additive feature bias cannot express a
    rescale, so if the target is a*base+b with a far from 1, only the affine can fit it.
    """
    import torch

    from msspeculator.chem import Peptide
    from msspeculator.data.encode import collate
    from msspeculator.data.precursors import Precursor
    from msspeculator.models.context import ChromRunbook
    from msspeculator.models.registry import build_student

    torch.manual_seed(0)
    m = build_student("small").eval()
    for p in m.parameters():  # freeze the student: only the runbook may learn
        p.requires_grad_(False)
    rb = ChromRunbook(1, m.cfg.context_dim)
    # Zero the context vector's path so the affine is the only route; otherwise the two
    # mechanisms are partly interchangeable and the recovered numbers are not identifiable.
    rb.emb.weight.requires_grad_(False)

    precs = [Precursor(Peptide(s), 2, "t") for s in ("SAMPLER", "PEPTIDEK", "ACDEFGHIK")]
    batch = collate(precs)
    ids = torch.ones(len(precs), dtype=torch.long)
    with torch.no_grad():
        base = m.forward_context(batch)["rt"]
    true_a, true_b = 3.5, -1.25
    target = true_a * base + true_b

    opt = torch.optim.Adam([rb.log_scale.weight, rb.shift.weight], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        rt = m.forward_context(batch, chrom_context=rb(ids), chrom_affine=rb.affine(ids))["rt"]
        loss = torch.nn.functional.mse_loss(rt, target)
        loss.backward()
        opt.step()

    a, b = rb.affine(ids)
    assert abs(float(a[0]) - true_a) < 0.05, f"scale {float(a[0])} != {true_a}"
    assert abs(float(b[0]) - true_b) < 0.05, f"shift {float(b[0])} != {true_b}"
    assert float(a[0]) > 0.0, "exp() must keep the scale positive"


def test_affine_never_touches_rt_base():
    """rt_base is the iRT anchor; conditioning it would destroy the frame it defines."""
    import torch

    from msspeculator.chem import Peptide
    from msspeculator.data.encode import collate
    from msspeculator.data.precursors import Precursor
    from msspeculator.models.context import ChromRunbook
    from msspeculator.models.registry import build_student

    m = build_student("small").eval()
    rb = ChromRunbook(1, m.cfg.context_dim)
    with torch.no_grad():  # a decidedly non-identity affine
        rb.log_scale.weight.fill_(1.0)
        rb.shift.weight.fill_(5.0)
    batch = collate([Precursor(Peptide("SAMPLER"), 2, "t")])
    ids = torch.tensor([1])

    with torch.no_grad():
        plain = m.forward_context(batch)
        conditioned = m.forward_context(batch, chrom_context=rb(ids), chrom_affine=rb.affine(ids))
    assert torch.equal(plain["rt_base"], conditioned["rt_base"]), "affine leaked into rt_base"
    assert not torch.equal(plain["rt"], conditioned["rt"]), "affine did not affect rt"


def _perturbed_encoder(context_dim: int = 8) -> MSContextEncoder:
    """A fresh encoder has energy_mlp's LAST Linear zero-init (weight AND bias), so
    energy_mlp(anything) == 0 regardless of input; every "does energy do X" test built
    on a bare MSContextEncoder is vacuous, because masked-vs-unmasked collapse to the
    same (zero) output no matter which side of a masking bug you're looking at. Perturb
    every energy_mlp parameter so mlp(0) is genuinely nonzero, the way it would be after
    real training, so these tests can actually tell implementations apart."""
    torch.manual_seed(0)
    enc = MSContextEncoder(context_dim=context_dim)
    for p in enc.energy_mlp.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return enc


def test_nan_energy_contributes_exactly_zero():
    enc = _perturbed_encoder()
    ids = torch.zeros(3, dtype=torch.long)
    none_out = enc(ids, ids, ids, energy=None)
    nan_out = enc(ids, ids, ids, energy=torch.full((3,), float("nan")))
    assert torch.allclose(none_out, nan_out)


def test_mixed_energy_masks_per_example_not_per_batch():
    enc = _perturbed_encoder()
    ids = torch.zeros(2, dtype=torch.long)
    mixed = enc(ids, ids, ids, energy=torch.tensor([28.0, float("nan")]))
    present = enc(ids, ids, ids, energy=torch.tensor([28.0, 28.0]))
    absent = enc(ids, ids, ids, energy=None)
    assert torch.allclose(mixed[0], present[0])  # row 0 keeps its energy term
    assert torch.allclose(mixed[1], absent[1])  # row 1 has none at all
    assert torch.isfinite(mixed).all()


def test_masking_happens_after_the_mlp_not_by_filling_zero():
    """energy_mlp has a bias, so mlp(0) != 0; filling would inject a learned constant."""
    enc = _perturbed_encoder()
    ids = torch.zeros(1, dtype=torch.long)
    filled = enc(ids, ids, ids, energy=torch.zeros(1))
    masked = enc(ids, ids, ids, energy=torch.full((1,), float("nan")))
    assert not torch.allclose(filled, masked)
    # Directional: pin down WHICH side carries the nonzero term. "not allclose" alone is
    # symmetric and also passes if the present/absent mask were reversed; a masked
    # example must equal the no-energy path exactly, not merely differ from the filled one.
    absent = enc(ids, ids, ids, energy=None)
    assert torch.allclose(masked, absent)
