"""Real-speclib regime: per-raw_file ctx_lc gradient-descends the run's RT offset.

Two synthetic sources share iRT but differ in raw retention time by a fixed offset. The
base RT head (context-free) should track iRT; ctx_lc should absorb the per-run offset.
"""

import torch
from pepdistill.distill.dataset import MSFactors

from pepdistill.chem import Peptide
from pepdistill.data.config import SplitConfig
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.data.split import assign_split
from pepdistill.distill.context_regime import RealSpeclibModule, fit_realspeclib
from pepdistill.data.prospect import RealLabels
from pepdistill.models.context import ContextBook, ContextEncoder
from pepdistill.models.registry import build_student
from pepdistill.teacher.base import PrecursorLabels
from pepdistill.teacher.fake import FakeTeacher

OFFSET = 25.0


def _real():
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["".join(aa[(i * 7 + j * 3) % 20] for j in range(8 + i % 4)) + "K" for i in range(40)]
    base = FakeTeacher().predict([Precursor(Peptide(s), 2, "t") for s in seqs])
    precs, labels, raw_rt, sources = [], [], [], []
    for s, lab in zip(seqs, base):
        split = assign_split(s, SplitConfig())
        for src, off in (("rfA", 0.0), ("rfB", OFFSET)):
            precs.append(Precursor(Peptide(s), 2, split))
            labels.append(PrecursorLabels(ms2=lab.ms2, rt=lab.rt, ccs=float("nan")))
            raw_rt.append(lab.rt + off)
            sources.append(src)
    return RealLabels(precs, labels, raw_rt, sources, {"rfA": {}, "rfB": {}}), seqs


def test_context_regime_learns_run_offset():
    real, seqs = _real()
    model = build_student("small")
    module = fit_realspeclib(
        model, real, epochs=80, batch_size=32, accelerator="cpu", enable_progress_bar=False
    )
    assert module.source_index == {"rfA": 0, "rfB": 1}

    # The two runs got distinct ctx_lc vectors.
    lc = module.book.lc.weight.detach()
    assert float((lc[0] - lc[1]).abs().sum()) > 1e-3

    # Same peptide, switch source -> predicted raw RT shifts by ~OFFSET.
    m = module.model.eval()
    b = collate([Precursor(Peptide(seqs[0]), 2, "t")])
    preds = []
    with torch.no_grad():
        for src in (0, 1):
            ids = torch.tensor([src])
            ctx_lc = module.book.lc(ids)  # ctx_acq is encoder-driven; base MS2 here (ctx_acq=None)
            out = m.forward_context(b, ctx_acq=None, ctx_lc=ctx_lc)
            preds.append(float(out["rt"][0] * m.rt_std + m.rt_mean))
            # base (context-free) RT must be finite (tracks iRT), independent of source.
            assert torch.isfinite(out["rt_base"]).all()
    assert 10.0 < (preds[1] - preds[0]) < 40.0, preds  # ~OFFSET=25, learned by ctx_lc alone


def test_freeze_backbone_trains_only_context():
    model = build_student("small")
    cdim = model.cfg.context_dim
    book = ContextBook(n_acq=1, n_lc=2, context_dim=cdim)
    encoder = ContextEncoder(context_dim=cdim)
    module = RealSpeclibModule(model, book, encoder, freeze_backbone=True)

    # Backbone is frozen; the context modules stay trainable.
    assert not any(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in book.parameters())
    assert all(p.requires_grad for p in encoder.parameters())

    # The optimizer therefore only sees the context vectors, not the backbone.
    opt = module.configure_optimizers()
    optimized = {id(p) for group in opt.param_groups for p in group["params"]}
    context_params = {id(p) for p in (*book.parameters(), *encoder.parameters())}
    assert optimized == context_params


def test_ms_factors_to_device_handles_none_energy():
    f = MSFactors(
        instrument_id=torch.zeros(2, dtype=torch.long),
        detector_id=torch.zeros(2, dtype=torch.long),
        fragmentation_id=torch.zeros(2, dtype=torch.long),
        energy=None,
    )
    moved = f.to("cpu")
    assert moved.energy is None
    assert moved.instrument_id.shape == (2,)
