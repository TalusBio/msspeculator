"""Real-speclib regime: per-raw_file ctx_lc gradient-descends the run's RT offset.

Two synthetic sources share iRT but differ in raw retention time by a fixed offset. The
base RT head (context-free) should track iRT; ctx_lc should absorb the per-run offset.
"""

import torch

from pepdistill.chem import Peptide
from pepdistill.data.config import SplitConfig
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.data.split import assign_split
from pepdistill.distill.context_regime import fit_realspeclib
from pepdistill.data.prospect import RealLabels
from pepdistill.models.registry import build_student
from pepdistill.teacher.base import PrecursorLabels
from pepdistill.teacher.fake import FakeTeacher

OFFSET = 25.0


def _real():
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = [
        "".join(aa[(i * 7 + j * 3) % 20] for j in range(8 + i % 4)) + "K" for i in range(40)
    ]
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
            ctx_acq, ctx_lc = module.book(ids, ids)
            out = m.forward_context(b, ctx_acq=ctx_acq, ctx_lc=ctx_lc)
            preds.append(float(out["rt"][0] * m.rt_std + m.rt_mean))
            # base (context-free) RT must be finite (tracks iRT), independent of source.
            assert torch.isfinite(out["rt_base"]).all()
    assert 10.0 < (preds[1] - preds[0]) < 40.0, preds  # ~OFFSET=25, learned by ctx_lc alone
