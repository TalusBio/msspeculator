"""DistillModule smoke test + shared-backbone check. Tiny CPU fit, negligible compute."""

import torch

from pepdistill.chem import Peptide
from pepdistill.distill.dataset import DistillDataset
from pepdistill.distill.lightning import DistillModule, fit_distill
from pepdistill.data.precursors import Precursor
from pepdistill.models.registry import build_student
from pepdistill.teacher.fake import FakeTeacher


def _dataset():
    seqs = [
        "SAMPLER",
        "PEPTIDEK",
        "ACDEFGHIK",
        "MKLVWYR",
        "LLLLLLK",
        "GASPVTCLIN",
        "EDCAAKR",
        "YYWWFFK",
    ]
    precs = [Precursor(Peptide(s), 2 + (i % 3), "train") for i, s in enumerate(seqs)]
    labels = FakeTeacher().predict(precs)
    return DistillDataset(precs, labels)


def test_fit_distill_runs_and_sets_norm():
    ds = _dataset()
    model = build_student("small")
    module = fit_distill(
        model,
        ds,
        ds,
        epochs=2,
        batch_size=4,
        accelerator="cpu",
        enable_progress_bar=False,
    )
    # Norm buffers were set from the data (not the identity default of 1.0).
    assert float(module.model.ccs_std) != 1.0
    # Trained backbone still produces bounded MS2 and finite RT/CCS.
    out = module.model(ds.batches(4, False, torch.Generator()).__next__().inputs)
    assert out["ms2"].min() >= 0.0 and out["ms2"].max() <= 1.0
    assert torch.isfinite(out["rt"]).all() and torch.isfinite(out["ccs"]).all()


def test_regimes_share_backbone():
    """Two regime modules built on one StudentModel share the SAME parameters."""
    model = build_student("small")
    a = DistillModule(model)
    b = DistillModule(model)
    # Same tensor object, not a copy — a fine-tune regime would train the same weights.
    assert a.model is b.model
    assert a.model.token_emb.weight is b.model.token_emb.weight
