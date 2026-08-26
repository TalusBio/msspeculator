"""DistillModule smoke test + shared-backbone check. Tiny CPU fit, negligible compute."""

import pytest
import torch

from msspeculator.chem import Peptide
from msspeculator.distill.dataset import DistillDataset, MSFactors, collate_with_labels
from msspeculator.distill.lightning import DistillModule, fit_distill
from msspeculator.data.precursors import Precursor
from msspeculator.models.context import MSContextEncoder
from msspeculator.models.registry import build_student
from msspeculator.teacher.fake import FakeTeacher


def _tiny_labeled_batch(ms_factors=None):
    """One precursor + FakeTeacher labels, collated, with the given ms_factors attached."""
    prec = Precursor(Peptide("SAMPLER"), 2, "train")
    labels = FakeTeacher().predict([prec])
    lb = collate_with_labels([prec], labels)
    lb.ms_factors = ms_factors
    return lb


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
    # Same tensor object, not a copy, a fine-tune regime would train the same weights.
    assert a.model is b.model
    assert a.model.token_emb.weight is b.model.token_emb.weight


def test_distill_predict_matches_base_when_context_off():
    """context_encoder=None must reproduce the base forward (no ms_context) exactly."""
    model = build_student("small")
    model.eval()  # dropout off, so the two forward passes are directly comparable
    mod = DistillModule(model)
    batch = _tiny_labeled_batch()
    out = mod._predict(batch)
    base = model(batch.inputs)
    for k in base:
        assert torch.equal(out[k], base[k])


def test_distill_requires_factors_when_context_on():
    m = build_student("small")
    mod = DistillModule(m, context_encoder=MSContextEncoder(context_dim=m.cfg.context_dim))
    # a batch with no ms_factors under an active encoder must error, not fabricate context
    with pytest.raises(ValueError):
        mod._predict(_tiny_labeled_batch(ms_factors=None))


def test_distill_uses_ms_factors_when_context_on():
    """With ms_factors present, _predict routes through context_encoder + forward(ms_context=...)."""
    m = build_student("small")
    encoder = MSContextEncoder(context_dim=m.cfg.context_dim)
    mod = DistillModule(m, context_encoder=encoder)
    factors = MSFactors(
        instrument_id=torch.tensor([encoder.instrument_id("Lumos")]),
        detector_id=torch.tensor([encoder.detector_id("FTMS")]),
        fragmentation_id=torch.tensor([encoder.fragmentation_id("HCD")]),
        energy=torch.tensor([30.0], dtype=torch.float32),
    )
    out = mod._predict(_tiny_labeled_batch(ms_factors=factors))
    assert torch.isfinite(out["ms2"]).all()
    assert torch.isfinite(out["rt"]).all()
    assert torch.isfinite(out["ccs"]).all()
