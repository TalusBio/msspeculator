"""Real-speclib regime: a per-dataset ChromRunbook row gradient-descends the run's RT offset.

Two synthetic sources share iRT but differ in raw retention time by a fixed offset. The
base RT head (context-free) should track iRT; the runbook's dataset row should absorb the
per-dataset offset.
"""

import torch
from pepdistill.distill.dataset import MSFactors

from pepdistill.chem import Peptide
from pepdistill.data.config import SplitConfig
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.data.split import assign_split
from pepdistill.distill.context_regime import (
    RealExample,
    RealSpeclibDataset,
    RealSpeclibModule,
    _build_examples,
    fit_realspeclib,
)
from pepdistill.data.prospect import RealLabels
from pepdistill.models.context import ChromRunbook, MSContextEncoder
from pepdistill.models.registry import build_student
from pepdistill.teacher.base import PrecursorLabels
from pepdistill.teacher.fake import FakeTeacher

OFFSET = 25.0


def _make_examples(n: int) -> list[RealExample]:
    """N examples over one peptide/split, varying only raw_rt -- enough columns for the
    batches()/masking tests, which don't care about realistic labels or acquisition factors."""
    prec = Precursor(Peptide("PEPTIDEK", ()), 2, "train")
    label = FakeTeacher().predict([prec])[0]
    return [
        RealExample(
            precursor=prec,
            label=label,
            raw_rt=float(i),
            instrument_id=0,
            detector_id=0,
            fragmentation_id=0,
            energy=25.0,
            dataset_id=1,
        )
        for i in range(n)
    ]


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


def test_runbook_learns_dataset_offset():
    real, seqs = _real()  # existing fixture; two raw_files, one dataset
    model = build_student("small")
    module = fit_realspeclib(
        model,
        real,
        epochs=60,
        batch_size=32,
        accelerator="cpu",
        dataset_name="dsA",
        enable_progress_bar=False,
    )
    assert module.dataset_index == {"dsA": 1}

    # raw RT (dataset row) shifts vs iRT (row 0) for the same peptide
    m = module.model.eval()
    b = collate([Precursor(Peptide(seqs[0]), 2, "t")])
    with torch.no_grad():
        ds_row = torch.tensor([min(module.dataset_index.values())])
        rt_ds = m.forward(b, chrom_context=module.runbook(ds_row))["rt"]
        rt_irt = m.forward(b, chrom_context=module.runbook(torch.tensor([0])))["rt"]
    assert not torch.allclose(rt_ds, rt_irt)


def test_build_examples_uses_config_instrument_not_per_run_metadata():
    """PROSPECT acquisition metadata carries no instrument column, so instrument must come from
    the config constant (threaded through fit_realspeclib) rather than a per-run lookup — every
    example gets the same instrument_id regardless of source, even if a run's acquisition dict
    happened to carry an 'instrument' key."""
    real, _ = _real()
    real.acquisition["rfB"]["instrument"] = "QExactive"  # per-run value must be ignored
    encoder = MSContextEncoder(context_dim=8)
    examples = _build_examples(real, encoder, dataset_id=1, instrument="Lumos")
    expected = encoder.instrument_id("Lumos")
    assert all(e.instrument_id == expected for e in examples)


def test_fit_realspeclib_threads_instrument_into_examples():
    real, _ = _real()
    model = build_student("small")
    module = fit_realspeclib(
        model,
        real,
        epochs=1,
        batch_size=32,
        accelerator="cpu",
        dataset_name="dsA",
        instrument="Lumos",
        enable_progress_bar=False,
    )
    expected = module.encoder.instrument_id("Lumos")
    examples = _build_examples(real, module.encoder, dataset_id=1, instrument="Lumos")
    assert all(e.instrument_id == expected for e in examples)


def test_freeze_backbone_trains_only_context():
    model = build_student("small")
    cdim = model.cfg.context_dim
    runbook = ChromRunbook(1, cdim)
    encoder = MSContextEncoder(context_dim=cdim)
    module = RealSpeclibModule(model, runbook, encoder, freeze_backbone=True)

    # Backbone is frozen; the context modules stay trainable.
    assert not any(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in runbook.parameters())
    assert all(p.requires_grad for p in encoder.parameters())

    # The optimizer therefore only sees the context modules, not the backbone.
    opt = module.configure_optimizers()
    optimized = {id(p) for group in opt.param_groups for p in group["params"]}
    context_params = {id(p) for p in (*runbook.parameters(), *encoder.parameters())}
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


def test_mixed_energy_batch_no_longer_raises():
    """Per-spectrum energy (Task 2) makes a batch mixing present/absent energy ordinary; it must
    be masked per example inside MSContextEncoder rather than rejected here."""
    examples = _make_examples(4)
    examples[1].energy = float("nan")
    ds = RealSpeclibDataset(examples)
    gen = torch.Generator().manual_seed(0)
    batch = next(iter(ds.batches(4, False, gen)))
    assert batch.ms_factors.energy is not None
    assert int(torch.isnan(batch.ms_factors.energy).sum()) == 1
