"""Real-speclib regime: a per-dataset ChromRunbook row gradient-descends the run's RT offset.

Two synthetic sources share iRT but differ in raw retention time by a fixed offset. The
base RT head (context-free) should track iRT; the runbook's dataset row should absorb the
per-dataset offset.
"""

import math

import pytest
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
    _RealValidationEarlyStop,
    _build_examples,
    establish_rt_norm,
    fit_realspeclib,
    fit_realspeclib_datasets,
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


def test_epoch_energy_counters_track_masking_and_reset():
    """train_energy_masked/train_energy_present are the only visible signal that missing
    energy is masked rather than silently dropped. Call the hooks directly (no Trainer
    needed) so a regression to "counters never increment" is caught without a full fit."""
    model = build_student("tiny")
    cdim = model.cfg.context_dim
    module = RealSpeclibModule(model, ChromRunbook(1, cdim), MSContextEncoder(context_dim=cdim))

    examples = _make_examples(4)
    examples[1].energy = float("nan")
    examples[3].energy = float("nan")  # 2 present, 2 masked
    ds = RealSpeclibDataset(examples)
    gen = torch.Generator().manual_seed(0)
    batch = next(iter(ds.batches(4, False, gen)))

    module.on_train_epoch_start()
    assert module._energy_masked == 0
    assert module._energy_present == 0

    module.training_step(batch, 0)
    module.training_step(batch, 1)  # a second step in the same epoch: counts accumulate
    assert module._energy_present == 4  # 2 present rows x 2 steps
    assert module._energy_masked == 4  # 2 masked rows x 2 steps

    module.on_train_epoch_start()  # next epoch must reset, not keep accumulating
    assert module._energy_masked == 0
    assert module._energy_present == 0


def test_establishes_from_combined_sufficient_statistics():
    model = build_student("small")
    # Two sources: values [10, 20] and [30]. mean 20, population std sqrt(200/3).
    stats = [(2, 30.0, 10.0**2 + 20.0**2), (1, 30.0, 30.0**2)]
    assert establish_rt_norm(model, stats) is True
    assert float(model.rt_mean) == pytest.approx(20.0)
    assert float(model.rt_std) == pytest.approx(math.sqrt(200.0 / 3.0))
    assert bool(model.norm_established)


def test_does_not_re_establish_an_existing_affine():
    model = build_student("small")
    model.set_norm(rt_mean=1.0, rt_std=2.0)
    assert establish_rt_norm(model, [(2, 30.0, 500.0)]) is False
    assert float(model.rt_mean) == pytest.approx(1.0)


def test_zero_rows_raises():
    model = build_student("small")
    with pytest.raises(ValueError, match="no examples to establish the RT affine"):
        establish_rt_norm(model, [(0, 0.0, 0.0)])


def test_degenerate_variance_falls_back_to_unit_std():
    model = build_student("small")
    # Three identical values -> variance 0; std must not be 0 (it divides).
    stats = [(3, 15.0, 75.0)]
    assert establish_rt_norm(model, stats) is True
    assert float(model.rt_std) == pytest.approx(1.0)


def test_fit_from_prebuilt_datasets_trains_and_reports_metrics(tmp_path):
    from pepdistill.distill.context_regime import RealSpeclibDataset

    model = build_student("small")
    model.set_norm(rt_mean=0.0, rt_std=1.0)
    cdim = model.cfg.context_dim
    examples = _make_examples(8)  # existing helper in this test module
    module = fit_realspeclib_datasets(
        model,
        RealSpeclibDataset(examples),
        RealSpeclibDataset(examples[:2]),
        runbook=ChromRunbook(n_datasets=1, context_dim=cdim),
        dataset_index={"pool": 1},
        encoder=MSContextEncoder(context_dim=cdim),
        epochs=1,
        batch_size=4,
        enable_progress_bar=False,
    )
    assert module.dataset_index == {"pool": 1}
    metrics = module.trainer.callback_metrics
    assert "val/pool/spectral_angle" in metrics
    assert "val/pool/irt_mae" in metrics
    assert "val/pool/rawrt_mae" in metrics
    assert metrics["val/pool/n"] == 2
    assert "val_spectral_angle" not in metrics


def test_validation_early_stop_fails_on_missing_metric_key():
    class Trainer:
        callback_metrics = {"val/other/spectral_angle": torch.tensor(0.5)}

    callback = _RealValidationEarlyStop(
        patience=5, min_delta=1e-3, expected_keys={"val/pool/spectral_angle"}
    )
    with pytest.raises(RuntimeError, match="missing=.*val/pool/spectral_angle"):
        callback.on_validation_epoch_end(Trainer(), None)
