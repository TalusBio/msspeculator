"""Real-speclib regime: a per-dataset ChromRunbook row gradient-descends the run's RT offset.

Two synthetic sources share iRT but differ in raw retention time by a fixed offset. The
base RT head (context-free) should track iRT; the runbook's dataset row should absorb the
per-dataset offset.
"""

import json
import math
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

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
    _RealPlateauDecay,
    _RealValidationEarlyStop,
    establish_rt_norm,
    fit_realspeclib_datasets,
)
from pepdistill.models.context import ChromRunbook, MSContextEncoder
from pepdistill.models.registry import build_student
from pepdistill.teacher.base import PrecursorLabels
from pepdistill.teacher.fake import FakeTeacher

OFFSET = 25.0


def _make_examples(n: int) -> list[RealExample]:
    """N examples over one peptide/split, varying only raw_rt; enough columns for the
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


def _two_run_examples(encoder: MSContextEncoder) -> tuple[list[RealExample], list[str]]:
    """One dataset observed through two runs whose raw RT differs by a fixed offset.

    Both runs share the dataset row: raw RT is keyed per dataset, not per raw file, so the row
    has to absorb their offset from the iRT frame. Splits come from the project's own hash, so a
    peptide's mod-forms and charges stay on one side of it.
    """
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["".join(aa[(i * 7 + j * 3) % 20] for j in range(8 + i % 4)) + "K" for i in range(40)]
    base = FakeTeacher().predict([Precursor(Peptide(s), 2, "t") for s in seqs])
    examples = []
    for sequence, label in zip(seqs, base):
        split = assign_split(sequence, SplitConfig())
        for offset in (0.0, OFFSET):
            examples.append(
                RealExample(
                    precursor=Precursor(Peptide(sequence), 2, split),
                    label=PrecursorLabels(ms2=label.ms2, rt=label.rt, ccs=float("nan")),
                    raw_rt=label.rt + offset,
                    instrument_id=encoder.instrument_id("Lumos"),
                    detector_id=0,
                    fragmentation_id=0,
                    energy=float("nan"),
                    dataset_id=1,
                )
            )
    return examples, seqs


def test_runbook_learns_dataset_offset():
    model = build_student("small")
    encoder = MSContextEncoder(context_dim=model.cfg.context_dim)
    examples, seqs = _two_run_examples(encoder)
    train = [e for e in examples if e.precursor.split != "val"]
    labels = [e.label.rt for e in train]
    assert establish_rt_norm(model, [(len(labels), sum(labels), sum(v * v for v in labels))]), (
        "the affine must be set before fitting, or the RT head trains in the wrong units"
    )

    module = fit_realspeclib_datasets(
        model,
        RealSpeclibDataset(train),
        RealSpeclibDataset([e for e in examples if e.precursor.split == "val"]),
        runbook=ChromRunbook(1, model.cfg.context_dim),
        dataset_index={"dsA": 1},
        encoder=encoder,
        epochs=60,
        batch_size=32,
        accelerator="cpu",
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
    model = build_student("flash")
    cdim = model.cfg.context_dim
    module = RealSpeclibModule(model, ChromRunbook(1, cdim), MSContextEncoder(context_dim=cdim))

    examples = _make_examples(4)
    examples[1].energy = float("nan")
    examples[3].energy = float("nan")  # 2 present, 2 masked
    ds = RealSpeclibDataset(examples)
    gen = torch.Generator().manual_seed(0)
    batch = next(iter(ds.batches(4, False, gen)))
    logged = []
    module.log_dict = lambda values, **_kwargs: logged.append(values)

    module.on_train_epoch_start()
    assert module._energy_masked == 0
    assert module._energy_present == 0

    module.training_step(batch, 0)
    module.training_step(batch, 1)  # a second step in the same epoch: counts accumulate
    assert all("train_spectral_angle" in values for values in logged)
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


def test_fit_from_prebuilt_datasets_trains_and_reports_metrics(tmp_path, capsys):
    from pepdistill.distill.context_regime import RealSpeclibDataset

    model = build_student("small")
    model.set_norm(rt_mean=0.0, rt_std=1.0)
    cdim = model.cfg.context_dim
    examples = _make_examples(8)  # existing helper in this test module
    mirrored: list[str] = []
    module = fit_realspeclib_datasets(
        model,
        RealSpeclibDataset(examples),
        RealSpeclibDataset(examples[:2]),
        runbook=ChromRunbook(n_datasets=1, context_dim=cdim),
        dataset_index={"pool": 1},
        encoder=MSContextEncoder(context_dim=cdim),
        epochs=1,
        batch_size=4,
        progress_log_every=1,
        enable_progress_bar=False,
        progress_metrics_path=tmp_path / "train_metrics.jsonl",
        checkpoint_dir=tmp_path,
        artifact_mirror=lambda path: mirrored.append(path.name),
        val_check_interval=timedelta(hours=1),
        check_val_every_n_epoch=None,
    )
    assert module.dataset_index == {"pool": 1}
    metrics = module.trainer.callback_metrics
    assert "val/pool/spectral_angle" in metrics
    assert "val/pool/irt_mae" in metrics
    assert "val/pool/rawrt_mae" in metrics
    assert metrics["val/pool/n"] == 2
    assert "val_spectral_angle" not in metrics
    assert "latest.ckpt" in mirrored
    assert "best.ckpt" in mirrored
    assert "train_metrics.jsonl" in mirrored
    assert module.trainer._val_check_time_interval == 3600.0
    assert module.trainer.check_val_every_n_epoch is None
    checkpoint = torch.load(tmp_path / "best.ckpt", weights_only=False)
    assert checkpoint["training"]["global_step"] == 2
    assert checkpoint["training"]["checkpoint_kind"] == "best"
    validation = checkpoint["training"]["validation"]
    assert validation["validated_at_step"] == 2
    assert validation["values"]["val/pool/spectral_angle"] == pytest.approx(
        float(metrics["val/pool/spectral_angle"])
    )
    assert validation["mean"] == pytest.approx(float(metrics["val/pool/spectral_angle"]))
    records = [
        json.loads(line) for line in (tmp_path / "train_metrics.jsonl").read_text().splitlines()
    ]
    assert records[-1]["validation_check"] == 1
    assert records[-1]["global_step"] == 2
    # The per-dataset spectral-angle distribution is retained along with its mean. A mean cannot be
    # drawn against the published teacher yardstick or the corpus replicate ceiling, and all three
    # share this grid so they overlay directly.
    from pepdistill.diagnostics import SA_HISTOGRAM_EDGES

    edges = records[-1]["val_sa_histogram_bin_edges"]
    assert edges == list(SA_HISTOGRAM_EDGES)
    counts = records[-1]["val_sa_histogram"]["pool"]
    assert len(counts) == len(edges) - 1
    # Every validation row is counted exactly once, and the mean recovered from the histogram
    # agrees with the logged scalar to within one bin width.
    assert sum(counts) == int(metrics["val/pool/n"])
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    recovered = sum(c * center for c, center in zip(counts, centers)) / sum(counts)
    assert recovered == pytest.approx(float(metrics["val/pool/spectral_angle"]), abs=0.02)
    progress = capsys.readouterr().out
    assert "approximately 2 batches" in progress
    assert "batch 1/~2 (50.0%)" in progress
    assert "examples/s" in progress
    assert "epoch_eta=" in progress


def test_validation_early_stop_fails_on_missing_metric_key():
    class Trainer:
        callback_metrics = {"val/other/spectral_angle": torch.tensor(0.5)}
        sanity_checking = False

    callback = _RealValidationEarlyStop(
        patience=5, min_delta=1e-3, expected_keys={"val/pool/spectral_angle"}
    )
    with pytest.raises(RuntimeError, match="missing=.*val/pool/spectral_angle"):
        callback.on_validation_epoch_end(Trainer(), None)


def test_validation_early_stop_ignores_sanity_check_missing_keys():
    class Trainer:
        callback_metrics = {"val/other/spectral_angle": torch.tensor(0.5)}
        sanity_checking = True

    callback = _RealValidationEarlyStop(
        patience=5, min_delta=1e-3, expected_keys={"val/pool/spectral_angle"}
    )
    callback.on_validation_epoch_end(Trainer(), None)
    assert callback.bad == 0


def test_validation_early_stop_reports_current_and_best_each_check():
    lines = []

    class Trainer:
        callback_metrics = {"val/pool/spectral_angle": torch.tensor(0.5)}
        sanity_checking = False
        current_epoch = 2
        global_step = 123
        print = lines.append
        should_stop = False

    callback = _RealValidationEarlyStop(
        patience=5, min_delta=1e-3, expected_keys={"val/pool/spectral_angle"}
    )
    callback.on_validation_epoch_end(Trainer(), None)
    assert lines == [
        "[early-stop] validation check at epoch 3, step 123: mean spectral agreement current=0.5000, "
        "best=0.5000, bad=0/5 (new best)"
    ]


def test_validation_early_stop_treats_higher_agreement_as_better():
    lines = []

    class Trainer:
        callback_metrics = {"val/pool/spectral_angle": torch.tensor(0.5)}
        sanity_checking = False
        current_epoch = 0
        global_step = 1
        print = lines.append
        should_stop = False

    callback = _RealValidationEarlyStop(
        patience=2, min_delta=1e-3, expected_keys={"val/pool/spectral_angle"}
    )
    callback.on_validation_epoch_end(Trainer(), None)
    Trainer.current_epoch = 1
    Trainer.callback_metrics = {"val/pool/spectral_angle": torch.tensor(0.6)}
    callback.on_validation_epoch_end(Trainer(), None)
    assert callback.best == pytest.approx(0.6)
    assert callback.bad == 0


def test_every_epoch_gets_a_validation_when_the_interval_never_elapses():
    """One boundary check per epoch, not one per two epochs, and not none at all.

    A timed `val_check_interval` is the only trigger a prepared streaming run has, so an epoch
    shorter than the interval used to end with nothing validated. The forced check runs on the
    first batch of the FOLLOWING epoch, so it has to be credited to the epoch that asked for it;
    crediting it to the epoch it lands in makes one check serve two epochs, halving the
    validation rate and the effective early-stop patience.

    Driven through a real Trainer on purpose: the trigger leans on Lightning's timed-interval
    internals, so a stub trainer would assert nothing about whether validation actually happens.
    """
    import lightning as L

    from pepdistill.distill.context_regime import RealSpeclibDataset

    checks: list[int] = []

    class Spy(L.Callback):
        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            # `fit_realspeclib_datasets` runs a closing `trainer.validate` outside fit; only the
            # in-fit checks say anything about the boundary rule.
            if trainer.state.fn == "fit":
                checks.append(trainer.global_step)

    model = build_student("small")
    model.set_norm(rt_mean=0.0, rt_std=1.0)
    cdim = model.cfg.context_dim
    examples = _make_examples(8)
    fit_realspeclib_datasets(
        model,
        RealSpeclibDataset(examples),
        RealSpeclibDataset(examples[:2]),
        runbook=ChromRunbook(n_datasets=1, context_dim=cdim),
        dataset_index={"pool": 1},
        encoder=MSContextEncoder(context_dim=cdim),
        epochs=3,
        batch_size=4,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        val_check_interval=timedelta(hours=1),
        check_val_every_n_epoch=None,
        callbacks=[Spy()],
    )
    # Epochs 1 and 2 each force a check, which lands on the first batch of the epoch after it.
    # Epoch 3's has no following batch and is covered by the closing validate, which the spy
    # deliberately ignores.
    assert len(checks) == 2, f"expected one check per epoch boundary, got steps {checks}"
    assert checks[0] < checks[1]


def test_epoch_shorter_than_the_validation_interval_still_checkpoints(tmp_path):
    """The end-of-epoch snapshot must survive an epoch in which nothing was validated.

    The real pipeline runs with `num_sanity_val_steps=0` and a wall-clock validation interval, so
    a first epoch shorter than that interval reaches its epoch boundary with
    `last_validation_step` still unset. Recording it as an int crashed there; losing the whole
    epoch, since this callback is what writes `latest.ckpt`. Every other test leaves sanity
    checking on, which sets the attribute as a side effect and hid this.
    """
    from pepdistill.distill.context_regime import RealSpeclibDataset

    model = build_student("small")
    model.set_norm(rt_mean=0.0, rt_std=1.0)
    cdim = model.cfg.context_dim
    examples = _make_examples(8)
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
        checkpoint_dir=tmp_path,
        # Exactly the real pipeline's configuration: no sanity pass, and a validation interval
        # far longer than this epoch takes.
        num_sanity_val_steps=0,
        val_check_interval=timedelta(hours=1),
        check_val_every_n_epoch=None,
    )
    # Reaching here at all is the regression: the epoch boundary used to raise.
    assert (tmp_path / "latest.ckpt").exists()

    # Pin what the epoch-end snapshot records when nothing has validated. Driven directly,
    # because the post-fit validation pass overwrites `latest.ckpt` with a real step.
    from pepdistill.distill.context_regime import _RealCheckpoint

    module.last_validation_step = None
    directory = tmp_path / "unvalidated"
    keys = {"val/pool/spectral_angle", "val/pool/irt_mae", "val/pool/rawrt_mae"}
    _RealCheckpoint(directory, expected_keys=keys).on_train_epoch_end(module.trainer, module)
    checkpoint = torch.load(directory / "latest.ckpt", weights_only=False)
    # None records "never validated" rather than claiming a step no validation produced.
    assert checkpoint["training"]["validation"]["validated_at_step"] is None

    # A run with no validation datasets has no expected keys, so the count matches at zero.
    # where a mean does not exist and dividing by it raised.
    bare = tmp_path / "no-val-datasets"
    _RealCheckpoint(bare, expected_keys=set()).on_train_epoch_end(module.trainer, module)
    assert (
        torch.load(bare / "latest.ckpt", weights_only=False)["training"]["validation"]["mean"]
        is None
    )


def _module_with_names(names: dict[str, int]) -> RealSpeclibModule:
    model = build_student("flash")
    cdim = model.cfg.context_dim
    model.set_norm(rt_mean=0.0, rt_std=1.0)
    return RealSpeclibModule(
        model,
        ChromRunbook(len(names), cdim),
        MSContextEncoder(context_dim=cdim),
        dataset_index=names,
    )


def _batch_missing_irt(rows: int, unlabeled: int) -> tuple[RealSpeclibModule, object]:
    """A batch where ``unlabeled`` of ``rows`` report a raw retention time and no iRT; the
    shape a spectral library arrives in."""
    examples = _make_examples(rows)
    for example in examples[:unlabeled]:
        example.label = replace(example.label, rt=float("nan"))
    ds = RealSpeclibDataset(examples)
    gen = torch.Generator().manual_seed(0)
    return _module_with_names({"rfA": 1}), next(iter(ds.batches(rows, False, gen)))


def test_a_row_without_irt_trains_its_other_heads():
    """The whole point of masking per row: the unlabeled row must still supervise MS2 and raw
    RT, and must not put a NaN into any shared gradient on its way there."""
    module, batch = _batch_missing_irt(rows=4, unlabeled=1)
    module.log_dict = lambda values, **_kwargs: None
    module.on_train_epoch_start()

    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_a_batch_with_no_irt_at_all_still_trains():
    """A library-only batch. Its iRT term is zero, so the run keeps going on MS2 and raw RT
    rather than early-stopping on a NaN it never had labels to avoid."""
    module, batch = _batch_missing_irt(rows=4, unlabeled=4)
    logged: list[dict] = []
    module.log_dict = lambda values, **_kwargs: logged.append(values)
    module.on_train_epoch_start()

    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert float(logged[-1]["train_irt"].detach()) == 0.0
    assert float(logged[-1]["train_irt_labeled_fraction"]) == 0.0


def test_validation_reports_no_irt_mae_for_a_source_that_has_no_irt():
    """A NaN mean is worse than a missing series: it poisons the chart and any metric watching
    it. Skip the key for the source that cannot supply it, keep the one it can."""
    module, batch = _batch_missing_irt(rows=4, unlabeled=4)
    module._trainer = SimpleNamespace(sanity_checking=False)
    logged: dict[str, float] = {}

    def record(name, value, **_kwargs):
        logged[name] = float(value.detach()) if torch.is_tensor(value) else float(value)

    module.log = record

    module.on_validation_epoch_start()
    module.validation_step(batch, 0)

    assert "val/rfA/irt_mae" not in logged
    assert math.isfinite(logged["val/rfA/rawrt_mae"])
    assert math.isfinite(logged["val/rfA/spectral_angle"])


class _DecayTrainer:
    """Just enough Trainer for the plateau callbacks: the metrics of one check, and one
    optimizer whose rate they are allowed to change."""

    sanity_checking = False
    current_epoch = 0
    global_step = 0
    should_stop = False

    def __init__(self, agreement: float, lr: float = 1e-3) -> None:
        self.callback_metrics = {"val/pool/spectral_angle": torch.tensor(agreement)}
        self.optimizers = [torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=lr)]
        self.lines: list[str] = []
        self.print = self.lines.append

    def check(self, callback, agreement: float) -> None:
        self.callback_metrics = {"val/pool/spectral_angle": torch.tensor(agreement)}
        callback.on_validation_epoch_end(self, None)

    @property
    def lr(self) -> float:
        return float(self.optimizers[0].param_groups[0]["lr"])


def _decay(patience: int = 2, factor: float = 0.5, min_lr: float = 0.0) -> _RealPlateauDecay:
    return _RealPlateauDecay(
        patience=patience,
        factor=factor,
        min_lr=min_lr,
        min_delta=1e-3,
        expected_keys={"val/pool/spectral_angle"},
    )


def test_a_plateau_cuts_the_rate_and_an_improvement_holds_it():
    callback, trainer = _decay(patience=2), _DecayTrainer(0.5)

    trainer.check(callback, 0.50)  # first check: a new best, nothing to decay from
    trainer.check(callback, 0.50)  # flat 1 of 2
    assert trainer.lr == pytest.approx(1e-3)
    trainer.check(callback, 0.50)  # flat 2 of 2 -> cut
    assert trainer.lr == pytest.approx(5e-4)
    assert callback.bad == 0  # the counter resets so the next cut needs another plateau
    assert "[lr-decay]" in trainer.lines[-1]

    trainer.check(callback, 0.60)  # the smaller rate found something: hold it
    trainer.check(callback, 0.60)
    assert trainer.lr == pytest.approx(5e-4)


def test_a_later_plateau_is_judged_against_the_best_agreement_not_the_plateau():
    """Keeping `best` across a cut is what stops a slowly sagging run from reading as progress:
    each new cut has to beat the best ever seen, not whatever the last plateau settled at."""
    callback, trainer = _decay(patience=1), _DecayTrainer(0.5)

    trainer.check(callback, 0.60)  # best = 0.60
    trainer.check(callback, 0.50)  # worse -> cut
    assert trainer.lr == pytest.approx(5e-4)
    trainer.check(callback, 0.55)  # better than the plateau, still below best -> cut again
    assert trainer.lr == pytest.approx(2.5e-4)
    assert callback.best == pytest.approx(0.60)


def test_the_rate_stops_at_its_floor_without_reporting_a_cut():
    callback, trainer = _decay(patience=1, min_lr=6e-4), _DecayTrainer(0.5)

    trainer.check(callback, 0.50)
    trainer.check(callback, 0.50)
    assert trainer.lr == pytest.approx(6e-4)  # clamped, not 5e-4
    cuts = len([line for line in trainer.lines if "[lr-decay]" in line])

    trainer.check(callback, 0.50)  # nothing left to give: silence, and early stop takes over
    assert trainer.lr == pytest.approx(6e-4)
    assert len([line for line in trainer.lines if "[lr-decay]" in line]) == cuts


def test_an_incomplete_check_is_left_to_early_stopping_to_report():
    """The decay must not count a check it could not read as a plateau; early stopping raises
    on the same check, and a rate cut on the way there would be noise in the traceback."""
    callback = _decay(patience=1)
    trainer = _DecayTrainer(0.5)
    trainer.callback_metrics = {"val/other/spectral_angle": torch.tensor(0.5)}

    callback.on_validation_epoch_end(trainer, None)
    callback.on_validation_epoch_end(trainer, None)

    assert callback.bad == 0
    assert trainer.lr == pytest.approx(1e-3)
