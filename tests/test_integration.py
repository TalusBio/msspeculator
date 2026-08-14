"""End-to-end teacher warmup and inference tests."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from pepdistill.cli import app
from pepdistill.distill.pipeline import (
    RunConfig,
    TrackingCfg,
    _final_training_metadata,
    _wandb_loggers,
    _wandb_metric_namespaces,
    run_pipeline,
)
from pepdistill.models.registry import load_checkpoint

FASTA = """>sp|TEST1|
MKWVTFISLLFLFSSAYSRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPF
>sp|TEST2|
SAMPLERPEPTIDEKACDEMKGGGGKLLLLLRTTTTTKVVVVVRNNNNNKQQQQQR
"""


def test_cli_run_pretrain_only(tmp_path: Path):
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)
    workdir = tmp_path / "work"
    config = tmp_path / "run.toml"
    config.write_text(
        f"""
out = "{workdir}"
preset = "flash"
device = "cpu"

[pretrain]
enabled = true
teacher = "fake"
passes = 1
chunk_size = 64
[[pretrain.sources]]
fasta = "{fasta}"

[train]
enabled = false

[diagnostics]
enabled = true
teacher = "fake"
butterflies = 2
every_n_epochs = 0
interval_minutes = 0
"""
    )
    result = CliRunner().invoke(app, ["run", str(config)])
    assert result.exit_code == 0, result.output
    assert (workdir / "model.ckpt").exists()
    assert (workdir / "pretrain.ckpt").exists()
    assert (workdir / "summary.json").exists()
    snapshots = sorted((workdir / "diagnostics").glob("pretrain-step-*"))
    assert len(snapshots) == 2
    assert all((snapshot / "reference-butterflies.png").exists() for snapshot in snapshots)

    context = load_checkpoint(workdir / "model.ckpt")
    assert context.cfg.d_model > 0


def test_disabled_train_does_not_require_prepared_prefix(tmp_path: Path):
    config = tmp_path / "run.toml"
    config.write_text(
        f"""
out = "{tmp_path / "out"}"
preset = "flash"
[pretrain]
enabled = false
[train]
enabled = false
"""
    )
    summary = run_pipeline(RunConfig.from_toml(config), log=lambda *_: None)
    assert "train" not in summary


def test_pipeline_mirrors_durable_outputs(tmp_path: Path):
    local = tmp_path / "out"
    remote = tmp_path / "remote"
    cfg = RunConfig(
        out=str(local),
        remote_output_prefix=remote.as_uri(),
        preset="flash",
    )
    cfg.pretrain.enabled = False
    cfg.train.enabled = False
    summary = run_pipeline(cfg, log=lambda *_: None)
    assert summary["remote_output_prefix"] == remote.as_uri()
    assert summary["artifacts"]["model.ckpt"] == f"{remote.as_uri()}/model.ckpt"
    assert (remote / "model.ckpt").exists()
    assert (remote / "summary.json").exists()


def test_artifact_mirror_preserves_run_relative_paths(tmp_path: Path):
    from pepdistill.distill.pipeline import _artifact_mirror

    local = tmp_path / "out"
    remote = tmp_path / "remote"
    plot = local / "diagnostics" / "epoch-0001" / "irt.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"png")
    mirror = _artifact_mirror(remote.as_uri(), relative_root=local, log=lambda *_: None)

    uri = mirror(plot)

    assert uri.endswith("/diagnostics/epoch-0001/irt.png")
    assert (remote / "diagnostics" / "epoch-0001" / "irt.png").read_bytes() == b"png"


def test_wandb_stage_loggers_share_one_run(tmp_path: Path, monkeypatch):
    class Experiment:
        pass

    class Logger:
        LOGGER_JOIN_CHAR = "-"

        def __init__(self, experiment=None, **kwargs):
            self.experiment = experiment or Experiment()
            self.kwargs = kwargs
            self.logged = []

        def log_metrics(self, metrics, step=None):
            self.logged.append((metrics, step))

        def finalize(self, status):
            self.status = status

    initialized = {}
    experiment = Experiment()
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: initialized.update(kwargs) or experiment,
        ),
    )
    monkeypatch.setattr("lightning.pytorch.loggers.WandbLogger", Logger)
    cfg = RunConfig(
        out=str(tmp_path),
        preset="flash",
        tracking=TrackingCfg(enabled=True, project="pepdistill-tests", mode="offline"),
    )
    root, pretrain, train = _wandb_loggers(cfg, tmp_path)
    assert pretrain.experiment is root.experiment
    assert train.experiment is root.experiment
    assert root.kwargs["log_model"] is False
    assert initialized["config"]["preset"] == "flash"
    assert initialized["mode"] == "offline"

    ticks = iter((0.0, 1.0, 2.0, 11.0, 12.0))
    monkeypatch.setattr("pepdistill.distill.pipeline.time.monotonic", lambda: next(ticks))
    train.log_metrics({"train_ms2": 0.3}, step=1)
    train.log_metrics({"train_ms2": 0.2}, step=2)
    train.log_metrics({"val/data/spectral_angle": 0.6}, step=3)
    train.log_metrics({"train_ms2": 0.1}, step=4)
    assert train.logged == [
        ({"train_metrics/ms2_cosine_loss": 0.3}, 1),
        ({"val_sa/data": 0.6}, 3),
    ]
    train.finalize("success")
    assert train.logged[-1] == ({"train_metrics/ms2_cosine_loss": 0.1}, 4)


def test_rate_limit_thins_batches_but_never_drops_an_epoch_boundary():
    """A per-epoch payload has to survive the batches that follow it milliseconds later.

    The rate limit keeps one pending payload and REPLACES it when a later step arrives. A
    diagnostics render lands right after validation has just flushed, so the interval is never
    due, and the next training batch used to overwrite it — silently costing a whole run every
    diagnostics image and scalar after step 0, while the per-batch losses looked fine.
    """
    from pepdistill.distill.pipeline import _RemoteLogThrottle

    sent: list[tuple[dict, int | None]] = []
    throttle = _RemoteLogThrottle(
        lambda metrics, step: sent.append((dict(metrics), step)),
        min_interval_seconds=10.0,
        max_interval_steps=1000,
        boundary_prefixes=("val_", "train_diagnostics/"),
    )
    clock = 0.0
    for step in range(400):  # an epoch of ordinary training, ~34 steps/s
        throttle.offer({"train_metrics/ms2_cosine_loss": 0.1}, step, clock)
        clock += 0.03
    batches_only = len(sent)

    throttle.offer({"val_sa/pool": 0.77}, 400, clock)
    clock += 0.5
    throttle.offer({"train_diagnostics/spectral_angle_violins": "<Image>"}, 400, clock)
    for step in range(401, 500):
        clock += 0.03
        throttle.offer({"train_metrics/ms2_cosine_loss": 0.1}, step, clock)

    keys = [key for metrics, _ in sent for key in metrics]
    assert "train_diagnostics/spectral_angle_violins" in keys
    assert "val_sa/pool" in keys
    # And it is still a rate limit: 400 batches at 30ms cost far fewer than 400 remote calls.
    assert batches_only < 5


def test_wandb_namespaces_split_validation_and_diagnostics_into_panels():
    metrics = _wandb_metric_namespaces(
        {
            "train_ms2": 0.1,
            "train_spectral_angle": 0.75,
            "val/pool/spectral_angle": 0.8,
            "val/pool/irt_mae": 2.0,
            "val/pool/rawrt_mae": 3.0,
            "val/pool/n": 10,
            "diagnostics/train/butterflies": "image",
        },
        "train",
    )
    assert metrics == {
        "train_metrics/ms2_cosine_loss": 0.1,
        "train_metrics/spectral_angle": 0.75,
        "val_sa/pool": 0.8,
        "val_irt_mae/pool": 2.0,
        "val_rawrt_mae/pool": 3.0,
        "val_n/pool": 10,
        "train_diagnostics/butterflies": "image",
    }


def test_final_checkpoint_metadata_records_early_stop_inputs_and_step():
    module = SimpleNamespace(
        trainer=SimpleNamespace(
            callback_metrics={
                "val/b/spectral_angle": torch.tensor(0.7),
                "val/a/spectral_angle": torch.tensor(0.9),
                "val/a/irt_mae": torch.tensor(2.0),
            },
            global_step=123,
            current_epoch=4,
        ),
        last_validation_step=120,
    )
    metadata = _final_training_metadata(module)
    assert metadata["global_step"] == 123
    assert metadata["validation"] == {
        "metric": "mean_per_dataset_spectral_angle",
        "values": {
            "val/a/spectral_angle": pytest.approx(0.9),
            "val/b/spectral_angle": pytest.approx(0.7),
        },
        "mean": pytest.approx(0.8),
        "validated_at_step": 120,
    }


def test_wandb_throttle_merges_metric_families_at_the_same_step(tmp_path: Path, monkeypatch):
    class Experiment:
        pass

    class Logger:
        LOGGER_JOIN_CHAR = "-"

        def __init__(self, experiment=None, **kwargs):
            self.experiment = experiment or Experiment()
            self.logged = []

        def log_metrics(self, metrics, step=None):
            self.logged.append((metrics, step))

        def finalize(self, status):
            pass

    experiment = Experiment()
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: experiment),
    )
    monkeypatch.setattr("lightning.pytorch.loggers.WandbLogger", Logger)
    cfg = RunConfig(
        out=str(tmp_path),
        preset="flash",
        tracking=TrackingCfg(enabled=True, project="pepdistill-tests", mode="offline"),
    )
    _, pretrain, _ = _wandb_loggers(cfg, tmp_path)
    ticks = iter((0.0, 0.1, 11.0, 11.1, 12.0))
    monkeypatch.setattr("pepdistill.distill.pipeline.time.monotonic", lambda: next(ticks))

    pretrain.log_metrics({"lr-AdamW": 1e-3}, step=50)
    pretrain.log_metrics({"train_ms2": 0.5, "train_total": 0.8}, step=50)
    pretrain.log_metrics({"lr-AdamW": 9e-4}, step=100)
    pretrain.log_metrics({"train_ms2": 0.4, "train_total": 0.7}, step=100)

    assert pretrain.logged == [
        (
            {
                "pretrain_metrics/lr-AdamW": 1e-3,
                "pretrain_metrics/ms2_cosine_loss": 0.5,
                "pretrain_metrics/total_loss": 0.8,
            },
            50,
        )
    ]
    pretrain.finalize("success")
    assert pretrain.logged[-1] == (
        {
            "pretrain_metrics/lr-AdamW": 9e-4,
            "pretrain_metrics/ms2_cosine_loss": 0.4,
            "pretrain_metrics/total_loss": 0.7,
        },
        100,
    )


def test_wandb_throttle_bounds_fast_stage_step_gaps(tmp_path: Path, monkeypatch):
    class Experiment:
        pass

    class Logger:
        LOGGER_JOIN_CHAR = "-"

        def __init__(self, experiment=None, **kwargs):
            self.experiment = experiment or Experiment()
            self.logged = []

        def log_metrics(self, metrics, step=None):
            self.logged.append((metrics, step))

        def finalize(self, status):
            pass

    experiment = Experiment()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: experiment))
    monkeypatch.setattr("lightning.pytorch.loggers.WandbLogger", Logger)
    cfg = RunConfig(
        out=str(tmp_path),
        preset="flash",
        tracking=TrackingCfg(
            enabled=True,
            project="pepdistill-tests",
            mode="offline",
            min_log_interval_seconds=10.0,
            max_log_interval_steps=3,
        ),
    )
    _, pretrain, _ = _wandb_loggers(cfg, tmp_path)
    monkeypatch.setattr("pepdistill.distill.pipeline.time.monotonic", lambda: 0.0)

    for step in range(8):
        pretrain.log_metrics({"lr-AdamW": step / 10}, step=step)
    pretrain.finalize("success")

    assert [step for _, step in pretrain.logged] == [0, 3, 6, 7]
