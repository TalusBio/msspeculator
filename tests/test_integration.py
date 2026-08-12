"""End-to-end teacher warmup and inference tests."""

from pathlib import Path
import sys
from types import SimpleNamespace

from typer.testing import CliRunner

from pepdistill.cli import app
from pepdistill.distill.pipeline import RunConfig, TrackingCfg, _wandb_loggers, run_pipeline
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
    assert pretrain.LOGGER_JOIN_CHAR == "/"
    assert train.LOGGER_JOIN_CHAR == "/"
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
        ({"train_ms2": 0.3}, 1),
        ({"val/data/spectral_angle": 0.6}, 3),
    ]
    train.finalize("success")
    assert train.logged[-1] == ({"train_ms2": 0.1}, 4)


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
            {"lr-AdamW": 1e-3, "train_ms2": 0.5, "train_total": 0.8},
            50,
        )
    ]
    pretrain.finalize("success")
    assert pretrain.logged[-1] == (
        {"lr-AdamW": 9e-4, "train_ms2": 0.4, "train_total": 0.7},
        100,
    )
