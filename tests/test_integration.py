"""End-to-end teacher warmup and inference tests."""

from pathlib import Path

from typer.testing import CliRunner

from pepdistill.cli import app
from pepdistill.distill.pipeline import RunConfig, run_pipeline
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
"""
    )
    result = CliRunner().invoke(app, ["run", str(config)])
    assert result.exit_code == 0, result.output
    assert (workdir / "model.ckpt").exists()
    assert (workdir / "pretrain.ckpt").exists()
    assert (workdir / "summary.json").exists()

    context = load_checkpoint(workdir / "model.ckpt")
    assert context.cfg.d_model > 0


def test_disabled_train_does_not_require_prepared_prefix(tmp_path: Path):
    config = tmp_path / "run.toml"
    config.write_text(
        f"""
out = "{tmp_path / 'out'}"
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
    assert (remote / "model.ckpt").exists()
    assert (remote / "summary.json").exists()
