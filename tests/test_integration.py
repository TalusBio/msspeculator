"""End-to-end: digest -> fake teacher labels -> Lightning fit -> predict library; run CLI."""

from pathlib import Path

import torch
from typer.testing import CliRunner

from pepdistill.chem import ION_TYPES
from pepdistill.cli import app
from pepdistill.data.config import DigestConfig, SplitConfig
from pepdistill.data.digest import digest_fasta
from pepdistill.data.precursors import enumerate_precursors
from pepdistill.distill.dataset import DistillDataset, collate_with_labels
from pepdistill.distill.lightning import fit_distill
from pepdistill.distill.losses import spectral_angle
from pepdistill.models.registry import build_student, load_checkpoint, save_checkpoint
from pepdistill.predict.library import predict_library
from pepdistill.teacher import FakeTeacher, labels_from_frames, labels_to_frames

FASTA = """>sp|TEST1|
MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPF
>sp|TEST2|
SAMPLERPEPTIDEKACDEMKGGGGKLLLLLRTTTTTKVVVVVRNNNNNKQQQQQR
"""


def _make_dataset(fasta_path, split_filter=None):
    dcfg = DigestConfig()
    scfg = SplitConfig()
    precs = enumerate_precursors(digest_fasta(fasta_path, dcfg), dcfg, scfg)
    if split_filter:
        precs = [p for p in precs if p.split == split_filter]
    return DistillDataset(precs, FakeTeacher().predict(precs))


def _spectral_angle(model, ds) -> float:
    model.eval()
    b = collate_with_labels(ds.precursors, ds.labels)
    with torch.no_grad():
        out = model(b.inputs)
    return float(spectral_angle(out["ms2"], b.ms2_target, b.inputs.frag_mask).mean())


def test_training_reduces_loss_and_predicts(tmp_path: Path):
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)

    train_ds = _make_dataset(fasta, "train")
    val_ds = _make_dataset(fasta, "val")
    assert len(train_ds) > 0

    baseline = _spectral_angle(build_student("tiny"), train_ds)
    model = build_student("tiny")
    fit_distill(
        model,
        train_ds,
        val_ds if len(val_ds) else None,
        epochs=80,
        batch_size=32,
        lr=2e-3,
        seed=1,
        accelerator="cpu",
        enable_progress_bar=False,
    )

    sa = _spectral_angle(model, train_ds)
    assert sa > 0.75  # fits the deterministic teacher targets
    assert sa > baseline + 0.1

    lib = predict_library(model, train_ds.precursors[:5])
    assert len(lib) > 0
    assert set(lib["ion_type"]).issubset({ion for ion, _ in ION_TYPES})
    assert lib["relative_intensity"].max() <= 1.0


def test_label_frame_roundtrip(tmp_path: Path):
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)
    ds = _make_dataset(fasta, "train")
    prec_df, frag_df = labels_to_frames(ds.precursors, ds.labels)
    back = labels_from_frames(prec_df, frag_df)
    assert len(back) == len(ds.labels)
    for a, b in zip(back, ds.labels):
        assert a.ms2.shape == b.ms2.shape
        assert abs(a.rt - b.rt) < 1e-4


def test_checkpoint_roundtrip(tmp_path: Path):
    model = build_student("small")
    model.set_norm(10, 2, 300, 20)
    ckpt = tmp_path / "m.ckpt"
    save_checkpoint(model, ckpt)
    loaded = load_checkpoint(ckpt)
    assert loaded.cfg.d_model == model.cfg.d_model
    assert float(loaded.rt_mean) == 10.0


def test_cli_run_pretrain_only(tmp_path: Path):
    """`run` with only the pretrain stage (fake teacher, train/export/bench off)."""
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)
    workdir = tmp_path / "work"
    config = tmp_path / "run.toml"
    config.write_text(
        f"""
out = "{workdir}"
preset = "tiny"
device = "cpu"

[pretrain]
enabled = true
teacher = "fake"
epochs = 3
[[pretrain.sources]]
fasta = "{fasta}"

[train]
enabled = false
ce_context = false
"""
    )
    result = CliRunner().invoke(app, ["run", str(config)])
    assert result.exit_code == 0, result.output
    assert (workdir / "model.ckpt").exists()
    assert (workdir / "summary.json").exists()
