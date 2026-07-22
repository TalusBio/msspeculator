"""End-to-end: digest -> fake teacher labels -> train -> predict library."""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from pepdistill.chem import ION_TYPES
from pepdistill.cli import app
from pepdistill.data.config import DigestConfig, SplitConfig
from pepdistill.data.digest import digest_fasta
from pepdistill.data.precursors import enumerate_precursors
from pepdistill.distill.dataset import DistillDataset
from pepdistill.distill.trainer import TrainConfig, evaluate, train
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
    peptides = digest_fasta(fasta_path, dcfg)
    precs = enumerate_precursors(peptides, dcfg, scfg)
    if split_filter:
        precs = [p for p in precs if p.split == split_filter]
    labels = FakeTeacher().predict(precs)
    return DistillDataset(precs, labels)


def test_training_reduces_loss_and_predicts(tmp_path: Path):
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)

    train_ds = _make_dataset(fasta, "train")
    val_ds = _make_dataset(fasta, "val")
    assert len(train_ds) > 0

    cfg = TrainConfig(epochs=80, batch_size=32, lr=2e-3, seed=1)
    baseline = evaluate(build_student("tiny"), train_ds, cfg)["spectral_angle"]

    model = build_student("tiny")
    history = train(model, train_ds, val_ds if len(val_ds) else None, cfg)

    assert history[-1]["train_total"] < history[0]["train_total"]

    metrics = evaluate(model, train_ds, cfg)
    # Training should fit the (deterministic) teacher targets well above an untrained net.
    assert metrics["spectral_angle"] > 0.75
    assert metrics["spectral_angle"] > baseline + 0.1

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


def test_cli_pipeline_smoke(tmp_path: Path):
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)
    workdir = tmp_path / "work"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["pipeline", str(fasta), "--workdir", str(workdir), "--teacher", "fake", "--epochs", "3"],
    )
    assert result.exit_code == 0, result.output
    assert (workdir / "model.ckpt").exists()
    lib = pd.read_parquet(workdir / "library.parquet")
    assert len(lib) > 0
