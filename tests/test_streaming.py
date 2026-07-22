"""Streaming sources + online distillation with the fake teacher."""

import numpy as np

from pepdistill.data.config import DigestConfig
from pepdistill.data.sources import (
    fasta_peptide_stream,
    precursors_from_sequences,
    random_peptide_stream,
)
from pepdistill.distill.streaming import build_val_set, curriculum_batches, estimate_norm
from pepdistill.distill.trainer import TrainConfig, evaluate, train_streaming
from pepdistill.models.registry import build_student
from pepdistill.teacher import FakeTeacher

FASTA = ">a\nSAMPLERPEPTIDEKACDEMKGGGGKLLLLLRTTTTTKVVVVVRNNNNNKQQQQQR\n"


def test_random_stream_shapes():
    rng = np.random.default_rng(0)
    s = random_peptide_stream(rng, 7, 30)
    seqs = [next(s) for _ in range(50)]
    assert all(7 <= len(x) <= 30 for x in seqs)
    assert all(x[-1] in "KR" for x in seqs)


def test_fasta_stream_loops(tmp_path):
    f = tmp_path / "t.fasta"
    f.write_text(FASTA)
    rng = np.random.default_rng(0)
    s = fasta_peptide_stream(f, DigestConfig(), rng, loop=True)
    got = [next(s) for _ in range(200)]  # more than #peptides -> must loop
    assert len(got) == 200


def test_precursors_from_sequences_have_fixed_mods():
    rng = np.random.default_rng(1)
    precs = precursors_from_sequences(["ACDEMK", "SAMPLER"], DigestConfig(), rng)
    assert len(precs) == 2
    acdemk = next(p for p in precs if p.peptide.sequence == "ACDEMK")
    assert any(n == "Carbamidomethyl@C" for _, n in acdemk.peptide.mods)


def test_curriculum_switches_source(tmp_path):
    f = tmp_path / "t.fasta"
    f.write_text(FASTA)
    rng = np.random.default_rng(2)
    teacher = FakeTeacher()
    batches = list(
        curriculum_batches(teacher, DigestConfig(), rng, batch_size=8, total_batches=4,
                            warmup_batches=2, fasta=f)
    )
    assert len(batches) == 4
    assert batches[0].ms2_target.shape[0] == 8


def test_streaming_training_reduces_loss(tmp_path):
    f = tmp_path / "t.fasta"
    f.write_text(FASTA)
    rng = np.random.default_rng(3)
    teacher = FakeTeacher()
    cfg = DigestConfig()

    model = build_student("flash")
    model.set_norm(*estimate_norm(teacher, cfg, rng, n=500))
    val = build_val_set(teacher, cfg, rng, 200, f)

    batches = curriculum_batches(teacher, cfg, rng, 64, total_batches=120, warmup_batches=40, fasta=f)
    tcfg = TrainConfig(batch_size=64, lr=2e-3, seed=0)
    hist = train_streaming(model, batches, 120, tcfg, val_ds=val, eval_every=40)

    assert hist[-1]["train_total"] < hist[0]["train_total"]
    assert evaluate(model, val, tcfg)["spectral_angle"] > 0.5
