"""End-to-end: digest -> fake teacher labels -> Lightning fit -> predict library; run CLI."""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from pepdistill.chem import ION_TYPES
from pepdistill.cli import app
from pepdistill.data.config import DigestConfig, SplitConfig
from pepdistill.data.digest import digest_fasta
from pepdistill.data.precursors import enumerate_precursors
from pepdistill.data.prospect import RECORDS
from pepdistill.data.split import assign_split
from pepdistill.distill.dataset import DistillDataset, collate_with_labels
from pepdistill.distill.lightning import fit_distill
from pepdistill.distill.losses import spectral_angle
from pepdistill.distill.pipeline import RunConfig, run_pipeline
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
    assert (workdir / "summary.json").exists()

    from pepdistill.models.registry import load_context

    ctx = load_context(workdir / "model.ckpt")
    assert ctx is not None and ctx.encoder is not None


# --- streaming multi-pool train stage -------------------------------------------------------
#
# Offline by construction: a synthetic meta parquet and a two-member annotation zip are seeded
# into a tmp_path cache under REAL 'prospect' catalog filenames, so every resolve_file hits the
# local tier and nothing reaches Zenodo.

_POOL_META = "TUM_third_pool_meta_data.parquet"
_POOL_ZIP = "TUM_third_pool.zip"
_RUN_A = "01812a_GA1-poolA-DDA-1h-R1"
_RUN_B = "01812a_GB1-poolB-DDA-1h-R1"

# (sequence, split it must hash to). The splits are asserted against assign_split when the
# fixture is built: if the salt or the ratios ever move, this test goes red instead of quietly
# becoming vacuous (an empty val set, or a val_only pool with nothing in val).
_POOL_A = [
    ("AAPEPTIDEK", "train"),
    ("AAAPEPTIDEK", "train"),
    ("AAAAPEPTIDEK", "train"),
    ("AAAAAPEPTIDEK", "train"),
    ("AAAAAAPEPTIDEK", "train"),
    ("AAAAAAAPEPTIDEK", "train"),
    ("ACPEPTIDEK", "train"),
    ("AACPEPTIDEK", "train"),
    ("AEPEPTIDEK", "test"),
    ("AAAAAAAAPEPTIDEK", "val"),
    ("AAAAACPEPTIDEK", "val"),
]
_POOL_B = [
    ("AAAAAEPEPTIDEK", "val"),
    ("AFPEPTIDEK", "val"),
    ("ADPEPTIDEK", "train"),
    ("AAACPEPTIDEK", "train"),
]

# iRT per split, chosen so any pool/split that must NOT reach the RT affine is far away from
# the ones that must: a wrong population moves the mean by hundreds, not by rounding.
_IRT = {"train": 10.0, "test": 500.0, "val": 200.0}
_POOL_B_IRT_BASE = 900.0


def _pool_frames(raw_file, entries, first_scan, irt_of):
    """(meta rows, fragment rows) for one raw file."""
    meta, frags = [], []
    for i, (seq, split) in enumerate(entries):
        assert assign_split(seq, SplitConfig()) == split, (
            f"fixture assumes {seq!r} hashes to {split!r}, got "
            f"{assign_split(seq, SplitConfig())!r}"
        )
        scan = first_scan + i
        meta.append(
            {
                "raw_file": raw_file,
                "scan_number": scan,
                "modified_sequence": seq,
                "precursor_charge": 2,
                "retention_time": 30.0 + i,
                "indexed_retention_time": irt_of(i, split),
                "aligned_collision_energy": 0.28 + 0.001 * i,
                "mass_analyzer": "FTMS",
                "fragmentation": "HCD",
                "andromeda_score": 100.0 + i,
            }
        )
        for ordinal, intensity in ((1, 0.9), (2, 0.5), (3, 0.3), (4, 0.2)):
            frags.append((raw_file, scan, "b", ordinal, 1, intensity, ""))
        frags.append((raw_file, scan, "y", 1, 1, 0.7, ""))
        frags.append((raw_file, scan, "y", 2, 1, 0.4, "H2O"))  # filtered out
    return meta, frags


def _seed_two_synthetic_pools(tmp_path, monkeypatch):
    """Seed the cache with one meta parquet and a two-shard annotation zip.

    Returns what the run must produce from it: the iRT mean of the RT affine (poolA's train
    rows and nothing else) and the number of examples the train stream must yield (poolA's
    train AND test rows; poolB is val_only)."""
    monkeypatch.setenv("PEPDISTILL_CACHE", str(tmp_path))
    root = tmp_path / "zenodo" / RECORDS["prospect"]
    os.makedirs(root, exist_ok=True)

    meta_a, frags_a = _pool_frames(_RUN_A, _POOL_A, 1, lambda i, s: _IRT[s] + i)
    meta_b, frags_b = _pool_frames(_RUN_B, _POOL_B, 100, lambda i, s: _POOL_B_IRT_BASE + i)
    pd.DataFrame(meta_a + meta_b).to_parquet(root / _POOL_META)

    cols = ["raw_file", "scan_number", "ion_type", "no", "charge", "intensity", "neutral_loss"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for member, rows in (
            ("TUM_third_pool/poolA_annotation.parquet", frags_a),
            ("TUM_third_pool/poolB_annotation.parquet", frags_b),
        ):
            b = io.BytesIO()
            pd.DataFrame(rows, columns=cols).to_parquet(b)
            z.writestr(member, b.getvalue())
    (root / _POOL_ZIP).write_bytes(buf.getvalue())

    train_irt = [m["indexed_retention_time"] for m, (_, s) in zip(meta_a, _POOL_A) if s == "train"]
    assert train_irt, "fixture must leave poolA some train rows"
    assert any(s == "val" for _, s in _POOL_B), "the val_only pool must own some val rows"
    return {
        "rt_mean": sum(train_irt) / len(train_irt),
        "n_train": sum(1 for _, s in _POOL_A if s in ("train", "test")),
        "n_val": sum(1 for _, s in _POOL_A + _POOL_B if s == "val"),
    }


def test_two_source_streaming_train_stage_runs_end_to_end(tmp_path, monkeypatch):
    """Two synthetic pools, one of them val_only, through the real train stage."""
    expected = _seed_two_synthetic_pools(tmp_path, monkeypatch)
    out = tmp_path / "out"
    toml = tmp_path / "run.toml"
    toml.write_text(f"""
out = "{out}"
preset = "small"
device = "cpu"

[pretrain]
enabled = false

[train]
enabled = true
epochs = 1
batch_size = 4
shuffle_buffer = 0

[[train.sources]]
record = "prospect"
meta = "{_POOL_META}"
zip = "{_POOL_ZIP}"
shards = [0]
dataset = "poolA"

[[train.sources]]
record = "prospect"
meta = "{_POOL_META}"
zip = "{_POOL_ZIP}"
shards = [1]
dataset = "poolB"
val_only = true
""")
    logs: list[str] = []
    summary = run_pipeline(RunConfig.from_toml(toml), log=logs.append)

    assert summary["dataset_index"] == {"poolA": 1, "poolB": 2}
    assert "train" in summary
    assert "val_spectral_angle" in summary["train"]  # val ran

    # Exactly poolA's train AND test rows were streamed: the val_only pool is held out of
    # training, and the stream takes test as well as train. train_energy_present counts every
    # example the epoch actually saw (every fixture row carries a collision energy).
    assert summary["train"]["train_energy_present"] == expected["n_train"]
    assert summary["train"]["train_energy_masked"] == 0
    # Val winners come from BOTH pools, one per (dataset, modified_sequence, charge).
    assert f"val {expected['n_val']} examples" in "\n".join(logs)

    # The RT affine is established from the non-val_only pool's TRAIN rows only: poolB is
    # val_only (iRT ~900) and poolA's test row sits at 500, so either leaking in would move
    # the mean far outside this tolerance.
    model = load_checkpoint(out / "model.ckpt")
    assert float(model.rt_mean) == pytest.approx(expected["rt_mean"])
