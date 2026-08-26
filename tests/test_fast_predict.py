"""Vectorized predictor matches the reference."""

import numpy as np
import torch

from msspeculator.chem import Peptide
from msspeculator.data.precursors import Precursor
from msspeculator.models.registry import build_student
from msspeculator.predict.fast import TorchRunner, predict_library_fast
from msspeculator.predict.library import predict_library

KEY = ["modified_sequence", "charge", "ion_type", "fragment_charge", "fragment_ordinal"]


def _precs():
    seqs = ["SAMPLER", "ACDEMKPEPTIDEK", "MYK", "AACDEFGHIKLMNR", "SAMPLERR"]
    out = []
    for s in seqs:
        mods = tuple((i, "UNIMOD:4") for i, a in enumerate(s) if a == "C")
        for z in (2, 3):
            out.append(Precursor(Peptide(s, mods), z, "train"))
    return out


def test_fast_matches_reference():
    model = build_student("flash")
    precs = _precs()
    ref = predict_library(model, precs, min_intensity=0.05).sort_values(KEY).reset_index(drop=True)
    fast = (
        predict_library_fast(TorchRunner(model, "cpu"), precs, min_intensity=0.05)
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    assert len(ref) == len(fast)
    for c in KEY:
        assert (ref[c].values == fast[c].values).all(), c
    for c in ["precursor_mz", "fragment_mz", "rt_pred", "ccs_pred", "relative_intensity"]:
        assert np.abs(ref[c].values - fast[c].values).max() < 1e-3, c


def test_fast_empty_on_high_threshold():
    model = build_student("flash")
    lib = predict_library_fast(TorchRunner(model, "cpu"), _precs(), min_intensity=1.1)
    assert len(lib) == 0
    assert list(lib.columns)  # still has the schema


def test_torch_runner_stores_ms_context(tmp_path):
    from msspeculator.models.context import MSContextEncoder

    m = build_student("small")
    enc = MSContextEncoder(context_dim=m.cfg.context_dim)
    torch.nn.init.normal_(enc.frag_emb.weight, std=0.3)
    ctx = (
        enc(
            torch.tensor([enc.instrument_id("Lumos")]),
            torch.tensor([enc.detector_id("FTMS")]),
            torch.tensor([enc.fragmentation_id("HCD")]),
            torch.tensor([30.0]),
        )
        .detach()
        .numpy()[0]
    )
    runner = TorchRunner(m, torch.device("cpu"), ms_context=ctx)
    assert isinstance(runner.ms_context, np.ndarray)


def _checkpoint_with_a_named_setup(tmp_path):
    """A checkpoint carrying one fitted acquisition setup, weights randomized so the row is not
    the neutral one."""
    from msspeculator.models.context import ChromRunbook, MSContextEncoder
    from msspeculator.models.registry import save_checkpoint

    torch.manual_seed(0)
    model = build_student("flash")
    encoder = MSContextEncoder(model.cfg.context_dim)
    encoder.ensure_setups(["Evosep60SPD_heron"])
    torch.nn.init.normal_(encoder.setup_emb.weight, std=0.3)
    model.set_norm(31.0, 4.0, 410.0, 25.0)
    path = tmp_path / "m.ckpt"
    save_checkpoint(
        model,
        path,
        encoder=encoder,
        runbook=ChromRunbook(1, model.cfg.context_dim),
        dataset_index={"ds": 1},
    )
    return path, encoder


def test_predict_takes_a_named_setup_as_ms_context(tmp_path):
    """`--ms-context` accepts a bare setup name in the torch CLI too, or the two runtimes would
    disagree about what the same flag means."""
    from typer.testing import CliRunner

    from msspeculator.cli import app

    path, _ = _checkpoint_with_a_named_setup(tmp_path)
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">p\nPEPTIDEK\n")

    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model",
            str(path),
            "--fasta",
            str(fasta),
            "-o",
            str(tmp_path / "lib.parquet"),
            "--ms-context",
            "Evosep60SPD_heron",
        ],
    )
    assert result.exit_code == 0, result.output
    # The row is randomized, so a neutral context would report a zero-length vector.
    assert "context-aware: Evosep60SPD_heron -> ms_context |v|=" in result.output
    assert "|v|=0.000" not in result.output


def test_predict_refuses_a_setup_the_checkpoint_has_no_row_for(tmp_path):
    from typer.testing import CliRunner

    from msspeculator.cli import app

    path, _ = _checkpoint_with_a_named_setup(tmp_path)
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">p\nPEPTIDEK\n")

    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model",
            str(path),
            "--fasta",
            str(fasta),
            "-o",
            str(tmp_path / "lib.parquet"),
            "--ms-context",
            "never_fitted",
        ],
    )
    assert result.exit_code != 0
    assert "never_fitted" in result.output and "Evosep60SPD_heron" in result.output
