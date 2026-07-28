"""Vectorized predictor matches the reference, and ONNX round-trips."""

import numpy as np
import pytest

from pepdistill.chem import Peptide
from pepdistill.data.precursors import Precursor
from pepdistill.models.registry import build_student
from pepdistill.predict.fast import TorchRunner, predict_library_fast
from pepdistill.predict.library import predict_library

KEY = ["modified_sequence", "charge", "ion_type", "fragment_charge", "fragment_ordinal"]


def _precs():
    seqs = ["SAMPLER", "ACDEMKPEPTIDEK", "MYK", "AACDEFGHIKLMNR", "SAMPLERR"]
    out = []
    for s in seqs:
        mods = tuple((i, "Carbamidomethyl@C") for i, a in enumerate(s) if a == "C")
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
    from pepdistill.models.context import MSContextEncoder
    from pepdistill.models.registry import build_student
    import torch

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


def test_onnx_roundtrip(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from pepdistill.predict.onnx import OnnxRunner, export_onnx

    # CNN exports with fully dynamic axes.
    model = build_student("tiny")
    path = tmp_path / "m.onnx"
    export_onnx(model, path)
    precs = _precs()
    ref = (
        predict_library_fast(TorchRunner(model, "cpu"), precs, min_intensity=0.05)
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    onx = (
        predict_library_fast(OnnxRunner(path), precs, min_intensity=0.05)
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    assert len(ref) == len(onx)
    assert np.abs(ref["relative_intensity"].values - onx["relative_intensity"].values).max() < 1e-3
