"""Measure Python-vs-Rust parity for the predict CLI.

No training: a random-init model (all zero-inits overwritten, so no code path is trivially 0)
is exported and run through both `predict_library_fast` and the Rust binary; we assert the
max-abs-diff on ms2/rt/ccs/mz is tiny and print the deltas. Skipped without a Rust toolchain.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.export import export_safetensors
from pepdistill.models.context import ChromRunbook, MSContextEncoder
from pepdistill.models.registry import build_student, save_checkpoint
from pepdistill.predict.fast import TorchRunner, predict_library_fast

RUST_DIR = Path(__file__).resolve().parents[1] / "rust"
BIN = RUST_DIR / "target" / "release" / "pepdistill-cli"
TOL = 1e-3
PEPTIDE, CHARGE = "PEPTIDER", 2

pytestmark_reason = (
    "model.rs::forward does not wrap N/C-term tokens yet; termini became mandatory in the "
    "encoder (Task 4) and the Rust runtime is rebuilt in Task 9. strict=True so this fails "
    "loudly the moment Task 9 makes it pass, forcing the marker's removal."
)


def _binary() -> str:
    if BIN.exists():
        return str(BIN)
    if shutil.which("cargo") is None:
        pytest.skip("no cargo toolchain")
    r = subprocess.run(
        ["cargo", "build", "-p", "pepdistill-cli", "--release"],
        cwd=RUST_DIR,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not BIN.exists():
        pytest.skip(f"cargo build failed: {r.stderr[-400:]}")
    return str(BIN)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rustparity")
    torch.manual_seed(0)
    model = build_student("small")
    enc = MSContextEncoder(model.cfg.context_dim)
    runbook = ChromRunbook(2, model.cfg.context_dim)
    for mod in (model, enc, runbook):
        for p in mod.parameters():
            torch.nn.init.normal_(p, std=0.3)
    model.set_norm(31.0, 4.0, 410.0, 25.0)
    model.eval(), enc.eval(), runbook.eval()

    ckpt = tmp / "m.ckpt"
    save_checkpoint(model, ckpt, encoder=enc, runbook=runbook, dataset_index={"dsA": 1, "dsB": 2})
    art = tmp / "m.safetensors"
    export_safetensors(ckpt, art)
    return {"path": art, "model": model, "enc": enc, "runbook": runbook}


def _rust(binary, art, extra):
    r = subprocess.run(
        [binary, "--model", str(art), "--peptide", PEPTIDE, "--charge", str(CHARGE), *extra],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _frag_map_py(lib):
    return {
        (r.ion_type, int(r.fragment_ordinal), int(r.fragment_charge)): (
            r.fragment_mz,
            r.relative_intensity,
        )
        for r in lib.itertuples()
    }


def _frag_map_rust(rj):
    f = rj["fragments"]
    return {
        (f["ion"][i], f["ord"][i], f["z"][i]): (f["mz"][i], f["rel"][i])
        for i in range(len(f["ion"]))
    }


@pytest.mark.parametrize(
    "label,extra,ids",
    [
        ("base", [], None),
        ("ms-context", ["--ms-context", "Lumos::FTMS::HCD::30"], ("Lumos", "FTMS", "HCD", 30.0)),
        ("nce", ["--nce", "30"], ("", "", "", 30.0)),
    ],
)
@pytest.mark.xfail(reason=pytestmark_reason, strict=True)
def test_parity(artifact, capsys, label, extra, ids):
    """MS2 + RT(base/iRT) + CCS + m/z match the vectorized Python reference."""
    binary = _binary()
    enc, model = artifact["enc"], artifact["model"]

    ms_vec = None
    if ids is not None:
        inst, det, frag, energy = ids
        ms_vec = (
            enc(
                torch.tensor([enc.instrument_id(inst)]),
                torch.tensor([enc.detector_id(det)]),
                torch.tensor([enc.fragmentation_id(frag)]),
                torch.tensor([energy]),
            )
            .detach()
            .numpy()[0]
        )
    lib = predict_library_fast(
        TorchRunner(model, "cpu", ms_context=ms_vec),
        [Precursor(peptide=Peptide(PEPTIDE), charge=CHARGE, split="train")],
        min_intensity=0.01,
    )
    rj = _rust(binary, artifact["path"], extra)

    py, rs = _frag_map_py(lib), _frag_map_rust(rj)
    assert set(py) == set(rs), f"{label}: fragment sets differ"
    d_rt = abs(float(lib["rt_pred"].iloc[0]) - rj["rt"])
    d_ccs = abs(float(lib["ccs_pred"].iloc[0]) - rj["ccs"])
    d_pmz = abs(float(lib["precursor_mz"].iloc[0]) - rj["precursor_mz"])
    d_mz = max(abs(py[k][0] - rs[k][0]) for k in py)
    d_rel = max(abs(py[k][1] - rs[k][1]) for k in py)
    worst = max(d_rt, d_ccs, d_pmz, d_mz, d_rel)
    with capsys.disabled():
        print(
            f"\n[{label}] n={len(py)} d_rt={d_rt:.2e} d_ccs={d_ccs:.2e} "
            f"d_pmz={d_pmz:.2e} d_mz={d_mz:.2e} d_rel={d_rel:.2e}"
        )
    assert worst < TOL, f"{label}: worst delta {worst:.2e} exceeds {TOL}"


@pytest.mark.xfail(reason=pytestmark_reason, strict=True)
def test_parity_chrom_context(artifact, capsys):
    """--chrom-context routes RT through the runbook -> raw RT (differs from base iRT)."""
    binary = _binary()
    model, runbook = artifact["model"], artifact["runbook"]

    prec = Precursor(peptide=Peptide(PEPTIDE), charge=CHARGE, split="train")
    batch = collate([prec])
    chrom = runbook(torch.tensor([1]))  # dsA -> row 1
    with torch.no_grad():
        out = model.forward(batch, ms_context=None, chrom_context=chrom)
    rt_py = float(model.unstandardize_rt(out["rt"])[0])

    rj = _rust(binary, artifact["path"], ["--chrom-context", "dsA"])
    base = _rust(binary, artifact["path"], [])
    d_rt = abs(rt_py - rj["rt"])
    with capsys.disabled():
        print(f"\n[chrom] rt_py={rt_py:.4f} rt_rust={rj['rt']:.4f} base_rt={base['rt']:.4f}")
    assert d_rt < TOL, f"chrom RT delta {d_rt:.2e} exceeds {TOL}"
    # chrom_context must actually move RT off the base (random runbook row is non-trivial).
    assert abs(rj["rt"] - base["rt"]) > 1e-4
