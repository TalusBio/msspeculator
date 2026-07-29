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


def _rust(binary, art, extra, peptide=PEPTIDE):
    r = subprocess.run(
        [binary, "--model", str(art), "--peptide", peptide, "--charge", str(CHARGE), *extra],
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


@pytest.mark.parametrize(
    "label,modseq,mods",
    [
        ("side-chain", "PEPC[Carbamidomethyl@C]IDER", ((3, "Carbamidomethyl@C"),)),
        ("n-terminal", "[TMT6plex]PEPTIDER", (("n", "TMT6plex"),)),
        ("mass-only", "PEP[+42.010565]TIDER", ((2, 42.010565),)),
        (
            "terminal-plus-side-chain",
            "[TMT6plex]PEPC[Carbamidomethyl@C]IDER",
            (("n", "TMT6plex"), (3, "Carbamidomethyl@C")),
        ),
        # Two named mods on ONE site: torch accumulates the compositions and runs comp_enc
        # once, so the site gets ONE comp_enc.bias. Encoding each mod separately and summing
        # the vectors would add the bias twice — a whole-bias-sized error, not a rounding one.
        (
            "co-sited",
            "PEPC[Oxidation@M][Phospho]IDER",
            ((3, "Oxidation@M"), (3, "Phospho")),
        ),
    ],
)
def test_parity_modified_peptides(artifact, capsys, label, modseq, mods):
    """The Rust runtime must encode modifications, not silently predict the bare peptide."""
    binary = _binary()
    model = artifact["model"]

    pep = Peptide("PEPCIDER" if "C[" in modseq else "PEPTIDER", mods)
    assert pep.modified_sequence() == modseq, "test fixture disagrees with the renderer"

    lib = predict_library_fast(
        TorchRunner(model, "cpu"),
        [Precursor(peptide=pep, charge=CHARGE, split="train")],
        min_intensity=0.01,
    )
    rj = _rust(binary, artifact["path"], [], peptide=modseq)
    assert rj["peptide"] == modseq, "the CLI must echo back how it read the input"

    py, rs = _frag_map_py(lib), _frag_map_rust(rj)
    assert set(py) == set(rs), f"{label}: fragment sets differ"
    d_rt = abs(float(lib["rt_pred"].iloc[0]) - rj["rt"])
    d_ccs = abs(float(lib["ccs_pred"].iloc[0]) - rj["ccs"])
    d_pmz = abs(float(lib["precursor_mz"].iloc[0]) - rj["precursor_mz"])
    d_mz = max(abs(py[k][0] - rs[k][0]) for k in py)
    d_rel = max(abs(py[k][1] - rs[k][1]) for k in py)
    worst = max(d_rt, d_ccs, d_pmz, d_mz, d_rel)
    with capsys.disabled():
        print(f"\n[{label}] n={len(py)} worst={worst:.2e}")
    assert worst < TOL, f"{label}: worst delta {worst:.2e} exceeds {TOL}"


def test_modified_peptide_differs_from_the_bare_one(artifact):
    """Guards the parity assertions above: if the mod were dropped on BOTH sides they would
    still agree, and the test would prove nothing."""
    binary = _binary()
    bare = _rust(binary, artifact["path"], [])
    modded = _rust(binary, artifact["path"], [], peptide="[TMT6plex]PEPTIDER")
    assert abs(bare["rt"] - modded["rt"]) > 1e-4
    assert abs(bare["precursor_mz"] - modded["precursor_mz"]) > 1e-4


def test_rust_rejects_an_out_of_range_charge(artifact):
    """An out-of-range charge must be a named error, not an ndarray bounds abort (rc=101)."""
    r = subprocess.run(
        [_binary(), "--model", str(artifact["path"]), "--peptide", PEPTIDE, "--charge", "20"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, f"expected a clean error, got rc={r.returncode}: {r.stderr[-400:]}"
    assert "charge 20" in r.stderr, r.stderr
    assert str(artifact["model"].cfg.max_charge) in r.stderr, r.stderr


def test_rust_refuses_a_site_carrying_a_named_and_a_mass_only_mod(artifact):
    """`comp_enc`/`mass_enc` routing is per column, so the second mod would vanish from the
    model input while still moving every m/z. Both runtimes must refuse instead."""
    modseq = "PEPC[Carbamidomethyl@C][+15.994915]IDER"
    r = subprocess.run(
        [_binary(), "--model", str(artifact["path"]), "--peptide", modseq, "--charge", "2"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, f"expected a clean error, got rc={r.returncode}: {r.stderr[-400:]}"
    assert "Carbamidomethyl@C" in r.stderr and "15.994915" in r.stderr, r.stderr


def test_rust_rejects_a_v1_artifact(artifact, tmp_path):
    """A v1 artifact read as v2 would produce plausible, wrong numbers. It must refuse."""
    import json as _json

    from safetensors import safe_open
    from safetensors.torch import save_file

    src = artifact["path"]
    with safe_open(str(src), framework="pt") as f:
        meta = _json.loads(f.metadata()["pepdistill"])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    meta["format_version"] = 1
    stale = tmp_path / "v1.safetensors"
    save_file(tensors, str(stale), metadata={"pepdistill": _json.dumps(meta)})

    r = subprocess.run(
        [_binary(), "--model", str(stale), "--peptide", PEPTIDE, "--charge", str(CHARGE)],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "format_version" in r.stderr
