"""Measure Python-vs-Rust parity for the predict CLI.

No training: a random-init model (all zero-inits overwritten, so no code path is trivially 0)
is exported and run through both `predict_library_fast` and the Rust binary; we assert the
max-abs-diff on ms2/rt/ccs/mz is tiny and print the deltas. Skipped without a Rust toolchain.
"""

import csv
import functools
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import numpy as np
import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import FRAG_OFFSET, collate
from pepdistill.data.precursors import Precursor
from pepdistill.export import export_safetensors
from pepdistill.models.context import ChromRunbook, MSContextEncoder
from pepdistill.models.registry import build_student, save_checkpoint
from pepdistill.predict.fast import TorchRunner, predict_library_fast

RUST_DIR = Path(__file__).resolve().parents[1] / "rust"
BIN = RUST_DIR / "target" / "release" / "pepdistill-cli"
# An acquisition setup addressed by name rather than composed from factors, which is the only
# thing available for a library that records neither instrument nor collision energy.
NAMED_SETUP = "Evosep60SPD_heron"
PRED_ATOL = 1e-3
PRED_RTOL = 2e-5
MZ_ATOL = 1e-7
MZ_RTOL = 1e-9
PEPTIDE, CHARGE = "PEPTIDER", 2


def _assert_close(label, actual, expected, *, atol=PRED_ATOL, rtol=PRED_RTOL):
    """Assert NumPy-style mixed tolerance while naming the output that drifted."""
    np.testing.assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        err_msg=f"{label} outside atol={atol:g}, rtol={rtol:g}",
    )


@functools.cache
def _binary() -> str:
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
    # A named setup before the random init below, so its row is as non-trivial as every other
    # weight: a Rust side that quietly fell back to the neutral row would answer differently.
    enc.ensure_setups([NAMED_SETUP])
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
        [
            binary,
            "predict",
            "--model",
            str(art),
            "--peptide",
            peptide,
            "--charge",
            str(CHARGE),
            *extra,
        ],
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


def test_rust_fasta_generates_diann_tsv(artifact, tmp_path):
    """The production Rust path digests FASTA and emits a timsseek-readable DIA-NN table."""
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">protein_one description\nPEPTIDEM\n")
    out = tmp_path / "library.tsv"
    r = subprocess.run(
        [
            _binary(),
            "library",
            "--model",
            str(artifact["path"]),
            "--fasta",
            str(fasta),
            "--out",
            str(out),
            "--min-length",
            "8",
            "--max-length",
            "8",
            "--min-charge",
            "2",
            "--max-charge",
            "3",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    with out.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert rows
    assert {row["ModifiedPeptide"] for row in rows} == {
        "PEPTIDEM",
        "PEPTIDEM(UniMod:35)",
    }
    assert {row["ProteinID"] for row in rows} == {"protein_one"}
    assert {int(row["PrecursorCharge"]) for row in rows} == {2, 3}
    assert {row["FragmentLossType"] for row in rows} == {"noloss"}
    assert {row["Decoy"] for row in rows} == {"0"}
    assert all(float(row["IonMobility"]) > 0 for row in rows)

    # FASTA mode batches all charge heads into larger GEMMs. Its spectra must still match the
    # independently evaluated single-charge path, not merely produce the expected row count.
    for charge in (2, 3):
        scalar = subprocess.run(
            [
                _binary(),
                "predict",
                "--model",
                str(artifact["path"]),
                "--peptide",
                "PEPTIDEM",
                "--charge",
                str(charge),
            ],
            capture_output=True,
            text=True,
        )
        assert scalar.returncode == 0, scalar.stderr
        scalar_map = _frag_map_rust(json.loads(scalar.stdout))
        batch_map = {
            (
                row["FragmentType"],
                int(row["FragmentNumber"]),
                int(row["FragmentCharge"]),
            ): (float(row["FragmentMz"]), float(row["RelativeIntensity"]))
            for row in rows
            if row["ModifiedPeptide"] == "PEPTIDEM" and int(row["PrecursorCharge"]) == charge
        }
        assert batch_map.keys() == scalar_map.keys()
        for key in batch_map:
            assert batch_map[key] == pytest.approx(scalar_map[key], abs=1e-7)


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
    with capsys.disabled():
        print(
            f"\n[{label}] n={len(py)} d_rt={d_rt:.2e} d_ccs={d_ccs:.2e} "
            f"d_pmz={d_pmz:.2e} d_mz={d_mz:.2e} d_rel={d_rel:.2e}"
        )
    _assert_close(f"{label} RT", float(lib["rt_pred"].iloc[0]), rj["rt"])
    _assert_close(f"{label} CCS", float(lib["ccs_pred"].iloc[0]), rj["ccs"])
    _assert_close(
        f"{label} precursor m/z",
        float(lib["precursor_mz"].iloc[0]),
        rj["precursor_mz"],
        atol=MZ_ATOL,
        rtol=MZ_RTOL,
    )
    _assert_close(
        f"{label} fragment m/z",
        [py[key][0] for key in py],
        [rs[key][0] for key in py],
        atol=MZ_ATOL,
        rtol=MZ_RTOL,
    )
    _assert_close(
        f"{label} relative intensity",
        [py[key][1] for key in py],
        [rs[key][1] for key in py],
    )


def test_parity_chrom_context(artifact, capsys):
    """--chrom-context routes RT through the runbook -> raw RT (differs from base iRT)."""
    binary = _binary()
    model, runbook = artifact["model"], artifact["runbook"]

    prec = Precursor(peptide=Peptide(PEPTIDE), charge=CHARGE, split="train")
    batch = collate([prec])
    ids = torch.tensor([1])  # dsA -> row 1
    # A named dataset supplies BOTH runbook terms, the additive context vector and the output
    # scale+shift. Passing only one here would test half of what the regime trains, and the
    # fixture's normal_(std=0.3) over every parameter makes the affine decidedly non-identity.
    chrom = runbook(ids)
    with torch.no_grad():
        out = model.forward(
            batch, ms_context=None, chrom_context=chrom, chrom_affine=runbook.affine(ids)
        )
    rt_py = float(model.unstandardize_rt(out["rt"])[0])

    rj = _rust(binary, artifact["path"], ["--chrom-context", "dsA"])
    base = _rust(binary, artifact["path"], [])
    with capsys.disabled():
        print(f"\n[chrom] rt_py={rt_py:.4f} rt_rust={rj['rt']:.4f} base_rt={base['rt']:.4f}")
    _assert_close("chrom RT", rt_py, rj["rt"])
    # chrom_context must actually move RT off the base (random runbook row is non-trivial).
    assert abs(rj["rt"] - base["rt"]) > 1e-4


@pytest.mark.parametrize(
    "label,modseq,canonical,mods",
    [
        ("side-chain", "PEPC[UNIMOD:4]IDER", "PEPC[UNIMOD:4]IDER", ((3, "UNIMOD:4"),)),
        ("n-terminal", "[UNIMOD:737]-PEPTIDER", "[UNIMOD:737]-PEPTIDER", (("n", "UNIMOD:737"),)),
        ("mass-only", "PEP[+42.010565]TIDER", "PEP[+42.010565]TIDER", ((2, 42.010565),)),
        (
            "terminal-plus-side-chain",
            "[UNIMOD:737]-PEPC[UNIMOD:4]IDER",
            "[UNIMOD:737]-PEPC[UNIMOD:4]IDER",
            (("n", "UNIMOD:737"), (3, "UNIMOD:4")),
        ),
        # Two composition-routed mods on ONE site: torch accumulates the compositions and runs comp_enc
        # once, so the site gets ONE comp_enc.bias. Encoding each mod separately and summing
        # the vectors would add the bias twice, a whole-bias-sized error, not a rounding one.
        (
            "co-sited",
            "PEPC[UNIMOD:35][UNIMOD:21]IDER",
            "PEPC[UNIMOD:21][UNIMOD:35]IDER",
            ((3, "UNIMOD:35"), (3, "UNIMOD:21")),
        ),
    ],
)
def test_parity_modified_peptides(artifact, capsys, label, modseq, canonical, mods):
    """The Rust runtime must encode modifications, not silently predict the bare peptide."""
    binary = _binary()
    model = artifact["model"]

    pep = Peptide("PEPCIDER" if "C[" in modseq else "PEPTIDER", mods)

    lib = predict_library_fast(
        TorchRunner(model, "cpu"),
        [Precursor(peptide=pep, charge=CHARGE, split="train")],
        min_intensity=0.01,
    )
    rj = _rust(binary, artifact["path"], [], peptide=modseq)
    assert rj["peptide"] == canonical, "the CLI must echo the canonical parsed identity"

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
    _assert_close(f"{label} RT", float(lib["rt_pred"].iloc[0]), rj["rt"])
    _assert_close(f"{label} CCS", float(lib["ccs_pred"].iloc[0]), rj["ccs"])
    _assert_close(
        f"{label} precursor m/z",
        float(lib["precursor_mz"].iloc[0]),
        rj["precursor_mz"],
        atol=MZ_ATOL,
        rtol=MZ_RTOL,
    )
    _assert_close(
        f"{label} fragment m/z",
        [py[key][0] for key in py],
        [rs[key][0] for key in py],
        atol=MZ_ATOL,
        rtol=MZ_RTOL,
    )
    _assert_close(
        f"{label} relative intensity",
        [py[key][1] for key in py],
        [rs[key][1] for key in py],
    )


def test_modified_peptide_differs_from_the_bare_one(artifact):
    """Guards the parity assertions above: if the mod were dropped on BOTH sides they would
    still agree, and the test would prove nothing."""
    binary = _binary()
    bare = _rust(binary, artifact["path"], [])
    modded = _rust(binary, artifact["path"], [], peptide="[UNIMOD:737]-PEPTIDER")
    assert abs(bare["rt"] - modded["rt"]) > 1e-4
    assert abs(bare["precursor_mz"] - modded["precursor_mz"]) > 1e-4


def test_rust_rejects_an_out_of_range_charge(artifact):
    """An out-of-range charge must be a named error, not an ndarray bounds abort (rc=101)."""
    r = subprocess.run(
        [
            _binary(),
            "predict",
            "--model",
            str(artifact["path"]),
            "--peptide",
            PEPTIDE,
            "--charge",
            "20",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, f"expected a clean error, got rc={r.returncode}: {r.stderr[-400:]}"
    assert "charge 20" in r.stderr, r.stderr
    assert str(artifact["model"].cfg.max_charge) in r.stderr, r.stderr


def test_rust_refuses_a_site_mixing_composition_and_mass_routes(artifact):
    """`comp_enc`/`mass_enc` routing is per column, so the second mod would vanish from the
    model input while still moving every m/z. Both runtimes must refuse instead."""
    modseq = "PEPC[UNIMOD:4][+15.994915]IDER"
    r = subprocess.run(
        [
            _binary(),
            "predict",
            "--model",
            str(artifact["path"]),
            "--peptide",
            modseq,
            "--charge",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, f"expected a clean error, got rc={r.returncode}: {r.stderr[-400:]}"
    assert "UNIMOD:4" in r.stderr and "15.994915" in r.stderr, r.stderr


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
        [
            _binary(),
            "predict",
            "--model",
            str(stale),
            "--peptide",
            PEPTIDE,
            "--charge",
            str(CHARGE),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "format_version" in r.stderr


def _spectronaut_library(path, model, peptides):
    """Write a Spectronaut TSV whose intensities are the model's own, shifted.

    Predictions rather than random numbers so a fit has something coherent to find, and shifted
    so the zero-context starting point is not already optimal, the point of the test is that
    fitting moves the held-out score, which a library the model already predicts perfectly
    could not show.
    """
    header = (
        "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\t"
        "IonMobility\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\t"
        "FragmentNumber\tFragmentCharge\tFragmentLossType\n"
    )
    rows = [header]
    for sequence in peptides:
        peptide = Peptide(sequence)
        batch = collate([Precursor(peptide, 2, "lib")])
        with torch.no_grad():
            ms2 = model.forward_context(batch)["ms2"][0].numpy()
        sites = len(sequence) - 1
        for site in range(sites):
            for ion, (kind, z) in enumerate((("b", 1), ("y", 1), ("b", 2), ("y", 2))):
                intensity = float(ms2[FRAG_OFFSET + site, ion])
                # Tilt the spectrum along the fragment axis: a systematic distortion is what an
                # acquisition context can absorb, unlike noise.
                intensity *= 1.0 + 0.6 * (site / max(sites - 1, 1))
                if intensity < 0.02:
                    continue
                ordinal = site + 1 if kind == "b" else sites - site
                rows.append(
                    f"_{sequence}_\t{sequence}\t500.0\t2\t0.5\t0.9\tP1\t0\t"
                    f"100.0\t{intensity:.5f}\t{kind}\t{ordinal}\t{z}\tnoloss\n"
                )
    path.write_text("".join(rows))
    return path


def test_fit_context_improves_a_held_out_library_score(artifact, tmp_path):
    """Fitting a context has to move a score on peptides it never fitted on.

    Not compared to the Python reference value: the two objectives differ slightly (Python's grid
    carries the padded fragment rows, and it differentiates exactly rather than by finite
    differences), so pinning a number here would pin the difference rather than the behaviour.
    """
    from pepdistill.data.split import assign_split
    from pepdistill.data.config import SplitConfig

    # Enough peptides that the project's hash split yields both halves; asserted, not assumed.
    peptides = [f"PEPTIDE{chr(65 + i % 26)}{chr(65 + i // 26)}K" for i in range(220)]
    splits = {assign_split(s, SplitConfig()) for s in peptides}
    assert {"train", "val"} <= splits, f"fixture needs both splits, got {splits}"

    library = _spectronaut_library(tmp_path / "lib.tsv", artifact["model"], peptides)
    r = subprocess.run(
        [
            _binary(),
            "fit-context",
            "--model",
            str(artifact["path"]),
            "--library",
            str(library),
            "--epochs",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)

    assert report["split"]["train"] > 0 and report["split"]["val"] > 0
    assert report["fit"]["context_dim"] == artifact["model"].cfg.context_dim
    assert len(report["fit"]["context"]) == report["fit"]["context_dim"]
    after = report["fit"]["spectral_angle_after"]
    before = report["fit"]["spectral_angle_before"]
    assert after > before, f"fitting did not improve held-out agreement: {before} -> {after}"
    # One entry per epoch, over the whole training set, so it is comparable across epochs.
    assert len(report["objective"]) == 2


def test_fit_context_refuses_an_undeclared_modification(artifact, tmp_path):
    """An unexplained mass shift stops the fit rather than being silently dropped."""
    library = tmp_path / "modified.tsv"
    library.write_text(
        "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\t"
        "IonMobility\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\t"
        "FragmentNumber\tFragmentCharge\tFragmentLossType\n"
        "_PEPT[+79.96633]IDEK_\tPEPTIDEK\t540.1\t2\t0.4\t0.9\tP1\t0\t120.0\t0.7\tb\t3\t1\tnoloss\n"
    )
    r = subprocess.run(
        [_binary(), "fit-context", "--model", str(artifact["path"]), "--library", str(library)],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "+79.96633" in r.stderr and "--add-unimod" in r.stderr


def test_parity_named_ms_context(artifact, capsys):
    """`--ms-context NAME` reaches the heads through the same projection a factor-composed
    context does, so what is under test is that both sides agree on which row the name means."""
    binary = _binary()
    enc, model = artifact["enc"], artifact["model"]
    unknown = torch.zeros(1, dtype=torch.long)
    ms_vec = (
        enc(unknown, unknown, unknown, None, setup_id=torch.tensor([enc.setup_row(NAMED_SETUP)]))
        .detach()
        .numpy()[0]
    )
    lib = predict_library_fast(
        TorchRunner(model, "cpu", ms_context=ms_vec),
        [Precursor(peptide=Peptide(PEPTIDE), charge=CHARGE, split="train")],
        min_intensity=0.01,
    )
    rj = _rust(binary, artifact["path"], ["--ms-context", NAMED_SETUP])
    base = _rust(binary, artifact["path"], [])

    py, rs = _frag_map_py(lib), _frag_map_rust(rj)
    assert set(py) == set(rs)
    with capsys.disabled():
        print(f"\n[named-setup] n={len(py)} rows={enc.setups}")
    _assert_close(
        "named setup relative intensity",
        [py[key][1] for key in py],
        [rs[key][1] for key in py],
    )
    # The named row must actually move MS2 off the base, or the comparison above proves nothing.
    base_map = _frag_map_rust(base)
    assert max(abs(rs[key][1] - base_map[key][1]) for key in rs) > 1e-4


def test_rust_refuses_a_setup_it_has_no_row_for(artifact):
    """Answering with the neutral row would report a prediction for a setup it never fitted."""
    r = subprocess.run(
        [
            _binary(),
            "predict",
            "--model",
            str(artifact["path"]),
            "--peptide",
            PEPTIDE,
            "--charge",
            str(CHARGE),
            "--ms-context",
            "never_fitted",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "unknown --ms-context" in r.stderr and NAMED_SETUP in r.stderr


def test_a_fitted_context_can_be_saved_and_then_addressed_by_name(artifact, tmp_path):
    """The loop the fit exists to close: fit a library's context, name it, predict with it.

    The saved artifact has to predict exactly what the fit reported the context to be, or the
    number in the report describes something other than what was stored.
    """
    peptides = [f"PEPTIDE{chr(65 + i % 26)}{chr(65 + i // 26)}K" for i in range(220)]
    library = _spectronaut_library(tmp_path / "lib.tsv", artifact["model"], peptides)
    out = tmp_path / "fitted.safetensors"
    r = subprocess.run(
        [
            _binary(),
            "fit-context",
            "--model",
            str(artifact["path"]),
            "--library",
            str(library),
            "--epochs",
            "1",
            "--save-as",
            "diaPASEF_cyspat",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["saved"] == {
        "setup": "diaPASEF_cyspat",
        "row": 2,  # 1 is the fixture's own named setup; a new name must not take its row
        "artifact": str(out),
    }

    fitted = _rust(_binary(), out, ["--ms-context", "diaPASEF_cyspat"])
    # The row the fit wrote, applied by hand through the exported weights, must reproduce it.
    enc, model = artifact["enc"], artifact["model"]
    context = torch.tensor([report["fit"]["context"]], dtype=torch.float32)
    lib = predict_library_fast(
        TorchRunner(model, "cpu", ms_context=context.numpy()[0]),
        [Precursor(peptide=Peptide(PEPTIDE), charge=CHARGE, split="train")],
        min_intensity=0.01,
    )
    py, rs = _frag_map_py(lib), _frag_map_rust(fitted)
    assert set(py) == set(rs)
    _assert_close(
        "saved context relative intensity",
        [py[key][1] for key in py],
        [rs[key][1] for key in py],
    )
    # The setup already in the artifact has to survive the write.
    assert enc.setups[NAMED_SETUP] == 1
    assert _rust(_binary(), out, ["--ms-context", NAMED_SETUP])["fragments"]["rel"]


def _mzspeclib_library(artifact_path, directory, out_name, extra=()):
    """Generate a four-precursor mzSpecLib library and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    fasta = directory / "tiny.fasta"
    fasta.write_text(">protein_one description\nPEPTIDEM\n")
    out = directory / out_name
    r = subprocess.run(
        [
            _binary(),
            "library",
            "--model",
            str(artifact_path),
            "--fasta",
            str(fasta),
            "--out",
            str(out),
            *extra,
            "--min-length",
            "8",
            "--max-length",
            "8",
            "--min-charge",
            "2",
            "--max-charge",
            "3",
            "--max-fragments",
            "4",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return out


def test_mzspeclib_output_is_readable_by_the_reference_implementation(artifact, tmp_path):
    """The point of the format is that someone else's parser can read it, so let one.

    Skipped unless the reference implementation is installed; it is deliberately not a dev
    dependency, because that group reaches every `uv run` including preparation workers. Run it
    with `uv run --with mzspeclib pytest tests/test_rust_parity.py -k mzspeclib`.
    """
    mzspeclib = pytest.importorskip("mzspeclib", reason="pip install mzspeclib to check validity")

    out = _mzspeclib_library(artifact["path"], tmp_path, "library.mzspeclib.txt")
    library = mzspeclib.SpectrumLibrary(filename=str(out))
    spectra = list(library.read())
    # PEPTIDEM and its oxidized form, at charges 2 and 3.
    assert len(spectra) == 4
    assert {spectrum.name for spectrum in spectra} == {
        "PEPTIDEM/2",
        "PEPTIDEM/3",
        "PEPTIDEM[UNIMOD:35]/2",
        "PEPTIDEM[UNIMOD:35]/3",
    }
    for spectrum in spectra:
        analyte = spectrum.get_analyte(1)
        assert analyte.get_attribute("MS:1000888|stripped peptide sequence") == "PEPTIDEM"
        assert analyte.get_attribute("MS:1000885|protein accession") == "protein_one"
        assert 0 < len(spectrum.peak_list) <= 4
        for mz, intensity, annotations, *_ in spectrum.peak_list:
            assert mz > 0 and 0 < intensity <= 1
            assert annotations, "every predicted peak carries its mzPAF annotation"

    # The provenance the sidecar carries is *in* the file, which is the whole reason for this
    # format: pairs are grouped, so a name and its value share a group id.
    named = {
        attribute.group_id: attribute.value
        for attribute in library.attributes
        if attribute.key.endswith("other attribute name")
    }
    values = {
        named[attribute.group_id]: attribute.value
        for attribute in library.attributes
        if attribute.key.endswith("other attribute value") and attribute.group_id in named
    }
    assert values["pepdistill:inputs.model"] == str(artifact["path"])
    assert len(values["pepdistill:inputs.model_blake2b_256"]) == 64
    assert str(values["pepdistill:fragments.max_fragments"]) == "4"
    assert values["pepdistill:digestion.enzyme"] == "trypsin"
    assert values["pepdistill:output.format"] == "mzspeclib-text"


def test_mzspeclib_output_violates_no_rule_at_any_level(artifact, tmp_path):
    """Parsing is not conformance: run the reference validator and require a clean log.

    Every validation level matters. A SHOULD violation still forces a reader to guess at something the
    format has a term for, which is the situation this export exists to end.
    """
    pytest.importorskip("mzspeclib", reason="pip install mzspeclib to check validity")
    from mzspeclib import SpectrumLibrary
    from mzspeclib.validate import validator

    out = _mzspeclib_library(artifact["path"], tmp_path, "library.mzspeclib.txt")
    chain = validator.load_default_validator()
    chain.validate_library(SpectrumLibrary(filename=str(out)))
    assert [error.message for error in chain.error_log] == []


def test_mzspeclib_reports_both_retention_quantities_under_a_chrom_context(artifact, tmp_path):
    """A named dataset makes `rt` a gradient time, so the index has to be reported separately.

    The two are not interconvertible. The context enters the RT head's input, not its output.
    A library that reported only one would have thrown the other away.
    """
    pytest.importorskip("mzspeclib", reason="pip install mzspeclib to check validity")
    from mzspeclib import SpectrumLibrary
    from mzspeclib.validate import validator

    out = _mzspeclib_library(
        artifact["path"], tmp_path, "library.mzspeclib.txt", extra=("--chrom-context", "dsA")
    )
    text = out.read_text()
    assert "MS:1000894|retention time=" in text
    assert "MS:1000896|normalized retention time=" in text

    library = SpectrumLibrary(filename=str(out))
    for spectrum in library.read():
        raw = spectrum.get_attribute("MS:1000894|retention time")
        index = spectrum.get_attribute("MS:1000896|normalized retention time")
        assert raw is not None and index is not None
        # A shifted head does not reproduce the context-free value, or the context did nothing.
        assert raw != index

    # The header states which of the two is an index, and on what standard.
    named = {
        attribute.group_id: attribute.value
        for attribute in library.attributes
        if attribute.key.endswith("other attribute name")
    }
    values = {
        named[attribute.group_id]: attribute.value
        for attribute in library.attributes
        if attribute.key.endswith("other attribute value") and attribute.group_id in named
    }
    assert values["pepdistill:retention.normalized.kind"] == "dimensionless index, minutes-like"
    assert "TFAHTESHISK = 0" in values["pepdistill:retention.normalized.scale"]
    assert values["pepdistill:retention.raw.chrom_context"] == "dsA"

    chain = validator.load_default_validator()
    chain.validate_library(SpectrumLibrary(filename=str(out)))
    assert [error.message for error in chain.error_log] == []


def test_the_binary_predicts_with_no_model_argument():
    """A fresh build has to be able to predict, which is the point of bundling the weights.

    Also pins the default: were it to fall back to a path, this would fail with a missing file
    rather than silently predicting from something unexpected.
    """
    r = subprocess.run(
        [_binary(), "predict", "--peptide", PEPTIDE, "--charge", str(CHARGE)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    prediction = json.loads(r.stdout)
    assert prediction["fragments"]["mz"], "bundled model produced no fragments"
    assert prediction["rt"] == pytest.approx(41.0, abs=40.0), "implausible retention index"

    unknown = subprocess.run(
        [_binary(), "predict", "--model", "builtin:nope", "--peptide", PEPTIDE, "--charge", "2"],
        capture_output=True,
        text=True,
    )
    assert unknown.returncode != 0
    # The error names what this build actually carries, so the fix is readable from it.
    assert "small-v0" in unknown.stderr


def test_retention_scale_claim_is_checked_not_asserted(artifact, tmp_path):
    """The anchor check has to *refuse* a model that is not on the corpus index.

    This fixture is randomly initialised, so its retention head cannot put `TFAHTESHISK` at 0 and
    `SILDYVSLVEK` at 100. A check that passed here would pass anything, and every library we
    publish would carry a scale claim worth nothing.
    """
    out = _mzspeclib_library(artifact["path"], tmp_path, "library.mzspeclib.txt")
    normalized = json.loads(Path(f"{out}.config.json").read_text())["retention"]["normalized"]

    assert normalized["anchor_check"]["on_scale"] is False
    assert normalized["anchor_check"]["max_abs_error"] > 2.0
    # The scale is described by the convention that defines it, and both anchors are named with
    # what they predicted, so a failure says which one moved.
    assert "TFAHTESHISK = 0" in normalized["scale"]
    assert [a["peptide"] for a in normalized["anchor_check"]["anchors"]] == [
        "TFAHTESHISK",
        "SILDYVSLVEK",
    ]


def test_mzspeclib_gzip_is_the_same_library_compressed(artifact, tmp_path):
    """`.gz` runs through the same writer thread, so the two can differ only in compression."""
    plain = _mzspeclib_library(artifact["path"], tmp_path / "plain", "library.mzspeclib.txt")
    zipped = _mzspeclib_library(artifact["path"], tmp_path / "zipped", "library.mzspeclib.txt.gz")
    # From the first spectrum on: the headers record their own paths, which differ by construction.
    body = lambda text: text.split("<Spectrum=1>", 1)[1]  # noqa: E731
    assert body(gzip.decompress(zipped.read_bytes()).decode()) == body(plain.read_text())
