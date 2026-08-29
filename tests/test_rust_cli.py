"""What the built `msspeculator-cli` binary does with a set of exported weights.

Library generation, single-peptide prediction, context fitting, mzSpecLib conformance, and the
error paths. These drive the real binary through a subprocess, so they build it once and skip
without a cargo toolchain.

Whether the two runtimes agree on what the weights predict is a different question, answered
in-process and without a toolchain by `test_rust_parity.py`.
"""

import csv
import functools
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import msspeculator_rs as rs_ext
from msspeculator.chem import Peptide
from msspeculator.data.encode import FRAG_OFFSET, collate
from msspeculator.data.precursors import Precursor

RUST_DIR = Path(__file__).resolve().parents[1] / "rust"
BIN = RUST_DIR / "target" / "release" / "msspeculator-cli"
PEPTIDE, CHARGE = "PEPTIDER", 2


@functools.cache
def _binary() -> str:
    if shutil.which("cargo") is None:
        pytest.skip("no cargo toolchain")
    r = subprocess.run(
        ["cargo", "build", "-p", "msspeculator-cli", "--release"],
        cwd=RUST_DIR,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not BIN.exists():
        pytest.skip(f"cargo build failed: {r.stderr[-400:]}")
    return str(BIN)


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
        meta = _json.loads(f.metadata()["msspeculator"])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    meta["format_version"] = 1
    stale = tmp_path / "v1.safetensors"
    save_file(tensors, str(stale), metadata={"msspeculator": _json.dumps(meta)})

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
    from msspeculator.data.split import assign_split
    from msspeculator.data.config import SplitConfig

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
    assert "unknown --ms-context" in r.stderr and artifact["setup"] in r.stderr


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

    # The row the fit wrote, addressed by its new name, must reproduce the context the report
    # claims it stored. Compared through the saved weights rather than the CLI's JSON, so a
    # disagreement points at the stored row and not at how a fragment table was rendered.
    enc, model = artifact["enc"], artifact["model"]
    context = torch.tensor([report["fit"]["context"]], dtype=torch.float32)
    batch = collate([Precursor(peptide=Peptide(PEPTIDE), charge=CHARGE, split="train")])
    with torch.no_grad():
        expected = model.denormalize(model(batch, ms_context=context))["ms2"][
            0, FRAG_OFFSET : FRAG_OFFSET + len(PEPTIDE) - 1
        ].numpy()
    saved_ms2, _, _ = rs_ext.PortableWeights.load(str(out)).forward(
        PEPTIDE, CHARGE, ms_setup="diaPASEF_cyspat"
    )
    np.testing.assert_allclose(
        saved_ms2,
        expected,
        atol=1e-3,
        rtol=2e-5,
        err_msg="the saved setup row does not predict what the fit reported",
    )
    # The setup already in the artifact has to survive the write.
    assert enc.setups[artifact["setup"]] == 1
    assert _rust(_binary(), out, ["--ms-context", artifact["setup"]])["fragments"]["rel"]


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
    assert values["msspeculator:inputs.model"] == str(artifact["path"])
    assert len(values["msspeculator:inputs.model_blake2b_256"]) == 64
    assert str(values["msspeculator:fragments.max_fragments"]) == "4"
    assert values["msspeculator:digestion.enzyme"] == "trypsin"
    assert values["msspeculator:output.format"] == "mzspeclib-text"


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
    assert values["msspeculator:retention.normalized.kind"] == "dimensionless index, minutes-like"
    assert "TFAHTESHISK = 0" in values["msspeculator:retention.normalized.scale"]
    assert values["msspeculator:retention.raw.chrom_context"] == "dsA"

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
