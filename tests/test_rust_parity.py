"""Torch and Rust must agree on what a set of weights predicts.

The comparison is at the tensor the two runtimes actually compute: dense MS2, RT and CCS, before
any base-peak normalization or intensity floor. Row assembly exists once, in Rust, so there is
nothing at row level left to compare, and filtering first would hide small disagreements.

In-process through `msspeculator_rs.PortableWeights`, so this runs everywhere pytest does. It
needs no cargo toolchain and starts no subprocess. `test_rust_cli.py` covers what the built
binary does with these same weights.

Fragment m/z is not compared here. Both runtimes read it from `msspeculator_core::chem`, and
`core::bucket::bucket_fragment_mz_matches_scalar` already pins the batched path to the scalar one.
"""

import numpy as np
import pytest
import torch

import msspeculator_rs as rs
from msspeculator.chem import Peptide
from msspeculator.data.encode import FRAG_OFFSET, collate
from msspeculator.data.precursors import Precursor

# Mixed tolerance: f32 heads accumulate differently under ndarray and torch, so RT and CCS at
# native scale (tens, hundreds) need the relative term while MS2 in [0,1] needs the absolute one.
PRED_ATOL = 1e-3
PRED_RTOL = 2e-5
PEPTIDE, CHARGE = "PEPTIDER", 2


def _assert_close(label, actual, expected):
    np.testing.assert_allclose(
        actual,
        expected,
        atol=PRED_ATOL,
        rtol=PRED_RTOL,
        err_msg=f"{label} outside atol={PRED_ATOL:g}, rtol={PRED_RTOL:g}",
    )


def _torch_forward(model, peptide, charge, **context):
    """Dense (ms2, rt, ccs) for one precursor, in native units."""
    batch = collate([Precursor(peptide=peptide, charge=charge, split="train")])
    with torch.no_grad():
        out = model.denormalize(model(batch, **context))
    frag_pos = peptide.length - 1
    ms2 = out["ms2"][0, FRAG_OFFSET : FRAG_OFFSET + frag_pos].numpy()
    return ms2, float(out["rt"][0]), float(out["ccs"][0])


def _compare(label, capsys, rust, torch_side):
    ms2_r, rt_r, ccs_r = rust
    ms2_t, rt_t, ccs_t = torch_side
    assert ms2_r.shape == ms2_t.shape, f"{label}: MS2 grids differ in shape"
    with capsys.disabled():
        print(
            f"\n[{label}] d_ms2={np.abs(ms2_r - ms2_t).max():.2e} "
            f"d_rt={abs(rt_r - rt_t):.2e} d_ccs={abs(ccs_r - ccs_t):.2e}"
        )
    _assert_close(f"{label} MS2", ms2_r, ms2_t)
    _assert_close(f"{label} RT", rt_r, rt_t)
    _assert_close(f"{label} CCS", ccs_r, ccs_t)


@pytest.fixture(scope="session")
def weights(artifact):
    return rs.PortableWeights.load(str(artifact["path"]))


def _encoded_context(enc, instrument, detector, fragmentation, energy):
    return enc(
        torch.tensor([enc.instrument_id(instrument)]),
        torch.tensor([enc.detector_id(detector)]),
        torch.tensor([enc.fragmentation_id(fragmentation)]),
        torch.tensor([energy]),
    )


@pytest.mark.parametrize(
    "label,factors",
    [
        ("base", None),
        ("ms-context", ("Lumos", "FTMS", "HCD", 30.0)),
        # The `--nce` shorthand: energy only, categoricals unknown.
        ("nce", ("", "", "", 30.0)),
    ],
)
def test_parity(artifact, weights, capsys, label, factors):
    """MS2, RT and CCS match, with and without an acquisition context."""
    enc, model = artifact["enc"], artifact["model"]
    context = None if factors is None else _encoded_context(enc, *factors)
    ms_vector = None if context is None else context.detach().numpy()[0]
    _compare(
        label,
        capsys,
        weights.forward(PEPTIDE, CHARGE, ms_context=ms_vector),
        _torch_forward(model, Peptide(PEPTIDE), CHARGE, ms_context=context),
    )


def test_parity_named_ms_context(artifact, weights, capsys):
    """A setup addressed by name reaches the heads through the same projection as factors do.

    The weights resolve the name themselves, so this also proves the exported setup index points
    at the row torch trained.
    """
    enc, model, setup = artifact["enc"], artifact["model"], artifact["setup"]
    context = enc(
        torch.tensor([enc.instrument_id("")]),
        torch.tensor([enc.detector_id("")]),
        torch.tensor([enc.fragmentation_id("")]),
        None,
        setup_id=torch.tensor([enc.setup_row(setup)]),
    )
    _compare(
        "named-setup",
        capsys,
        weights.forward(PEPTIDE, CHARGE, ms_setup=setup),
        _torch_forward(model, Peptide(PEPTIDE), CHARGE, ms_context=context),
    )


def test_parity_chrom_context(artifact, weights, capsys):
    """A named dataset routes RT through the runbook, giving raw RT instead of the index."""
    model, runbook = artifact["model"], artifact["runbook"]
    ids = torch.tensor([artifact["chrom_row"]])
    # A named dataset supplies BOTH runbook terms, the additive context vector and the output
    # scale+shift. Passing only one here would test half of what the regime trains, and the
    # fixture's normal_(std=0.3) over every parameter makes the affine decidedly non-identity.
    rust = weights.forward(PEPTIDE, CHARGE, chrom_context=artifact["chrom_dataset"])
    _compare(
        "chrom-context",
        capsys,
        rust,
        _torch_forward(
            model,
            Peptide(PEPTIDE),
            CHARGE,
            ms_context=None,
            chrom_context=runbook(ids),
            chrom_affine=runbook.affine(ids),
        ),
    )
    # The context has to move RT off the context-free index, or agreement proves nothing.
    assert abs(rust[1] - weights.forward(PEPTIDE, CHARGE)[1]) > 1e-4


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
        # Two composition-routed mods on ONE site: torch accumulates the compositions and runs
        # comp_enc once, so the site gets ONE comp_enc.bias. Encoding each mod separately and
        # summing the vectors would add the bias twice, a whole-bias-sized error, not a rounding
        # one.
        (
            "co-sited",
            "PEPC[UNIMOD:35][UNIMOD:21]IDER",
            "PEPC[UNIMOD:21][UNIMOD:35]IDER",
            ((3, "UNIMOD:35"), (3, "UNIMOD:21")),
        ),
    ],
)
def test_parity_modified_peptides(artifact, weights, capsys, label, modseq, canonical, mods):
    """The Rust runtime must encode modifications, not silently predict the bare peptide."""
    model = artifact["model"]
    peptide = Peptide("PEPCIDER" if "C[" in modseq else "PEPTIDER", mods)
    assert Peptide.from_string(modseq).modified_sequence() == canonical, (
        "the parser must round-trip to the canonical spelling"
    )
    _compare(
        label,
        capsys,
        weights.forward(modseq, CHARGE),
        _torch_forward(model, peptide, CHARGE),
    )


def test_a_modification_changes_the_prediction(weights):
    """Guards the parity assertions above: a mod dropped on BOTH sides would still agree."""
    bare = weights.forward(PEPTIDE, CHARGE)
    modded = weights.forward("[UNIMOD:737]-PEPTIDER", CHARGE)
    assert abs(bare[1] - modded[1]) > 1e-4
    assert np.abs(bare[0] - modded[0]).max() > 1e-4


def test_context_and_setup_are_mutually_exclusive(artifact, weights):
    """Two ways to name the same axis, so accepting both would silently honour one."""
    with pytest.raises(ValueError, match="not both"):
        weights.forward(
            PEPTIDE, CHARGE, ms_context=np.zeros(4, np.float32), ms_setup=artifact["setup"]
        )


def test_an_unfitted_setup_is_refused(weights):
    """Answering with the neutral row would report a prediction for a setup never fitted."""
    with pytest.raises(ValueError, match="unknown"):
        weights.forward(PEPTIDE, CHARGE, ms_setup="never-fitted")
