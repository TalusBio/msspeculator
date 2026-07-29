import torch

from pepdistill.chem import ION_TYPES, Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.distill.losses import ms2_cosine_loss, spectral_angle
from pepdistill.models.registry import PRESETS, build_student


def _precs():
    return [
        Precursor(Peptide("SAMPLER"), 2, "train"),
        Precursor(Peptide("ACDEMKPEPTIDE", ((1, "Carbamidomethyl@C"),)), 3, "train"),
    ]


def test_collate_always_wraps_termini():
    batch = collate(_precs())
    tok_len = 13 + 2  # longest peptide + mandatory N/C-term tokens
    assert batch.tokens.shape == (2, tok_len)
    assert batch.pad_mask.shape == (2, tok_len)
    assert batch.frag_mask.shape == (2, tok_len - 1)
    # First peptide has length 7 -> 6 inter-residue fragment sites, starting at index 1.
    assert batch.frag_mask[0].sum().item() == 6
    assert batch.frag_mask[0, 0].item() is False or not bool(batch.frag_mask[0, 0])
    assert bool(batch.frag_mask[0, 1])
    assert batch.pad_mask[0].sum().item() == tok_len - (7 + 2)


def test_termini_toggle_is_gone():
    import pepdistill.data.encode as enc

    for name in ("use_termini", "set_termini", "frag_offset", "_USE_TERMINI"):
        assert not hasattr(enc, name), f"{name} should have been deleted"


def test_student_forward_shapes_and_bounds():
    for preset in PRESETS:
        model = build_student(preset).eval()
        batch = collate(_precs())
        out = model(batch)

        extra = 2  # mandatory N/C-term tokens
        assert out["ms2"].shape == (2, 13 + extra - 1, len(ION_TYPES))
        assert out["rt"].shape == (2,)
        assert out["ccs"].shape == (2,)
        assert out["ms2"].min() >= 0.0 and out["ms2"].max() <= 1.0
        assert model.num_parameters() > 0


def test_padding_invariance_all_backbones():
    """A peptide's prediction must not depend on batch padding (all backbones)."""
    p20 = Precursor(Peptide("ACDEFGHIKLMNPQRSTVWY"), 2, "t")  # len 20
    p30 = Precursor(Peptide("ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"), 2, "t")  # len 30 -> pads p20
    off = 1  # mandatory N-term token occupies index 0
    for preset in PRESETS:
        m = build_student(preset).eval()
        with torch.no_grad():
            alone = m(collate([p20]))
            padded = m(collate([p20, p30]))
        assert abs(float(alone["rt"][0] - padded["rt"][0])) < 1e-5, preset
        assert abs(float(alone["ccs"][0] - padded["ccs"][0])) < 1e-5, preset
        # p20 has 19 fragment sites at adjacent-pool indices [off, off+19).
        a = alone["ms2"][0, off : off + 19]
        p = padded["ms2"][0, off : off + 19]
        assert float((a - p).abs().max()) < 1e-5, preset


def test_ms2_cosine_loss_zero_when_identical():
    x = torch.rand(3, 5, len(ION_TYPES))
    mask = torch.ones(3, 5, dtype=torch.bool)
    assert ms2_cosine_loss(x, x.clone(), mask).item() < 1e-6


def test_spectral_angle_one_when_identical():
    x = torch.rand(3, 5, len(ION_TYPES)) + 0.1
    mask = torch.ones(3, 5, dtype=torch.bool)
    assert spectral_angle(x, x.clone(), mask).mean().item() > 0.999


def test_denormalize_roundtrip():
    model = build_student("tiny")
    model.set_norm(50.0, 10.0, 400.0, 25.0)
    out = {"ms2": torch.zeros(1, 1, 1), "rt": torch.tensor([1.0]), "ccs": torch.tensor([-1.0])}
    den = model.denormalize(out)
    assert torch.isclose(den["rt"], torch.tensor([60.0]))
    assert torch.isclose(den["ccs"], torch.tensor([375.0]))


def test_set_norm_leaves_unspecified_stats_untouched():
    """A regime with no data for a property must be able to skip it.

    Regression: the real-speclib regime passed (0.0, 1.0) for CCS because PROSPECT has no
    CCS column. That did not disable the CCS head — it overwrote the calibration pretrain
    had learned, so a trained head denormalized to raw standardized values and emitted
    negative CCS that looked like plausible small numbers.
    """
    model = build_student("tiny")
    model.set_norm(50.0, 10.0, 400.0, 25.0)

    model.set_norm(rt_mean=43.0, rt_std=30.0)  # what the real regime now does

    assert float(model.rt_mean) == 43.0 and float(model.rt_std) == 30.0
    assert float(model.ccs_mean) == 400.0, "CCS calibration was clobbered"
    assert float(model.ccs_std) == 25.0, "CCS calibration was clobbered"
    # And the round trip still lands in native units.
    assert abs(float(model.denormalize(
        {"ms2": torch.zeros(1, 1, 1), "rt": torch.tensor([0.0]), "ccs": torch.tensor([1.0])}
    )["ccs"]) - 425.0) < 1e-4


def test_set_norm_rejects_non_finite():
    """A NaN std would surface as a confident, meaningless prediction rather than a failure."""
    import math as _math

    import pytest

    model = build_student("tiny")
    for kwargs in ({"rt_std": _math.nan}, {"ccs_mean": _math.inf}, {"rt_mean": -_math.inf}):
        with pytest.raises(ValueError, match="must be finite"):
            model.set_norm(**kwargs)
