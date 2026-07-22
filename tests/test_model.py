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


def test_collate_shapes_and_masks():
    from pepdistill.data.encode import use_termini

    batch = collate(_precs())
    b = 2
    extra = 2 if use_termini() else 0
    tok_len = 13 + extra  # longest peptide (13) + optional N/C-term tokens
    assert batch.tokens.shape == (b, tok_len)
    assert batch.pad_mask.shape == (b, tok_len)
    assert batch.frag_mask.shape == (b, tok_len - 1)
    # First peptide has length 7 -> 6 inter-residue fragment sites.
    assert batch.frag_mask[0].sum().item() == 6
    assert batch.pad_mask[0].sum().item() == tok_len - (7 + extra)


def test_student_forward_shapes_and_bounds():
    for preset in PRESETS:
        model = build_student(preset).eval()
        batch = collate(_precs())
        out = model(batch)
        from pepdistill.data.encode import use_termini

        extra = 2 if use_termini() else 0
        assert out["ms2"].shape == (2, 13 + extra - 1, len(ION_TYPES))
        assert out["rt"].shape == (2,)
        assert out["ccs"].shape == (2,)
        assert out["ms2"].min() >= 0.0 and out["ms2"].max() <= 1.0
        assert model.num_parameters() > 0


def test_padding_invariance_all_backbones():
    """A peptide's prediction must not depend on batch padding (all backbones)."""
    from pepdistill.data.encode import frag_offset

    p20 = Precursor(Peptide("ACDEFGHIKLMNPQRSTVWY"), 2, "t")  # len 20
    p30 = Precursor(Peptide("ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"), 2, "t")  # len 30 -> pads p20
    off = frag_offset()
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
