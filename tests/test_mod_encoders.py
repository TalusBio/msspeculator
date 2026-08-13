"""Either/or modification encoders: routing, masking, and the shared-space alignment."""

import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.models.registry import build_student


def _batch():
    return collate(
        [
            Precursor(Peptide("ACDEK", ((1, "UNIMOD:4"),)), 2, "t"),
            Precursor(Peptide("PEPTK", ((2, 42.010565),)), 2, "t"),
        ]
    )


def test_fourier_features_shape_and_range():
    from pepdistill.models.student import FourierFeatures

    ff = FourierFeatures(16)
    out = ff(torch.tensor([[0.0, 229.16, -18.01]]))
    assert out.shape == (1, 3, 32)
    assert float(out.abs().max()) <= 1.0 + 1e-6


def test_unmodified_positions_contribute_exactly_zero():
    m = build_student("flash").eval()
    b = _batch()
    with torch.no_grad():
        vec, _, _ = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_has_composition)
    assert float(vec[~b.mod_present].abs().max()) == 0.0
    assert float(vec[b.mod_present].abs().max()) > 0.0


def test_eval_routes_named_sites_to_comp_encoder():
    m = build_student("flash").eval()
    b = _batch()
    with torch.no_grad():
        vec, g, _ = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_has_composition)
    named = b.mod_has_composition
    assert named.any()
    assert torch.allclose(vec[named], g[named], atol=0)


def test_eval_routes_mass_only_sites_to_mass_encoder():
    m = build_student("flash").eval()
    b = _batch()
    with torch.no_grad():
        vec, _, mm = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_has_composition)
    mass_only = b.mod_present & ~b.mod_has_composition
    assert mass_only.any()
    assert torch.allclose(vec[mass_only], mm[mass_only], atol=0)


def test_eval_is_deterministic_across_calls():
    m = build_student("flash").eval()
    b = _batch()
    with torch.no_grad():
        a = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_has_composition)[0]
        c = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_has_composition)[0]
    assert torch.equal(a, c)


def test_swap_probability_endpoints():
    b = _batch()
    named = b.mod_has_composition

    m = build_student("flash").train()
    m.cfg.mass_swap_p = 0.0
    with torch.no_grad():
        vec, g, _ = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_has_composition)
    assert torch.allclose(vec[named], g[named], atol=0)

    m.cfg.mass_swap_p = 1.0
    with torch.no_grad():
        vec, _, mm = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_has_composition)
    assert torch.allclose(vec[named], mm[named], atol=0)


def test_isobaric_mods_collapse_to_one_mass_vector():
    """Two different compositions with the same delta mass are indistinguishable to m.

    This is an inherent property of a scalar-input fallback, asserted here so it is a known
    documented limit rather than a surprise.
    """
    m = build_student("flash").eval()
    mass = torch.tensor([[57.021464, 57.021464]])
    with torch.no_grad():
        out = m.mass_enc(mass)
    assert torch.allclose(out[0, 0], out[0, 1], atol=0)


def test_forward_exposes_both_mod_vectors():
    m = build_student("flash").eval()
    out = m(_batch())
    assert "mod_g" in out and "mod_m" in out
    assert out["mod_g"].shape == out["mod_m"].shape
    assert out["mod_g"].shape[:2] == _batch().tokens.shape


def test_mod_align_is_zero_with_no_named_sites():
    from pepdistill.distill.losses import mod_align_loss

    g = torch.randn(2, 5, 8)
    m = torch.randn(2, 5, 8)
    named = torch.zeros(2, 5, dtype=torch.bool)
    assert float(mod_align_loss(g, m, named)) == 0.0


def test_mod_align_ignores_unnamed_sites():
    from pepdistill.distill.losses import mod_align_loss

    g = torch.zeros(1, 3, 4)
    m = torch.zeros(1, 3, 4)
    m[0, 2] = 100.0  # a large error at an unnamed site must not register
    named = torch.tensor([[True, True, False]])
    assert float(mod_align_loss(g, m, named)) == 0.0


def test_mod_align_measures_named_site_error():
    from pepdistill.distill.losses import mod_align_loss

    g = torch.zeros(1, 2, 4)
    m = torch.full((1, 2, 4), 3.0)
    named = torch.tensor([[True, False]])
    assert abs(float(mod_align_loss(g, m, named)) - 9.0) < 1e-6


def test_mod_align_does_not_train_the_comp_encoder():
    """g is the teacher: the align term must leave comp_enc's gradients untouched."""
    from pepdistill.distill.losses import mod_align_loss

    model = build_student("flash").train()
    b = _batch()
    out = model(b)
    mod_align_loss(out["mod_g"], out["mod_m"], b.mod_has_composition).backward()
    assert (
        model.comp_enc.weight.grad is None or float(model.comp_enc.weight.grad.abs().max()) == 0.0
    )
    assert float(model.mass_enc[-1].weight.grad.abs().max()) > 0.0


def test_mod_align_decreases_when_fitted():
    from pepdistill.distill.losses import mod_align_loss

    torch.manual_seed(0)
    model = build_student("flash").train()
    b = _batch()
    opt = torch.optim.Adam(model.mass_enc.parameters(), lr=1e-2)
    out = model(b)
    first = float(mod_align_loss(out["mod_g"], out["mod_m"], b.mod_has_composition))
    for _ in range(50):
        opt.zero_grad()
        out = model(b)
        loss = mod_align_loss(out["mod_g"], out["mod_m"], b.mod_has_composition)
        loss.backward()
        opt.step()
    assert float(loss) < first
