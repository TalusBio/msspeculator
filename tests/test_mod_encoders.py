"""Either/or modification encoders: routing, masking, and the shared-space alignment."""

import torch

from pepdistill.chem import Peptide
from pepdistill.data.encode import collate
from pepdistill.data.precursors import Precursor
from pepdistill.models.registry import build_student


def _batch():
    return collate([
        Precursor(Peptide("ACDEK", ((1, "Carbamidomethyl@C"),)), 2, "t"),
        Precursor(Peptide("PEPTK", ((2, 42.010565),)), 2, "t"),
    ])


def test_fourier_features_shape_and_range():
    from pepdistill.models.student import FourierFeatures

    ff = FourierFeatures(16)
    out = ff(torch.tensor([[0.0, 229.16, -18.01]]))
    assert out.shape == (1, 3, 32)
    assert float(out.abs().max()) <= 1.0 + 1e-6


def test_unmodified_positions_contribute_exactly_zero():
    m = build_student("tiny").eval()
    b = _batch()
    with torch.no_grad():
        vec, _, _ = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_named)
    assert float(vec[~b.mod_present].abs().max()) == 0.0
    assert float(vec[b.mod_present].abs().max()) > 0.0


def test_eval_routes_named_sites_to_comp_encoder():
    m = build_student("tiny").eval()
    b = _batch()
    with torch.no_grad():
        vec, g, _ = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_named)
    named = b.mod_named
    assert named.any()
    assert torch.allclose(vec[named], g[named], atol=0)


def test_eval_routes_mass_only_sites_to_mass_encoder():
    m = build_student("tiny").eval()
    b = _batch()
    with torch.no_grad():
        vec, _, mm = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_named)
    mass_only = b.mod_present & ~b.mod_named
    assert mass_only.any()
    assert torch.allclose(vec[mass_only], mm[mass_only], atol=0)


def test_eval_is_deterministic_across_calls():
    m = build_student("tiny").eval()
    b = _batch()
    with torch.no_grad():
        a = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_named)[0]
        c = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_named)[0]
    assert torch.equal(a, c)


def test_swap_probability_endpoints():
    b = _batch()
    named = b.mod_named

    m = build_student("tiny").train()
    m.cfg.mass_swap_p = 0.0
    with torch.no_grad():
        vec, g, _ = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_named)
    assert torch.allclose(vec[named], g[named], atol=0)

    m.cfg.mass_swap_p = 1.0
    with torch.no_grad():
        vec, _, mm = m._mod_vectors(b.mod_comp, b.mod_mass, b.mod_present, b.mod_named)
    assert torch.allclose(vec[named], mm[named], atol=0)


def test_isobaric_mods_collapse_to_one_mass_vector():
    """Two different compositions with the same delta mass are indistinguishable to m.

    This is an inherent property of a scalar-input fallback, asserted here so it is a known
    documented limit rather than a surprise.
    """
    m = build_student("tiny").eval()
    mass = torch.tensor([[57.021464, 57.021464]])
    with torch.no_grad():
        out = m.mass_enc(mass)
    assert torch.allclose(out[0, 0], out[0, 1], atol=0)


def test_forward_exposes_both_mod_vectors():
    m = build_student("tiny").eval()
    out = m(_batch())
    assert "mod_g" in out and "mod_m" in out
    assert out["mod_g"].shape == out["mod_m"].shape
    assert out["mod_g"].shape[:2] == _batch().tokens.shape
