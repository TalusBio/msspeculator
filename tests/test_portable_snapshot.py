"""Every training checkpoint gets portable weights, and they get checked against torch."""

import torch

from msspeculator.distill.portable_snapshot import export_beside, export_drift
from msspeculator.models.context import MSContextEncoder
from msspeculator.models.registry import build_student, save_checkpoint


def _checkpoint(tmp_path, name="latest.ckpt"):
    torch.manual_seed(0)
    model = build_student("small")
    encoder = MSContextEncoder(context_dim=model.cfg.context_dim)
    # Random init throughout, so nothing is trivially zero on either side of the export.
    for module in (model, encoder):
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.3)
    model.set_norm(31.0, 4.0, 410.0, 25.0)
    path = tmp_path / name
    save_checkpoint(model, path, encoder=encoder)
    return path, model


def test_export_lands_beside_the_checkpoint_and_is_mirrored(tmp_path):
    path, _ = _checkpoint(tmp_path)
    mirrored = []

    def mirror(item):
        mirrored.append(item.name)
        return f"s3://bucket/{item.name}"

    weights = export_beside(path, mirror)
    assert weights == tmp_path / "latest.safetensors"
    assert weights.exists()
    assert mirrored == ["latest.safetensors"]


def test_export_records_the_checkpoint_it_came_from(tmp_path):
    """The reason the export reads the path back instead of the live module."""
    import json

    from safetensors import safe_open

    path, _ = _checkpoint(tmp_path)
    weights = export_beside(path)
    with safe_open(str(weights), framework="pt") as handle:
        provenance = json.loads(handle.metadata()["msspeculator"])["provenance"]
    assert provenance["checkpoint"] == "latest.ckpt"
    assert len(provenance["checkpoint_blake2b_256"]) == 64


def test_drift_is_at_the_noise_floor_for_a_faithful_export(tmp_path):
    path, model = _checkpoint(tmp_path)
    drift = export_drift(export_beside(path), model)
    assert drift is not None
    # f32 accumulates differently under ndarray and torch; MS2 lives in [0, 1], so anything above
    # this is a disagreement about the weights rather than about arithmetic.
    assert drift < 1e-3


def test_drift_leaves_the_module_in_the_mode_it_found_it(tmp_path):
    """It runs mid-training, so flipping the model to eval and leaving it there would silently
    disable dropout for the rest of the run."""
    path, model = _checkpoint(tmp_path)
    weights = export_beside(path)
    model.train()
    export_drift(weights, model)
    assert model.training


def test_an_unreadable_export_is_reported_not_raised(tmp_path):
    """An epoch boundary is the wrong place to lose a run over a diagnostic."""
    broken = tmp_path / "not-really.safetensors"
    broken.write_bytes(b"not a safetensors file")
    _, model = _checkpoint(tmp_path)
    assert export_drift(broken, model) is None
    assert export_drift(tmp_path / "absent.safetensors", model) is None
