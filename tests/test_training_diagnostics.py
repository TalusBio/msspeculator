from pathlib import Path

import torch

from msspeculator.models.context import MSContextEncoder
from msspeculator.models.registry import build_student
from msspeculator.teacher.fake import FakeTeacher
from msspeculator.training_diagnostics import TrainingDiagnosticRenderer


def test_training_renderer_writes_fixed_panel_and_preserves_module_modes(tmp_path: Path):
    model = build_student("flash").train()
    encoder = MSContextEncoder(context_dim=model.cfg.context_dim).train()
    with torch.no_grad():
        encoder.inst_emb.weight[1, 0] = 1.0
    renderer = TrainingDiagnosticRenderer(tmp_path, FakeTeacher(), butterflies=2)

    first = renderer.render(model, encoder, "initial")
    bases = dict(renderer._bases)
    second = renderer.render(model, encoder, "epoch-0001")

    assert set(first.paths) == {
        "amino_acids",
        "modifications",
        "acquisition_contexts",
        "butterflies",
        "irt",
    }
    assert all(path.stat().st_size > 0 for path in (*first.paths.values(), *second.paths.values()))
    assert renderer._bases.keys() == bases.keys()
    assert all(renderer._bases[key] is basis for key, basis in bases.items())
    assert model.training
    assert encoder.training
    assert 0.0 <= first.metrics["teacher_spectral_angle"] <= 1.0
    assert "irt_r_squared" in first.metrics
