"""ONNX export and onnxruntime inference for the student.

The no-LSTM student exports cleanly. We export the mask-free dense forward
(``forward_dense``) with dynamic batch and length axes, so one ONNX file serves every
length bucket. Requires the ``onnx`` extra (onnx + onnxruntime).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models.student import StudentModel
from .fast import ModelRunner

_IMPORT_HINT = "ONNX support needs the extra:\n    uv pip install 'pepdistill[onnx]'"

INPUT_NAMES = ["tokens", "mod_delta", "charge"]
OUTPUT_NAMES = ["ms2", "rt", "ccs"]


def export_onnx(model: StudentModel, path: str | Path, opset: int = 17) -> Path:
    import torch

    model = model.eval()
    dummy_len = 12
    tokens = torch.randint(1, 20, (2, dummy_len), dtype=torch.long)
    mod_delta = torch.zeros(2, dummy_len, dtype=torch.float32)
    charge = torch.tensor([2, 3], dtype=torch.long)

    dynamic = {n: {0: "batch", 1: "length"} for n in ("tokens", "mod_delta")}
    dynamic["charge"] = {0: "batch"}
    dynamic["ms2"] = {0: "batch", 1: "frag"}
    dynamic["rt"] = {0: "batch"}
    dynamic["ccs"] = {0: "batch"}

    class _Wrap(torch.nn.Module):
        def __init__(self, m: StudentModel) -> None:
            super().__init__()
            self.m = m

        def forward(self, tokens, mod_delta, charge):
            return self.m.forward_dense(tokens, mod_delta, charge)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _Wrap(model),
        (tokens, mod_delta, charge),
        str(path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic,
        opset_version=opset,
        dynamo=False,
    )
    return Path(path)


class OnnxRunner(ModelRunner):
    def __init__(self, path: str | Path, intra_threads: int = 0) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(_IMPORT_HINT) from exc

        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def run(self, tokens: np.ndarray, mod_delta: np.ndarray, charge: np.ndarray):
        ms2, rt, ccs = self.sess.run(
            OUTPUT_NAMES,
            {"tokens": tokens, "mod_delta": mod_delta.astype(np.float32), "charge": charge},
        )
        return ms2, rt, ccs
