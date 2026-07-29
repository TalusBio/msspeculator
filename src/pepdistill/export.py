"""Export a trained checkpoint to a self-contained ``.safetensors`` artifact for the Rust CLI.

One file carries everything the Rust runtime needs: the student weights (plus the acquisition
encoder / chrom runbook if the checkpoint had them) as tensors, and the config + vocab + target
normalization stats + dataset index as a single JSON blob in safetensors' ``__metadata__`` map.
No pickle crosses the language boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import save_file

from .models.registry import load_checkpoint, load_context

# v2: the two-encoder mod representation (comp_enc / mass_enc) replaced the single scaled
# mod_proj scalar, and the N/C-term tokens became mandatory. A v1 artifact's tensors mean
# something different, so the Rust reader rejects it rather than reading it with defaults.
FORMAT_VERSION = 2
# StudentModel registers these as buffers; they are 1-element scalars, hoisted into metadata
# rather than shipped as tensors (simpler for the Rust reader).
_NORM_KEYS = ("rt_mean", "rt_std", "ccs_mean", "ccs_std")
# Training-side bookkeeping buffer (has the RT affine been established?). It guards against
# re-standardizing mid-curriculum and means nothing at inference, so it is dropped rather
# than shipped as a tensor the Rust reader would have to know to ignore.
_TRAINING_ONLY_KEYS = ("norm_established",)


def export_safetensors(ckpt_path: str | Path, out_path: str | Path) -> Path:
    """Read a ``.ckpt`` and write ``out_path`` as a Rust-loadable ``.safetensors``."""
    model = load_checkpoint(ckpt_path)
    ctx = load_context(ckpt_path)

    tensors = {}
    norm: dict[str, float] = {}
    for key, val in model.state_dict().items():
        if key in _NORM_KEYS:
            norm[key] = float(val.reshape(-1)[0])
        elif key in _TRAINING_ONLY_KEYS:
            continue
        else:
            tensors[f"model.{key}"] = val.contiguous().cpu()

    meta: dict = {
        "format_version": FORMAT_VERSION,
        "config": model.cfg.to_dict(),
        "norm": norm,
        "has_encoder": False,
        "has_runbook": False,
    }

    if ctx is not None and ctx.encoder is not None:
        enc = ctx.encoder
        for key, val in enc.state_dict().items():
            tensors[f"enc.{key}"] = val.contiguous().cpu()
        meta["has_encoder"] = True
        meta["vocab"] = {
            "instruments": list(enc.instruments),
            "detectors": list(enc.detectors),
            "fragmentations": list(enc.fragmentations),
        }
    if ctx is not None and ctx.runbook is not None:
        for key, val in ctx.runbook.state_dict().items():
            tensors[f"runbook.{key}"] = val.contiguous().cpu()
        meta["has_runbook"] = True
    if ctx is not None and ctx.dataset_index is not None:
        meta["dataset_index"] = ctx.dataset_index

    out_path = Path(out_path)
    save_file(tensors, str(out_path), metadata={"pepdistill": json.dumps(meta)})
    return out_path
