"""Write portable weights beside a training checkpoint, and check they still agree with torch.

Training produces a `.ckpt` that only torch can read. What ships is the `.safetensors` export, and
the two can disagree without anything failing: a tensor renamed, a buffer dropped, a metadata key
the Rust reader interprets differently. The parity suite catches that on a random-init model, but
the case that would actually bite is a *trained* one, where the weights have structure and the
disagreement can be small enough to look like a bad epoch rather than a broken export.

So every checkpoint gets an export, and every export gets compared against the torch model that
produced it, on a fixed panel. The number is logged like any other metric: it should sit at the
noise floor of f32 accumulation, and a jump means the export stopped describing the model.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from ..diagnostics import IRT_STANDARDS

#: Enough peptides to exercise several lengths without making an epoch boundary wait on it. These
#: are the same standards the Rust doctor reports on, so a drift here and a bad slope there are
#: talking about the same peptides.
_PANEL = tuple(standard.sequence for standard in IRT_STANDARDS[:4])
_PANEL_CHARGE = 2


def export_beside(ckpt_path: str | Path, mirror: Callable[[Path], str] | None = None) -> Path:
    """Export `ckpt_path` to portable weights with a matching name, and mirror them.

    Reads the checkpoint back from disk rather than exporting the live module: the export records
    the name and blake2b of the file it came from, and provenance that cannot name its own source
    is not provenance. One reload per checkpoint is noise next to an epoch.
    """
    from ..export import export_safetensors

    ckpt_path = Path(ckpt_path)
    weights = ckpt_path.with_suffix(".safetensors")
    export_safetensors(ckpt_path, weights)
    if mirror is not None:
        mirror(weights)
    return weights


def export_drift(weights_path: str | Path, model) -> float | None:
    """Largest absolute MS2 disagreement between torch and Rust on the exported weights.

    Returns ``None`` when the comparison cannot be made rather than raising: this runs at an epoch
    boundary inside a training run, and a checkpoint that cannot be re-read is a thing to report,
    not a reason to lose the epoch.
    """
    import torch

    import msspeculator_rs as rs

    from ..chem import Peptide
    from ..data.encode import FRAG_OFFSET, collate
    from ..data.precursors import Precursor

    try:
        weights = rs.PortableWeights.load(str(weights_path))
    except Exception as exc:
        warnings.warn(f"export check could not read {weights_path}: {exc}", stacklevel=2)
        return None

    # The context-free axis only. An acquisition context would have to be encoded identically on
    # both sides to mean anything here, and that is what the parity suite already pins; this is
    # asking the narrower question of whether the exported tensors are the trained ones.
    was_training = model.training
    # Eval mode, or dropout makes the torch side a different model from the one that was
    # exported: observed as a 1.5e-2 disagreement that is entirely sampling noise.
    model.eval()
    # Training runs on whatever accelerator was picked, so the batch has to follow the weights
    # rather than assume CPU.
    device = next(model.parameters()).device
    try:
        worst = 0.0
        for sequence in _PANEL:
            peptide = Peptide(sequence)
            batch = collate([Precursor(peptide, _PANEL_CHARGE, "export-check")]).to(device)
            with torch.no_grad():
                out = model(batch)
            frag_pos = peptide.length - 1
            expected = out["ms2"][0, FRAG_OFFSET : FRAG_OFFSET + frag_pos].cpu().numpy()
            actual, _, _ = weights.forward(sequence, _PANEL_CHARGE)
            worst = max(worst, float(np.abs(actual - expected).max()))
        return worst
    except Exception as exc:
        # Reported, not raised: this runs at an epoch boundary inside a training run, and losing
        # the epoch over a diagnostic is the worse failure. Reported rather than swallowed,
        # because a check that quietly stops checking is indistinguishable from a passing one.
        warnings.warn(f"export check failed on {weights_path}: {exc!r}", stacklevel=2)
        return None
    finally:
        model.train(was_training)
