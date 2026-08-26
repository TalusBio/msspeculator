"""Export a trained checkpoint to a self-contained ``.safetensors`` artifact for the Rust CLI.

One file carries everything the Rust runtime needs: the student weights (plus the acquisition
encoder / chrom runbook if the checkpoint had them) as tensors, and the config + vocab + target
normalization stats + dataset index as a single JSON blob in safetensors' ``__metadata__`` map.
No pickle crosses the language boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors.torch import save_file

from .models.registry import _load_checkpoint_blob, load_checkpoint, load_context

# v2: the two-encoder mod representation (comp_enc / mass_enc) replaced the single scaled
# mod_proj scalar, and the N/C-term tokens became mandatory. A v1 artifact's tensors mean
# something different, so the Rust reader rejects it rather than reading it with defaults.
# v3: the ChromRunbook gained a per-dataset RT output affine (log_scale / shift). A v2
# artifact lacks those tensors; the reader rejects it rather than assuming identity.
FORMAT_VERSION = 3
# StudentModel registers these as buffers; they are 1-element scalars, hoisted into metadata
# rather than shipped as tensors (simpler for the Rust reader).
_NORM_KEYS = ("rt_mean", "rt_std", "ccs_mean", "ccs_std")
# Training-side bookkeeping buffer (has the RT affine been established?). It guards against
# re-standardizing mid-curriculum and means nothing at inference, so it is dropped rather
# than shipped as a tensor the Rust reader would have to know to ignore.
_TRAINING_ONLY_KEYS = ("norm_established",)


def _checkpoint_training_metadata(ckpt_path: str | Path) -> dict | None:
    """The checkpoint's own training record, or ``None`` if it carries none.

    Written by ``save_checkpoint`` and deliberately not modelled here: whatever the trainer chose
    to record travels through unaltered, so adding a field there needs no change on this side.
    """
    blob = _load_checkpoint_blob(ckpt_path)
    training = blob.get("training")
    return training if isinstance(training, dict) else None


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

    # Where these weights came from, carried inside the artifact. The artifact is the thing that
    # gets redistributed and bundled into a binary, so its provenance has to travel with it rather
    # than live in a note beside it; the same reason the vendored UNIMOD tables open with a
    # provenance header. The checkpoint already records stage, epoch, step and per-dataset
    # validation; that is copied verbatim, alongside the identity of the file it was read from.
    #
    # `Meta` in the Rust reader does not model this key and does not need to: unknown metadata is
    # preserved through a read/write round trip, so nothing here forces a format version bump.
    source = Path(ckpt_path)
    provenance: dict = {"checkpoint": source.name}
    if source.is_file():
        digest = hashlib.blake2b(source.read_bytes(), digest_size=32).hexdigest()
        provenance["checkpoint_blake2b_256"] = digest
    training = _checkpoint_training_metadata(ckpt_path)
    if training is not None:
        provenance["training"] = training
    meta["provenance"] = provenance

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
        # Named acquisition setups: `--ms-context NAME` resolves through this map into the
        # `enc.setup_emb.weight` rows exported above it, the same arrangement `dataset_index`
        # has with the runbook. Omitted when empty so its absence stays unambiguous; no
        # index means no setup was ever named, not one whose names were lost.
        if enc.setups:
            meta["ms_context_index"] = enc.setups
    if ctx is not None and ctx.runbook is not None:
        for key, val in ctx.runbook.state_dict().items():
            tensors[f"runbook.{key}"] = val.contiguous().cpu()
        meta["has_runbook"] = True
    # From the book itself when it has an index: the runtime resolves `--chrom-context NAME`
    # through this map, so it has to be the one that names the rows being exported beside it.
    index = (
        ctx.runbook.names
        if ctx is not None and ctx.runbook is not None and ctx.runbook.names
        else (ctx.dataset_index if ctx is not None else None)
    )
    if index:
        meta["dataset_index"] = index

    out_path = Path(out_path)
    # Write both keys during the rename. New readers use msspeculator; old artifacts and tools
    # still look for pepdistill.
    encoded_meta = json.dumps(meta)
    save_file(
        tensors,
        str(out_path),
        metadata={"msspeculator": encoded_meta, "pepdistill": encoded_meta},
    )
    return out_path
