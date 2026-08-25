"""Export a checkpoint to .safetensors and verify tensors + metadata round-trip."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from pepdistill.export import export_safetensors
from pepdistill.models.context import ChromRunbook, MSContextEncoder
from pepdistill.models.registry import build_student, save_checkpoint


def test_export_safetensors_roundtrip(tmp_path: Path):
    model = build_student("small")
    model.set_norm(31.0, 4.0, 410.0, 25.0)
    enc = MSContextEncoder(model.cfg.context_dim)
    runbook = ChromRunbook(2, model.cfg.context_dim)
    dataset_index = {"dsA": 1, "dsB": 2}

    ckpt = tmp_path / "m.ckpt"
    save_checkpoint(model, ckpt, encoder=enc, runbook=runbook, dataset_index=dataset_index)

    out = tmp_path / "m.safetensors"
    export_safetensors(ckpt, out)
    assert out.exists()

    with safe_open(out, framework="pt") as f:
        meta = json.loads(f.metadata()["pepdistill"])
        keys = set(f.keys())
        # A couple of representative tensors match the torch weights exactly.
        tok = f.get_tensor("model.token_emb.weight")
        assert torch.equal(tok, model.state_dict()["token_emb.weight"])
        enc_e = f.get_tensor("enc.inst_emb.weight")
        assert torch.equal(enc_e, enc.state_dict()["inst_emb.weight"])
        assert torch.equal(f.get_tensor("runbook.emb.weight"), runbook.state_dict()["emb.weight"])

    # Norm buffers are hoisted to metadata, not shipped as tensors.
    assert not any(k in keys for k in ("model.rt_mean", "model.rt_std", "model.ccs_mean"))
    assert meta["norm"] == {"rt_mean": 31.0, "rt_std": 4.0, "ccs_mean": 410.0, "ccs_std": 25.0}
    assert meta["config"]["d_model"] == model.cfg.d_model
    assert "charge_in_trunk" not in meta["config"]  # removed from the model
    assert meta["has_encoder"] and meta["has_runbook"]
    assert meta["vocab"]["instruments"] == list(enc.instruments)
    assert meta["dataset_index"] == dataset_index


def test_export_carries_the_provenance_of_its_checkpoint(tmp_path: Path):
    """The artifact is what gets redistributed and bundled, so its origin travels inside it.

    A note in a README drifts from the file it describes; metadata cannot. The Rust reader does not
    model this key, which is the point -- whatever the trainer recorded passes through unaltered.
    """
    model = build_student("small")
    model.set_norm(31.0, 4.0, 410.0, 25.0)
    ckpt = tmp_path / "m.ckpt"
    training = {
        "stage": "train",
        "checkpoint_kind": "best",
        "epoch": 33,
        "validation": {"metric": "mean_per_dataset_spectral_angle", "values": {"val/a": 0.8}},
    }
    save_checkpoint(model, ckpt, training_metadata=training)

    out = tmp_path / "m.safetensors"
    export_safetensors(ckpt, out)
    with safe_open(out, framework="pt") as f:
        provenance = json.loads(f.metadata()["pepdistill"])["provenance"]

    assert provenance["checkpoint"] == "m.ckpt"
    assert len(provenance["checkpoint_blake2b_256"]) == 64
    assert provenance["training"] == training


def test_export_without_a_training_record_still_names_its_source(tmp_path: Path):
    """A checkpoint saved without training metadata is exportable; it just says less."""
    model = build_student("flash")
    model.set_norm(31.0, 4.0, 410.0, 25.0)
    ckpt = tmp_path / "bare.ckpt"
    save_checkpoint(model, ckpt)

    out = tmp_path / "bare.safetensors"
    export_safetensors(ckpt, out)
    with safe_open(out, framework="pt") as f:
        provenance = json.loads(f.metadata()["pepdistill"])["provenance"]

    assert provenance["checkpoint"] == "bare.ckpt"
    assert "training" not in provenance
