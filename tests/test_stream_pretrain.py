"""Streaming NCE-sweep pretrain: teacher labels live per batch, CE feeds the encoder."""

import torch

from pepdistill.data.config import DigestConfig
from pepdistill.distill.lightning import DistillModule
from pepdistill.distill.stream_pretrain import (
    StreamMix,
    StreamPretrainCfg,
    default_mixes,
    fit_stream_pretrain,
)
from pepdistill.models.context import ContextEncoder
from pepdistill.models.registry import build_student
from pepdistill.teacher import FakeTeacher

FASTA = """>p1
MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPF
>p2
SAMPLERPEPTIDEKACDEMKGGGGKLLLLLRTTTTTKVVVVVRNNNNNKQQQQQR
"""


def test_default_mixes_shape(tmp_path):
    fa = str(tmp_path / "t.fasta")
    mixes = default_mixes(fa)
    assert [m.kind for m in mixes] == ["unspecific", "tryptic"]
    assert all(m.fasta == fa for m in mixes)


def test_stream_pretrain_runs_and_moves_encoder(tmp_path):
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)
    model = build_student("tiny")
    enc = ContextEncoder(context_dim=model.cfg.context_dim)
    mixes = [
        StreamMix(
            "immuno",
            "unspecific",
            str(fasta),
            DigestConfig(enzyme="unspecific", min_charge=1, max_charge=2, max_variable_mods=0),
            min_len=8,
            max_len=11,
        ),
        StreamMix("tryptic", "tryptic", str(fasta), DigestConfig()),
    ]
    cfg = StreamPretrainCfg(
        mixes=mixes, nce_range=(20.0, 40.0), chunk_size=64, batch_size=8, passes=1, seed=0
    )

    before = enc.proj.weight.detach().clone()
    lines: list[str] = []
    module = fit_stream_pretrain(
        model, enc, FakeTeacher(), cfg, accelerator="cpu", log=lines.append, log_every=2
    )

    assert isinstance(module, DistillModule)
    assert any("step" in ln for ln in lines)  # _StepLogger fired (guards the .log shadow bug)
    # CE was fed through the projection -> encoder weights received gradient and moved.
    assert not torch.allclose(before, enc.proj.weight.detach())
    # rt/ccs norm was estimated from a teacher sample (not left at the 0/1 identity).
    assert float(model.rt_mean) != 0.0 or float(model.rt_std) != 1.0
