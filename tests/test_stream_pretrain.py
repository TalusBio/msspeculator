"""Streaming NCE-sweep pretrain: teacher labels live per batch, energy feeds the encoder."""

import numpy as np
import torch

from pepdistill.data.config import DigestConfig
from pepdistill.distill.lightning import DistillModule
from pepdistill.distill.stream_pretrain import (
    StreamMix,
    StreamPretrainCfg,
    _peptides,
    _StreamingDataset,
    default_mixes,
    fit_stream_pretrain,
)
from pepdistill.models.context import MSContextEncoder
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
    enc = MSContextEncoder(context_dim=model.cfg.context_dim)
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

    before = enc.energy_mlp[-1].weight.detach().clone()
    lines: list[str] = []
    module = fit_stream_pretrain(
        model, enc, FakeTeacher(), cfg, accelerator="cpu", log=lines.append, log_every=2
    )

    assert isinstance(module, DistillModule)
    assert any("step" in ln for ln in lines)  # _StepLogger fired (guards the .log shadow bug)
    # energy was fed through the MLP -> encoder weights received gradient and moved.
    assert not torch.allclose(before, enc.energy_mlp[-1].weight.detach())
    # rt/ccs norm was estimated from a teacher sample (not left at the 0/1 identity).
    # Pretrain establishes the CCS frame -- it is the only source of CCS -- but deliberately
    # NOT the RT frame: iRT is canonical and the teacher's RT is a different quantity in its
    # own normalized units. The first real dataset establishes RT.
    assert float(model.ccs_mean) != 0.0 or float(model.ccs_std) != 1.0
    assert float(model.rt_mean) == 0.0 and float(model.rt_std) == 1.0
    assert not bool(model.norm_established), "RT frame must stay unclaimed for the real stage"


def test_stream_pretrain_early_stops_on_plateau(tmp_path):
    """With many passes but a patience, a saturated loss halts before exhausting the stream."""
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)
    model = build_student("tiny")
    enc = MSContextEncoder(context_dim=model.cfg.context_dim)
    mixes = [StreamMix("tryptic", "tryptic", str(fasta), DigestConfig())]
    cfg = StreamPretrainCfg(
        mixes=mixes,
        chunk_size=32,
        batch_size=8,
        passes=1000,  # would run ~forever without stop
        patience=2,
        min_delta=1e9,
        check_every=3,
        warmup_steps=3,
        seed=0,
    )
    lines: list[str] = []
    module = fit_stream_pretrain(
        model, enc, FakeTeacher(), cfg, accelerator="cpu", log=lines.append
    )
    # min_delta huge -> every window "fails to improve" -> stops after patience windows.
    assert any("early-stop" in ln for ln in lines)
    assert module.trainer.global_step < 1000  # nowhere near 1000 passes


def test_stream_pretrain_cfg_defaults_teacher_acquisition():
    cfg = StreamPretrainCfg()
    assert (cfg.instrument, cfg.detector, cfg.fragmentation) == ("Lumos", "FTMS", "HCD")


def test_label_chunk_sets_ms_factors_with_swept_energy(tmp_path):
    """Each yielded LabeledBatch carries ms_factors: constant categorical ids (the teacher's
    fixed acquisition) but per-peptide swept energy, matching the NCEs the teacher was fed."""
    fasta = tmp_path / "t.fasta"
    fasta.write_text(FASTA)
    model = build_student("tiny")
    enc = MSContextEncoder(context_dim=model.cfg.context_dim)
    mixes = [StreamMix("tryptic", "tryptic", str(fasta), DigestConfig())]
    cfg = StreamPretrainCfg(mixes=mixes, nce_range=(20.0, 40.0), chunk_size=16, batch_size=4)
    ds = _StreamingDataset(FakeTeacher(), enc, cfg)
    rng = np.random.default_rng(cfg.seed)
    iters = [_peptides(m, loop=False) for m in mixes]
    items = list(ds._round_robin(iters))
    batches = list(ds._label_chunk(items, rng))
    assert batches, "expected at least one labeled batch"
    seen_energy: list[float] = []
    for lb in batches:
        f = lb.ms_factors
        assert f is not None
        n = f.instrument_id.shape[0]
        assert torch.equal(f.instrument_id, torch.full((n,), enc.instrument_id("Lumos")))
        assert torch.equal(f.detector_id, torch.full((n,), enc.detector_id("FTMS")))
        assert torch.equal(f.fragmentation_id, torch.full((n,), enc.fragmentation_id("HCD")))
        assert f.energy is not None
        assert f.energy.dtype == torch.float32
        seen_energy.extend(f.energy.tolist())
    # The energy sweep isn't constant across peptides — a genuine per-peptide NCE draw.
    assert len(set(seen_energy)) > 1
    assert all(cfg.nce_range[0] <= e <= cfg.nce_range[1] for e in seen_energy)
