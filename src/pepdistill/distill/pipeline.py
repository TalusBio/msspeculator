"""One config-driven Lightning pipeline: pretrain -> train -> export -> bench.

A single :class:`RunConfig` replaces the old per-stage CLI + hand-rolled trainer. Every stage
is independently toggleable; the model and the shared :class:`MSContextEncoder` are built once
and threaded through, so the teacher warmup and the real-data sink share one acquisition-factor
(instrument/detector/fragmentation/energy) axis (the teacher's NCE is a factor, not a baked base).

Stages:
- **pretrain** — online teacher-distill warmup. Enumerate the ``sources`` live (unspecific
  enzyme -> immunopeptidome windows, else tryptic) with the teacher labeling over an NCE sweep,
  so collision energy comes from the data (never fabricated) and the encoder learns a real CE
  axis. (A fixed-energy corpus would just be a dataset that carries its own CE — no special mode.)
- **train** — real-speclib sink on a PROSPECT pool (streamed shard-by-shard), per-run
  ``chrom_context`` and factor-driven ``ms_context``.
- **export** — ONNX. **bench** — library-generation throughput on a FASTA digest.

Inference (predict a library from a finished model) is deliberately NOT here — it is the
standalone ``predict`` command.
"""

from __future__ import annotations

import gc
import json
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..data.config import DigestConfig, SplitConfig
from ..data.digest import digest_fasta
from ..data.precursors import enumerate_precursors
from ..data.prospect import ProspectSource, merge_real_labels
from ..models.context import MSContextEncoder
from ..models.registry import build_student, load_checkpoint, save_checkpoint
from ..predict.fast import TorchRunner, predict_library_fast
from ..teacher import get_teacher
from .context_regime import fit_realspeclib
from .stream_pretrain import StreamMix, StreamPretrainCfg, fit_stream_pretrain


@dataclass
class DigestSource:
    """One FASTA + its digestion settings for the pretrain stream (enzyme 'unspecific' ->
    immunopeptidome windows, else tryptic)."""

    fasta: str
    enzyme: str = "trypsin"
    missed: int = 2
    min_len: int = 7
    max_len: int = 30
    min_charge: int = 2
    max_charge: int = 4
    max_var_mods: int = 1


@dataclass
class PretrainCfg:
    # Online teacher-distill warmup: enumerate `sources` live over `passes` full digests, sweep
    # NCE per-peptide in [nce_min, nce_max], label in `chunk_size` teacher calls. Collision
    # energy always comes from the sweep (never fabricated), so a real CE axis is learned.
    enabled: bool = True
    sources: list[DigestSource] = field(default_factory=list)
    teacher: str = "alphapeptdeep"  # fake | alphapeptdeep
    instrument: str = "Lumos"
    detector: str = "FTMS"  # teacher acquisition -> ms_context factors (peptdeep = Orbitrap/HCD)
    fragmentation: str = "HCD"
    device: str = "cpu"  # teacher device (peptdeep); student device is RunConfig.device
    batch_size: int = 256
    lr: float = 1e-3
    nce_min: float = 20.0
    nce_max: float = 40.0
    passes: int = 1
    chunk_size: int = 10000
    # Emit every charge per peptide (consecutively, so they share a mini-batch) instead of
    # sampling one. Charge only reaches the MS2/CCS heads, which learn it from the contrast
    # between charges of the same peptide — sampling never shows them that. Costs
    # len(charges)x teacher time.
    all_charge_states: bool = True
    # Early stop the stream when MS2 loss plateaus (student saturated the teacher). 0 = off.
    patience: int = 0
    min_delta: float = 1e-3
    check_every: int = 200
    warmup_steps: int = 500
    mod_align_weight: float = 1.0


@dataclass
class TrainCfg:
    enabled: bool = True
    record: str = "prospect"
    meta: str = "TUM_third_pool_meta_data.parquet"
    zip: str = "TUM_third_pool.zip"
    shards: list[int] = field(default_factory=lambda: [0])
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # ms2, irt, raw_rt
    dataset: str | None = None  # label for the val best-per-entry reduction
    instrument: str = (
        "Lumos"  # pool-level MS instrument (PROSPECT ~ Lumos); set per-source as more are added
    )
    mod_align_weight: float = 1.0


@dataclass
class ExportCfg:
    enabled: bool = False
    opset: int = 17


@dataclass
class BenchCfg:
    enabled: bool = False
    fasta: str = ""
    repeats: int = 3


@dataclass
class RunConfig:
    out: str = "runs/exp"
    preset: str = "small"
    device: str = "auto"
    seed: int = 0
    model_in: str | None = None  # load this checkpoint instead of building (export/bench only)
    pretrain: PretrainCfg = field(default_factory=PretrainCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
    bench: BenchCfg = field(default_factory=BenchCfg)

    @classmethod
    def from_toml(cls, path: str | Path) -> "RunConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        pre = raw.get("pretrain", {})
        sources = [DigestSource(**s) for s in pre.pop("sources", [])]
        return cls(
            **{k: raw[k] for k in ("out", "preset", "device", "seed", "model_in") if k in raw},
            pretrain=PretrainCfg(sources=sources, **pre),
            train=TrainCfg(**_tuple_lw(raw.get("train", {}))),
            export=ExportCfg(**raw.get("export", {})),
            bench=BenchCfg(**raw.get("bench", {})),
        )


def _tuple_lw(d: dict) -> dict:
    """TOML arrays are lists; loss_weights must be a 3-tuple."""
    d = dict(d)
    if "loss_weights" in d:
        d["loss_weights"] = tuple(d["loss_weights"])
    return d


def _accelerator(device: str) -> str:
    """Map a device string to a Lightning accelerator ('auto'/'cpu'/'mps'/'gpu')."""
    return {"cuda": "gpu"}.get(device, device)


def _digest_cfg(s: DigestSource) -> DigestConfig:
    return DigestConfig(
        enzyme=s.enzyme,
        missed_cleavages=s.missed,
        min_length=s.min_len,
        max_length=s.max_len,
        min_charge=s.min_charge,
        max_charge=s.max_charge,
        max_variable_mods=s.max_var_mods,
    )


def _stream_mixes(cfg: PretrainCfg) -> list[StreamMix]:
    """Map each pretrain source to a StreamMix (enzyme 'unspecific' -> immunopeptidome windows)."""
    return [
        StreamMix(
            name=s.enzyme,
            kind="unspecific" if s.enzyme == "unspecific" else "tryptic",
            fasta=s.fasta,
            cfg=_digest_cfg(s),
            min_len=s.min_len,
            max_len=s.max_len,
        )
        for s in cfg.sources
    ]


def _release_accelerator_cache(acc: str) -> None:
    """Return cached device memory after dropping a stage's objects.

    Freeing Python references does not hand accelerator memory back on its own, so a stage
    boundary has to ask explicitly or the next stage allocates against a still-full cache.
    """
    import torch

    if acc == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif acc == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_real(cfg: TrainCfg, log):
    """Decode the chosen shards of a PROSPECT pool into one RealLabels, held in memory.

    Only the shard *read* streams. Decoding accumulates: every shard's labels are materialized
    in ``parts`` and ``merge_real_labels`` then builds a second full copy, so peak use is about
    twice the final size, and the result (a list of per-example objects each carrying its own
    MS2 array) stays resident for training. Shard count is therefore bounded by RAM, not by
    download time — budget accordingly before adding shards.
    """
    src = ProspectSource(cfg.record)
    meta = src.read(cfg.meta)
    parts = []
    for name, ann in src.iter_annotation_shards(cfg.zip, cfg.shards):
        parts.append(src.to_labels(meta, ann))
        log(f"  decoded {name.split('/')[-1]}: {len(parts[-1].precursors)} examples")
        del ann
    del meta  # not needed past decoding; keeps it out of the merge's peak
    gc.collect()
    return merge_real_labels(parts)


def _run_pretrain(cfg: RunConfig, model, encoder, acc, log):
    p = cfg.pretrain
    assert encoder is not None  # guaranteed by need_encoder in run_pipeline
    kw = {} if p.teacher == "fake" else {"device": p.device, "instrument": p.instrument}
    teacher = get_teacher(p.teacher, **kw)
    spc = StreamPretrainCfg(
        mixes=_stream_mixes(p),
        nce_range=(p.nce_min, p.nce_max),
        chunk_size=p.chunk_size,
        batch_size=p.batch_size,
        passes=p.passes,
        all_charge_states=p.all_charge_states,
        lr=p.lr,
        seed=cfg.seed,
        patience=p.patience,
        min_delta=p.min_delta,
        check_every=p.check_every,
        warmup_steps=p.warmup_steps,
        instrument=p.instrument,
        detector=p.detector,
        fragmentation=p.fragmentation,
        mod_align_weight=p.mod_align_weight,
    )
    log(
        f"[pretrain] stream: {[m.name for m in spc.mixes]}, NCE {spc.nce_range}, "
        f"{spc.passes} pass(es), chunk {spc.chunk_size}"
    )
    return fit_stream_pretrain(model, encoder, teacher, spc, accelerator=acc, log=log)


def run_pipeline(cfg: RunConfig, log=print) -> dict:
    """Execute the enabled stages in order. Returns a summary dict of per-stage metrics."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = _accelerator(cfg.device)
    summary: dict = {}

    model = load_checkpoint(cfg.model_in) if cfg.model_in else build_student(cfg.preset)
    # The MSContextEncoder is needed by the real-data sink AND by streaming pretrain (the NCE
    # sweep); build once and share it across both. Either enabled stage conditions on it.
    need_encoder = cfg.train.enabled or cfg.pretrain.enabled
    encoder = MSContextEncoder(context_dim=model.cfg.context_dim) if need_encoder else None
    log(f"student '{cfg.preset}' — {model.num_parameters():,} params (device={cfg.device})")

    if cfg.pretrain.enabled:
        mod = _run_pretrain(cfg, model, encoder, acc, log)
        summary["pretrain"] = {k: float(v) for k, v in mod.trainer.callback_metrics.items()}
        log(f"[pretrain] {summary['pretrain']}")
        # Release the pretrain module before the train stage allocates. `mod` is needed only
        # for the metrics just extracted, but it transitively pins the Lightning Trainer ->
        # dataloader -> _StreamingDataset -> teacher -> peptdeep's models (hundreds of MB).
        # Left alive, that sits resident through the whole real-data stage; observed as an OOM
        # killing a single-shard train stage on a laptop. The student weights are unaffected —
        # `model` is the shared backbone and is held separately.
        del mod
        gc.collect()
        _release_accelerator_cache(acc)

    runbook = None
    dataset_index = None
    if cfg.train.enabled:
        log(f"[train] streaming shards {cfg.train.shards} of {cfg.train.zip}")
        real = _load_real(cfg.train, log)
        module = fit_realspeclib(
            model,
            real,
            dataset_name=cfg.train.dataset,
            instrument=cfg.train.instrument,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            lr=cfg.train.lr,
            loss_weights=cfg.train.loss_weights,
            seed=cfg.seed,
            accelerator=acc,
            encoder=encoder,
            mod_align_weight=cfg.train.mod_align_weight,
            enable_progress_bar=False,
        )
        runbook = module.runbook
        dataset_index = module.dataset_index
        summary["train"] = {k: float(v) for k, v in module.trainer.callback_metrics.items()}
        summary["dataset_index"] = dataset_index
        log(f"[train] {summary['train']}")

    if encoder is not None:
        summary["energy_curve"] = _energy_curve(encoder, cfg.pretrain.nce_min, cfg.pretrain.nce_max)

    ckpt = out / "model.ckpt"
    # Persist the context too, or the artifact can only make base (context-free) predictions.
    save_checkpoint(model, ckpt, encoder=encoder, runbook=runbook, dataset_index=dataset_index)
    log(f"saved {ckpt}")

    if cfg.export.enabled:
        from ..predict.onnx import export_onnx  # optional [onnx] extra — import only if used

        onnx_path = out / "model.onnx"
        export_onnx(model, onnx_path, opset=cfg.export.opset)
        summary["export"] = str(onnx_path)
        log(f"[export] {onnx_path}")

    if cfg.bench.enabled and cfg.bench.fasta:
        summary["bench"] = _bench(model, cfg.bench, log)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _energy_curve(encoder, ce_min: float, ce_max: float, n: int = 5) -> dict:
    """ms_context magnitude across the energy range — a quick read on what the encoder learned."""
    import torch

    ces = [ce_min + (ce_max - ce_min) * i / (n - 1) for i in range(n)]
    zeros = torch.zeros(n, dtype=torch.long)
    energy = torch.tensor(ces, dtype=torch.float32)
    with torch.no_grad():
        norms = encoder(zeros, zeros, zeros, energy=energy).norm(dim=1)
    return {round(c, 1): round(float(v), 4) for c, v in zip(ces, norms)}


def _bench(model, cfg: BenchCfg, log) -> dict:
    dcfg = DigestConfig()
    precs = enumerate_precursors(digest_fasta(cfg.fasta, dcfg), dcfg, SplitConfig())
    runner = TorchRunner(model, "cpu")
    predict_library_fast(runner, precs[: min(2000, len(precs))])  # warmup
    best = float("inf")
    for _ in range(cfg.repeats):
        t = time.perf_counter()
        lib = predict_library_fast(runner, precs)
        best = min(best, time.perf_counter() - t)
    rate = len(precs) / best if best > 0 else float("inf")
    log(f"[bench] {len(precs)} precursors, {len(lib)} rows, best {best:.3f}s -> {rate:,.0f}/s")
    return {"precursors": len(precs), "rows": len(lib), "best_s": best, "rate": rate}


__all__ = [
    "RunConfig",
    "DigestSource",
    "PretrainCfg",
    "TrainCfg",
    "ExportCfg",
    "BenchCfg",
    "run_pipeline",
]
