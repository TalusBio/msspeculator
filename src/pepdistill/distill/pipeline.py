"""One config-driven Lightning pipeline: pretrain -> train -> export -> bench.

A single :class:`RunConfig` replaces the old per-stage CLI + hand-rolled trainer. Every stage
is independently toggleable; the model and the collision-energy :class:`ContextEncoder` are
built once and threaded through, so the teacher warmup and the real-data sink share one CE
axis (the teacher's NCE is a factor, not a baked base).

Stages:
- **pretrain** — teacher-distill warmup. Labels a UNION of (FASTA x digest-config) sources
  with the teacher, fits the student to those soft labels (MS2 + RT + CCS).
- **train** — real-speclib sink on a PROSPECT pool (streamed shard-by-shard), with per-run
  ctx_lc and CE-driven ctx_acq.
- **export** — ONNX.
- **bench** — library-generation throughput on a FASTA digest.

Inference (predict a library from a finished model) is deliberately NOT here — it is the
standalone ``predict`` command.
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DigestSource:
    """One FASTA + its digestion settings for the pretrain teacher-labeling union."""

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
    enabled: bool = True
    sources: list[DigestSource] = field(default_factory=list)
    teacher: str = "alphapeptdeep"  # fake | alphapeptdeep
    nce: float = 30.0
    instrument: str = "Lumos"
    device: str = "cpu"  # teacher device (peptdeep); student device is RunConfig.device
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3


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
    ce_context: bool = True  # ctx_acq from collision energy; shared with pretrain
    dataset: str | None = None  # label for the val best-per-entry reduction


@dataclass
class ExportCfg:
    enabled: bool = False
    opset: int = 17


@dataclass
class BenchCfg:
    enabled: bool = False
    fasta: str | None = None
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


def _pretrain_precursors(cfg: PretrainCfg):
    """Digest every (fasta x config) source into one precursor list."""
    from ..data.config import DigestConfig, SplitConfig
    from ..data.digest import digest_fasta
    from ..data.precursors import enumerate_precursors

    scfg = SplitConfig()
    out = []
    for s in cfg.sources:
        dcfg = DigestConfig(
            enzyme=s.enzyme, missed_cleavages=s.missed, min_length=s.min_len,
            max_length=s.max_len, min_charge=s.min_charge, max_charge=s.max_charge,
            max_variable_mods=s.max_var_mods,
        )
        out += enumerate_precursors(digest_fasta(s.fasta, dcfg), dcfg, scfg)
    return out


def _load_real(cfg: TrainCfg, log):
    """Stream the chosen shards of a PROSPECT pool and decode them into one RealLabels."""
    from ..data.prospect import ProspectSource, merge_real_labels

    src = ProspectSource(cfg.record)
    meta = src.read(cfg.meta)
    names = src.annotation_shards(cfg.zip)
    chosen = [names[i] for i in cfg.shards]
    parts = []
    for name in chosen:
        ann = src.read_annotation_streaming(cfg.zip, members=[name])
        parts.append(src.to_labels(meta, ann))
        log(f"  decoded {name.split('/')[-1]}: {len(parts[-1].precursors)} examples")
        del ann
    return merge_real_labels(parts)


def run_pipeline(cfg: RunConfig, log=print) -> dict:
    """Execute the enabled stages in order. Returns a summary dict of per-stage metrics."""
    from ..models.context import ContextEncoder
    from ..models.registry import build_student, load_checkpoint, save_checkpoint

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = _accelerator(cfg.device)
    summary: dict = {}

    model = load_checkpoint(cfg.model_in) if cfg.model_in else build_student(cfg.preset)
    encoder = ContextEncoder(context_dim=model.cfg.context_dim) if cfg.train.ce_context else None
    log(f"student '{cfg.preset}' — {model.num_parameters():,} params (device={cfg.device})")

    if cfg.pretrain.enabled:
        from ..distill.dataset import DistillDataset
        from ..distill.lightning import fit_distill
        from ..teacher import get_teacher

        precs = _pretrain_precursors(cfg.pretrain)
        kw = {} if cfg.pretrain.teacher == "fake" else {
            "device": cfg.pretrain.device, "nce": cfg.pretrain.nce, "instrument": cfg.pretrain.instrument,
        }
        teacher = get_teacher(cfg.pretrain.teacher, **kw)
        t = time.perf_counter()
        labels = teacher.predict(precs)
        pairs = [(p, lab) for p, lab in zip(precs, labels) if lab is not None]
        tr = DistillDataset([p for p, _ in pairs if p.split != "val"],
                            [lab for p, lab in pairs if p.split != "val"])
        va_pairs = [(p, lab) for p, lab in pairs if p.split == "val"]
        va = DistillDataset([p for p, _ in va_pairs], [lab for _, lab in va_pairs]) if va_pairs else None
        log(f"[pretrain] {len(pairs)} teacher labels in {time.perf_counter()-t:.0f}s; fitting {cfg.pretrain.epochs} ep")
        mod = fit_distill(
            model, tr, va, epochs=cfg.pretrain.epochs, batch_size=cfg.pretrain.batch_size,
            lr=cfg.pretrain.lr, accelerator=acc, context_encoder=encoder, distill_ce=cfg.pretrain.nce,
            enable_progress_bar=False,
        )
        summary["pretrain"] = {k: float(v) for k, v in mod.trainer.callback_metrics.items()}
        log(f"[pretrain] {summary['pretrain']}")

    if cfg.train.enabled:
        from ..distill.context_regime import fit_realspeclib

        log(f"[train] streaming shards {cfg.train.shards} of {cfg.train.zip}")
        real = _load_real(cfg.train, log)
        module = fit_realspeclib(
            model, real, dataset=cfg.train.dataset, epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size, lr=cfg.train.lr, loss_weights=cfg.train.loss_weights,
            seed=cfg.seed, accelerator=acc, encoder=encoder, enable_progress_bar=False,
        )
        summary["train"] = {k: float(v) for k, v in module.trainer.callback_metrics.items()}
        summary["source_index"] = module.source_index
        log(f"[train] {summary['train']}")

    ckpt = out / "model.ckpt"
    save_checkpoint(model, ckpt)
    log(f"saved {ckpt}")

    if cfg.export.enabled:
        from ..predict.onnx import export_onnx

        onnx_path = out / "model.onnx"
        export_onnx(model, onnx_path, opset=cfg.export.opset)
        summary["export"] = str(onnx_path)
        log(f"[export] {onnx_path}")

    if cfg.bench.enabled and cfg.bench.fasta:
        summary["bench"] = _bench(model, cfg.bench, log)

    (out / "summary.json").write_text(__import__("json").dumps(summary, indent=2, default=str))
    return summary


def _bench(model, cfg: BenchCfg, log) -> dict:
    from ..data.config import DigestConfig, SplitConfig
    from ..data.digest import digest_fasta
    from ..data.precursors import enumerate_precursors
    from ..predict.fast import TorchRunner, predict_library_fast

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


__all__ = ["RunConfig", "DigestSource", "PretrainCfg", "TrainCfg", "ExportCfg", "BenchCfg", "run_pipeline"]
