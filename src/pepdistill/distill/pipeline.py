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
- **train** — real-speclib sink over a prepared Parquet manifest, streamed from local storage or
  object storage, with per-dataset ``chrom_context`` and factor-driven ``ms_context``.
- **export** — ONNX. **bench** — library-generation throughput on a FASTA digest.

Inference (predict a library from a finished model) is deliberately NOT here — it is the
standalone ``predict`` command.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
import tomllib
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import fsspec
import lightning as L
import torch

from ..data.config import DigestConfig, SplitConfig
from ..data.digest import digest_fasta
from ..data.prepared import PreparedManifest, PreparedStreamingDataset
from ..data.precursors import enumerate_precursors
from ..models.context import ChromRunbook, MSContextEncoder
from ..models.registry import PRESETS, build_student, load_checkpoint, load_context, save_checkpoint
from ..predict.fast import TorchRunner, predict_library_fast
from ..teacher import get_teacher
from .context_regime import establish_rt_norm, fit_realspeclib_datasets
from .stream_pretrain import StreamMix, StreamPretrainCfg, fit_stream_pretrain


def _configure_runtime_warnings() -> None:
    """Filter known dependency chatter while preserving model/data warnings.

    Lightning 2.6 still probes torch 2.3+'s deprecated ``LeafSpec`` API.  Its worker-count
    suggestion is also not applicable to our stateful streaming iterable: adding workers
    without explicit shard partitioning would duplicate each epoch.  Keep these two known,
    non-actionable messages out of long training logs; genuine warnings continue through.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        category=DeprecationWarning,
        module=r"lightning\.pytorch\.utilities\._pytree",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The '.*_dataloader' does not have many workers.*",
        category=UserWarning,
        module=r"lightning\.pytorch\.trainer\.connectors\.data_connector",
    )


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
    # OneCycle is on by default. Streaming datasets have no cheap length, so this explicit step
    # count should be adjusted when a corpus is expected to produce substantially more/less work.
    onecycle_max_lr: float | None = 1e-3
    onecycle_total_steps: int | None = 2500
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    # This is an inference-ready warm start, not an optimizer or streaming-cursor snapshot.
    checkpoint_every_steps: int = 500


@dataclass
class TrainCfg:
    enabled: bool = True
    prepared_prefix: str | None = None
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # ms2, irt, raw_rt
    mod_align_weight: float = 1.0
    # Validation spectral-angle early stop; 0 disables it. Patience is in real-data epochs.
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-3
    # Examples held for cross-shard shuffling. Shards are also visited in a per-epoch shuffled
    # order, so this only has to break up within-shard correlation. 0 disables shuffling
    # entirely (sequential shard order) for debugging a data problem.
    shuffle_buffer: int = 50_000


# What the real-data stage trains on — and therefore the population the RT affine is estimated
# from. ONE constant for both, because they must not drift: the affine is set once and is
# permanent for the run, so a mismatched population is a silent, unrecoverable change of scale.
#
# Train only. Both other splits are genuinely held out: val is what the run is evaluated on,
# and test is untouched by this pipeline end to end — it is not trained on and it is not
# normalised from. This is a deliberate departure from the pre-streaming `fit_realspeclib`,
# which trained on `split != "val"` and so consumed test as well.
def _train_cfg(raw: dict) -> TrainCfg:
    d = dict(raw)
    if "sources" in d or any(key in d for key in ("record", "meta", "zip", "shards")):
        raise ValueError(
            "[train.sources] was removed; run the prepared ETL first and set "
            "[train] prepared_prefix = \"...\""
        )
    prepared_prefix = d.pop("prepared_prefix", None)
    if "loss_weights" in d:
        d["loss_weights"] = tuple(d["loss_weights"])
    cfg = TrainCfg(prepared_prefix=prepared_prefix, **d)
    if cfg.enabled and not cfg.prepared_prefix:
        raise ValueError(
            "[train] requires prepared_prefix; run the prepared ETL before training"
        )
    return cfg


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
    # Mirror durable artifacts as they are produced while retaining the local output directory.
    remote_output_prefix: str | None = None
    preset: str = "small"
    activation: str | None = None  # override preset activation for controlled retraining sweeps
    device: str = "auto"
    seed: int = 0
    model_in: str | None = None  # optional checkpoint to initialize pretrain/train/export/bench
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
            **{
                k: raw[k]
                for k in (
                    "out",
                    "remote_output_prefix",
                    "preset",
                    "activation",
                    "device",
                    "seed",
                    "model_in",
                )
                if k in raw
            },
            pretrain=PretrainCfg(sources=sources, **pre),
            train=_train_cfg(raw.get("train", {})),
            export=ExportCfg(**raw.get("export", {})),
            bench=BenchCfg(**raw.get("bench", {})),
        )


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


def _artifact_mirror(prefix: str, log=print):
    """Build a synchronous, fail-loud artifact mirror rooted at an fsspec URI."""
    fs, root = fsspec.core.url_to_fs(prefix)
    root = root.rstrip("/")

    def mirror(path: str | Path) -> str:
        source = Path(path)
        target = f"{root}/{source.name}" if root else source.name
        parent = target.rpartition("/")[0]
        if parent:
            fs.makedirs(parent, exist_ok=True)
        with source.open("rb") as src, fs.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        uri = f"{prefix.rstrip('/')}/{source.name}"
        log(f"[artifact] mirrored {source.name} -> {uri}")
        return uri

    return mirror


def _runbook_for_index(
    existing: ChromRunbook | None, dataset_index: dict[str, int], context_dim: int
) -> ChromRunbook:
    """Reuse a checkpoint's runbook, expanding it without discarding learned rows if needed."""
    needed = max(dataset_index.values(), default=0)
    if existing is None:
        return ChromRunbook(n_datasets=needed, context_dim=context_dim)
    if existing.context_dim != context_dim:
        raise ValueError(
            f"checkpoint runbook context_dim {existing.context_dim} != model's {context_dim}"
        )
    if existing.n_datasets >= needed:
        return existing
    expanded = ChromRunbook(n_datasets=needed, context_dim=context_dim)
    rows = existing.n_datasets + 1  # include neutral row zero
    with torch.no_grad():
        expanded.emb.weight[:rows].copy_(existing.emb.weight)
        expanded.log_scale.weight[:rows].copy_(existing.log_scale.weight)
        expanded.shift.weight[:rows].copy_(existing.shift.weight)
    return expanded


def _run_pretrain(cfg: RunConfig, model, encoder, acc, out: Path, mirror, log):
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
        onecycle_max_lr=p.onecycle_max_lr,
        onecycle_total_steps=p.onecycle_total_steps,
        onecycle_pct_start=p.onecycle_pct_start,
        onecycle_div_factor=p.onecycle_div_factor,
        onecycle_final_div_factor=p.onecycle_final_div_factor,
    )
    log(
        f"[pretrain] stream: {[m.name for m in spc.mixes]}, NCE {spc.nce_range}, "
        f"{spc.passes} pass(es), chunk {spc.chunk_size}, "
        f"OneCycle max_lr={spc.onecycle_max_lr} over {spc.onecycle_total_steps} step(s)"
    )
    return fit_stream_pretrain(
        model,
        encoder,
        teacher,
        spc,
        accelerator=acc,
        log=log,
        checkpoint_every=p.checkpoint_every_steps,
        checkpoint_path=out / "pretrain-latest.ckpt",
        artifact_mirror=mirror,
    )


def run_pipeline(cfg: RunConfig, log=print) -> dict:
    """Execute the enabled stages in order. Returns a summary dict of per-stage metrics."""
    _configure_runtime_warnings()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = _accelerator(cfg.device)
    summary: dict = {}
    mirror = _artifact_mirror(cfg.remote_output_prefix, log) if cfg.remote_output_prefix else None
    if cfg.remote_output_prefix:
        summary["remote_output_prefix"] = cfg.remote_output_prefix
        log(f"artifacts will be mirrored to {cfg.remote_output_prefix.rstrip('/')}/")

    loaded_context = None
    if cfg.model_in:
        model = load_checkpoint(cfg.model_in)
        loaded_context = load_context(cfg.model_in)
        log(f"loaded checkpoint {cfg.model_in}")
    else:
        model_cfg = PRESETS[cfg.preset]
        if cfg.activation is not None:
            model_cfg = replace(model_cfg, activation=cfg.activation)
        model = build_student(model_cfg)
    # The MSContextEncoder is needed by the real-data sink AND by streaming pretrain (the NCE
    # sweep); build once and share it across both. Either enabled stage conditions on it.
    need_encoder = cfg.train.enabled or cfg.pretrain.enabled
    if need_encoder and loaded_context is not None and loaded_context.encoder is not None:
        encoder = loaded_context.encoder
    else:
        encoder = MSContextEncoder(context_dim=model.cfg.context_dim) if need_encoder else None
    if need_encoder:
        # Registry loaders return inference-ready modules; put a checkpoint initializer back in
        # training mode before Lightning constructs its Trainer (which otherwise warns about an
        # eval-mode module at the start of a resumed run).
        model.train()
        assert encoder is not None
        encoder.train()
        if loaded_context is not None and loaded_context.runbook is not None:
            loaded_context.runbook.train()
    log(f"student '{cfg.preset}' — {model.num_parameters():,} params (device={cfg.device})")

    if cfg.pretrain.enabled:
        mod = _run_pretrain(cfg, model, encoder, acc, out, mirror, log)
        summary["pretrain"] = {k: float(v) for k, v in mod.trainer.callback_metrics.items()}
        log(f"[pretrain] {summary['pretrain']}")
        pretrain_ckpt = out / "pretrain.ckpt"
        save_checkpoint(model, pretrain_ckpt, encoder=encoder)
        if mirror is not None:
            mirror(pretrain_ckpt)
        summary["pretrain_checkpoint"] = str(pretrain_ckpt)
        log(f"[pretrain] saved {pretrain_ckpt}")
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
        assert encoder is not None, "need_encoder covers cfg.train.enabled"
        if not cfg.train.prepared_prefix:
            raise ValueError("[train] requires prepared_prefix; run the prepared ETL first")
        # Seed before the runbook is built: fit_realspeclib_datasets deliberately does NOT seed
        # globally, so a second call there would reset the stream after the context modules had
        # already drawn from it.
        L.seed_everything(cfg.seed, verbose=False)
        prepared_manifest = PreparedManifest.load(cfg.train.prepared_prefix)
        dataset_index = prepared_manifest.datasets
        train_ds = PreparedStreamingDataset(
            prepared_manifest, encoder, frozenset({"train"}), seed=cfg.seed,
            shuffle_buffer=cfg.train.shuffle_buffer,
        )
        val_ds = PreparedStreamingDataset(
            prepared_manifest, encoder, frozenset({"val"}), seed=cfg.seed,
            shuffle_buffer=0,
        )
        log(
            f"[train] prepared prefix: {cfg.train.prepared_prefix}; "
            f"{len(prepared_manifest.chunks)} chunk(s), "
            f"{len(prepared_manifest.datasets)} dataset(s)"
        )
        # Whether the affine was set here or inherited is the difference between a cold start
        # and a continued curriculum, for a value that is permanent once set — so say which.
        if establish_rt_norm(model, [prepared_manifest.irt_stats]):
            log(f"[train] RT affine set: mean {float(model.rt_mean):.4g}, "
                f"std {float(model.rt_std):.4g}")
        else:
            log("[train] RT affine inherited from an earlier stage; not recalibrated")
        # Size by the HIGHEST row, not the count: rows are contiguous from 1 only when the
        # index was built from scratch. resolve_dataset_index(existing=...) keeps whatever rows
        # a continued curriculum already had, which can be sparse, and len() would then size an
        # embedding that the top row indexes past.
        runbook = _runbook_for_index(
            loaded_context.runbook if loaded_context else None,
            dataset_index,
            model.cfg.context_dim,
        )
        log(
            f"[train] streaming prepared chunks directly; "
            f"train rows={len(train_ds):,}, val rows={len(val_ds):,}"
        )
        module = fit_realspeclib_datasets(
            model,
            train_ds,
            val_ds,
            runbook=runbook,
            dataset_index=dataset_index,
            encoder=encoder,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            lr=cfg.train.lr,
            loss_weights=cfg.train.loss_weights,
            seed=cfg.seed,
            accelerator=acc,
            mod_align_weight=cfg.train.mod_align_weight,
            early_stop_patience=cfg.train.early_stop_patience,
            early_stop_min_delta=cfg.train.early_stop_min_delta,
            enable_progress_bar=False,
            progress_metrics_path=out / "train_metrics.jsonl",
            checkpoint_dir=out,
            artifact_mirror=mirror,
        )
        summary["train"] = {k: float(v) for k, v in module.trainer.callback_metrics.items()}
        summary["dataset_index"] = dataset_index
        log(f"[train] {summary['train']}")

    if encoder is not None:
        summary["energy_curve"] = _energy_curve(encoder, cfg.pretrain.nce_min, cfg.pretrain.nce_max)

    ckpt = out / "model.ckpt"
    # Persist the context too, or the artifact can only make base (context-free) predictions.
    save_checkpoint(model, ckpt, encoder=encoder, runbook=runbook, dataset_index=dataset_index)
    if mirror is not None:
        mirror(ckpt)
    log(f"saved {ckpt}")

    if cfg.export.enabled:
        from ..predict.onnx import export_onnx  # optional [onnx] extra — import only if used

        onnx_path = out / "model.onnx"
        export_onnx(model, onnx_path, opset=cfg.export.opset)
        if mirror is not None:
            mirror(onnx_path)
        summary["export"] = str(onnx_path)
        log(f"[export] {onnx_path}")

    if cfg.bench.enabled and cfg.bench.fasta:
        summary["bench"] = _bench(model, cfg.bench, log)

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    if mirror is not None:
        mirror(summary_path)
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
