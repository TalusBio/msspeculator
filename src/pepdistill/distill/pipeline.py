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
- **train** — real-speclib sink over one or more PROSPECT pools (each ``[[train.sources]]``
  entry is one pool with its own ChromRunbook row; shards are extracted once and then streamed
  per epoch, never materialised), per-run ``chrom_context`` and factor-driven ``ms_context``.
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

import lightning as L

from ..data.config import DigestConfig, SplitConfig
from ..data.digest import digest_fasta
from ..data.meta_index import MetaIndex, build_meta_index
from ..data.precursors import enumerate_precursors
from ..data.prospect import ProspectSource
from ..data.shard_store import extract_shards, select_members, shard_raw_files
from ..models.context import ChromRunbook, MSContextEncoder
from ..models.registry import build_student, load_checkpoint, save_checkpoint
from ..predict.fast import TorchRunner, predict_library_fast
from ..teacher import get_teacher
from .context_regime import RealSpeclibDataset, establish_rt_norm, fit_realspeclib_datasets
from .real_stream import ShardSpec, StreamingRealDataset, collect_val_examples
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
class TrainSource:
    """One PROSPECT pool contributing to the real-data stage.

    ``instrument`` is per-source, not pool-level: two pools can differ, and acquisition
    factors are never fabricated. ``val_only`` keeps only the examples whose sequence already
    hashes to val and vacuums the rest — leak-free, because ``assign_split`` hashes the
    stripped sequence globally, so a val-hashed sequence is val in every source. This filters
    TO the existing split; it never overrides it.
    """

    record: str
    meta: str
    zip: str
    shards: list[int]
    dataset: str | None = None  # ChromRunbook row key; required once there is >1 source
    instrument: str = "Lumos"
    val_only: bool = False


@dataclass
class TrainCfg:
    enabled: bool = True
    sources: list[TrainSource] = field(default_factory=list)
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # ms2, irt, raw_rt
    mod_align_weight: float = 1.0
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
_TRAIN_SPLITS = frozenset({"train"})

# Pool identity used to live directly under [train]; it is per-source now.
_FLAT_POOL_KEYS = ("record", "meta", "zip", "shards", "dataset", "instrument")


def _check_train_sources(cfg: TrainCfg) -> None:
    """Reject an enabled train stage that cannot train, naming which of the two causes it is.

    Both are decidable from the config alone, so they are checked at parse time — otherwise
    they surface only after every shard has been downloaded and extracted, which on a real pool
    is multiple GB spent to discover a typo. Called again from :func:`_build_train_stage`
    because a :class:`RunConfig` assembled in Python never passes through :func:`_train_cfg`.
    """
    if not cfg.sources:
        raise ValueError(
            "[train] is enabled but declares no [[train.sources]]; add one entry per pool "
            "with record/meta/zip/shards (+ dataset, instrument, val_only)"
        )
    if all(s.val_only for s in cfg.sources):
        raise ValueError(
            f"every [[train.sources]] is val_only ({[s.zip for s in cfg.sources]}); there is "
            "nothing to train on"
        )


def _train_cfg(raw: dict) -> TrainCfg:
    d = dict(raw)
    stale = [k for k in _FLAT_POOL_KEYS if k in d]
    if stale:
        raise ValueError(
            f"[train] no longer takes pool keys {stale}; declare each pool as a "
            "[[train.sources]] entry with record/meta/zip/shards (+ dataset, instrument, "
            "val_only)"
        )
    sources = [TrainSource(**s) for s in d.pop("sources", [])]
    if len(sources) > 1:
        unnamed = [s.zip for s in sources if not s.dataset]
        if unnamed:
            raise ValueError(
                f"every [[train.sources]] needs an explicit dataset name once there is more "
                f"than one source; missing for {unnamed}. The name fixes the ChromRunbook row, "
                "and deriving it from the zip name would let rows drift between runs."
            )
    if "loss_weights" in d:
        d["loss_weights"] = tuple(d["loss_weights"])
    cfg = TrainCfg(sources=sources, **d)
    if cfg.enabled:
        _check_train_sources(cfg)
    return cfg


def resolve_dataset_index(
    sources: list[TrainSource], existing: dict[str, int] | None = None
) -> dict[str, int]:
    """Map dataset name -> ChromRunbook row, in config declaration order, from row 1.

    Row 0 is the neutral/iRT row and is never assigned. Sources sharing a name share a row.
    An ``existing`` index (continuing a curriculum) keeps every row it already has — by
    construction, not by check — and new names append after the highest. The index is baked
    into the exported artifact, so a shifted row would make an old artifact address the wrong
    dataset.

    ``existing`` is validated rather than trusted: a row 0 entry would collide with the
    neutral row, and duplicate rows would make two datasets share one ChromRunbook entry.
    Both are silent corruptions of an artifact, so both raise.
    """
    index = dict(existing or {})
    if 0 in index.values():
        bad = sorted(n for n, r in index.items() if r == 0)
        raise ValueError(
            f"existing dataset_index assigns row 0 to {bad}; row 0 is reserved for the "
            "neutral/iRT row and must not name a dataset"
        )
    if len(set(index.values())) != len(index):
        seen: dict[int, list[str]] = {}
        for n, r in sorted(index.items()):
            seen.setdefault(r, []).append(n)
        clashes = {r: ns for r, ns in seen.items() if len(ns) > 1}
        raise ValueError(
            f"existing dataset_index has duplicate rows {clashes}; each dataset needs its "
            "own ChromRunbook row"
        )
    next_row = max(index.values(), default=0) + 1
    for s in sources:
        name = s.dataset or "default"
        if name in index:
            continue
        index[name] = next_row
        next_row += 1
    return index


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


def _build_train_stage(cfg: RunConfig, log):
    """Extract every source's shards, build its meta index, and assemble what the real-data
    stage needs: the train and val shard lists, one merged meta index, the dataset index (plus
    its inverse), and the iRT sufficient statistics the RT affine is established from.

    Nothing here decodes a fragment. Shards are extracted to local parquet and read per epoch
    by :class:`~pepdistill.distill.real_stream.StreamingRealDataset`, so the resident cost is
    the meta index, not the spectra.
    """
    _check_train_sources(cfg.train)
    dataset_index = resolve_dataset_index(cfg.train.sources)
    names_by_row = {row: name for name, row in dataset_index.items()}
    train_shards: list[ShardSpec] = []
    val_shards: list[ShardSpec] = []
    indices: list[MetaIndex] = []
    stats: list[tuple[int, float, float]] = []
    for s in cfg.train.sources:
        src = ProspectSource(s.record)
        members = select_members(src, s.zip, s.shards)
        # Extract first: raw_files come from each shard's own column, never from its filename
        # (a third-pool shard holds three, none matching the member stem). One extract_shards
        # call per source, not one extract_shard call per member: each remote-zip open re-reads
        # the central directory and, cold, re-opens the whole HTTP stream, and a source can
        # have ~10 shards — enough opens in a row to trip Zenodo's rate limiter on its own.
        paths = extract_shards(src, s.zip, members)
        per_shard = [tuple(shard_raw_files(p)) for p in paths]
        raw_files = sorted({r for rs in per_shard for r in rs})
        index = build_meta_index(src, s.meta, raw_files)
        indices.append(index)
        row = dataset_index[s.dataset or "default"]
        specs = [
            ShardSpec(path=p, raw_files=rs, dataset_id=row, instrument=s.instrument)
            for p, rs in zip(paths, per_shard)
        ]
        val_shards.extend(specs)
        if s.val_only:
            # A val_only source with nothing in val is a no-op that does not look like one: it
            # is still downloaded, extracted, indexed, and still burns a ChromRunbook row that
            # gets baked into the exported artifact — while contributing to neither train nor
            # val. The check is meta-only, so it costs nothing.
            if not index.allowed_keys(raw_files, frozenset({"val"})):
                raise ValueError(
                    f"val_only source {s.zip} shards {s.shards} has no val-hashed sequences; "
                    "it would contribute to neither train nor val"
                )
            log(f"[train] {s.zip}: val_only, {len(specs)} shard(s) held out")
        else:
            train_shards.extend(specs)
            # val_only sources contribute nothing to the RT affine: it is established from the
            # population training actually sees, which is _TRAIN_SPLITS by construction.
            stats.append(index.irt_stats(_TRAIN_SPLITS))
        log(f"[train] {s.zip}: {len(specs)} shard(s), dataset row {row}")

    if not train_shards:
        # Unreachable by construction: _check_train_sources rejects both config-level causes
        # (no sources, all val_only), and a non-val_only source with an empty `shards` list
        # dies earlier still in build_meta_index ("no meta rows ... for raw_files []"). Kept
        # as a cheap invariant so a future change that reintroduces the state is not silent.
        raise ValueError(
            f"no train shards from {[(s.zip, s.shards) for s in cfg.train.sources]}, which "
            "should be unreachable; the source validation above no longer covers some case"
        )

    # One index across sources: keys are (raw_file, scan_number), which is globally unique,
    # so the union cannot collide.
    merged = MetaIndex()
    for ix in indices:
        merged.by_key.update(ix.by_key)
    log(f"[train] meta index: {len(merged.by_key):,} spectra resident")
    return train_shards, val_shards, merged, dataset_index, names_by_row, stats


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
        assert encoder is not None, "need_encoder covers cfg.train.enabled"
        # Seed before the runbook is built: fit_realspeclib_datasets deliberately does NOT seed
        # globally, so a second call there would reset the stream after the context modules had
        # already drawn from it.
        L.seed_everything(cfg.seed, verbose=False)
        train_shards, val_shards, index, dataset_index, names_by_row, stats = _build_train_stage(
            cfg, log
        )
        # Whether the affine was set here or inherited is the difference between a cold start
        # and a continued curriculum, for a value that is permanent once set — so say which.
        if establish_rt_norm(model, stats):
            log(f"[train] RT affine set: mean {float(model.rt_mean):.4g}, "
                f"std {float(model.rt_std):.4g}")
        else:
            log("[train] RT affine inherited from an earlier stage; not recalibrated")
        # Size by the HIGHEST row, not the count: rows are contiguous from 1 only when the
        # index was built from scratch. resolve_dataset_index(existing=...) keeps whatever rows
        # a continued curriculum already had, which can be sparse, and len() would then size an
        # embedding that the top row indexes past.
        runbook = ChromRunbook(
            n_datasets=max(dataset_index.values(), default=0),
            context_dim=model.cfg.context_dim,
        )
        train_ds = StreamingRealDataset(
            train_shards,
            index,
            encoder,
            _TRAIN_SPLITS,
            seed=cfg.seed,
            shuffle_buffer=cfg.train.shuffle_buffer,
        )
        val_examples = collect_val_examples(val_shards, index, encoder, names_by_row, log=log)
        val_ds = RealSpeclibDataset(val_examples) if val_examples else None
        log(f"[train] streaming {len(train_shards)} shard(s); val {len(val_examples)} examples")
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
            enable_progress_bar=False,
        )
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
    "TrainSource",
    "ExportCfg",
    "BenchCfg",
    "run_pipeline",
    "resolve_dataset_index",
]
