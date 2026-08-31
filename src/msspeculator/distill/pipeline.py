"""One config-driven Lightning pipeline: pretrain -> train.

A single :class:`RunConfig` replaces the old per-stage CLI + hand-rolled trainer. Every stage
is independently toggleable; the model and the shared :class:`MSContextEncoder` are built once
and threaded through, so the teacher warmup and the real-data sink share one acquisition-factor
(instrument/detector/fragmentation/energy) axis (the teacher's NCE is a factor, not a baked base).

Stages:
- **pretrain**, online teacher-distill warmup. Enumerate the ``sources`` live (unspecific
  enzyme -> immunopeptidome windows, else tryptic) with the teacher labeling over an NCE sweep,
  so collision energy comes from the data (never fabricated) and the encoder learns a real CE
  axis. A fixed-energy corpus would just be a dataset that carries its own CE. It needs no
  special mode.
- **train**, real-speclib sink over a prepared Parquet manifest, streamed from local storage or
  object storage, with per-dataset ``chrom_context`` and factor-driven ``ms_context``.

Inference is not here. A finished run writes portable weights, and the Rust CLI generates
libraries from them; see ``docs/adr/0001``.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
import tomllib
import warnings
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from pathlib import Path

import fsspec
import lightning as L
import torch

from ..data.config import DigestConfig
from ..data.digest import resolve_fasta
from ..data.prepared import PreparedManifest, PreparedStreamingDataset
from ..models.context import ChromRunbook, MSContextEncoder
from ..models.registry import PRESETS, build_student, load_checkpoint, load_context, save_checkpoint
from ..teacher import get_teacher
from .context_regime import establish_rt_norm, fit_realspeclib_datasets
from .stream_pretrain import StreamMix, StreamPretrainCfg, fit_stream_pretrain


def _configure_runtime_warnings() -> None:
    """Filter known dependency chatter while preserving model/data warnings.

    Lightning 2.6 still probes torch 2.3+'s deprecated ``LeafSpec`` API. Its generic worker-count
    suggestion ignores the explicitly configured, benchmarked prepared-loader worker count.
    Keep these two known, non-actionable messages out of long logs; genuine warnings continue.
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
    # Mods are per-source for the same reason charge is: a tryptic proteome and an unspecific
    # immunopeptidome source do not carry the same chemistry. Defaults match DigestConfig, which
    # is what every run silently used while these were not plumbed through at all.
    fixed_mods: tuple[str, ...] = ("C[UNIMOD:4]",)
    variable_mods: tuple[tuple[str, float], ...] = (
        ("M[UNIMOD:35]", 0.1),
        ("STY[UNIMOD:21]", 0.001),
        ("K[UNIMOD:1]", 0.001),
        ("K[UNIMOD:121]", 0.001),
    )


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
    # between charges of the same peptide, sampling never shows them that. Costs
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
    # Directory shards are cached in on first read. The full corpus is ~1.2 GiB, so a warmed
    # cache takes S3 out of the training loop entirely; unset streams every epoch from the
    # prefix. Shared across runs against the same corpus, since published shards never change.
    local_cache: str | None = None
    # Hold every decoded shard in RAM after its first read, trading memory for the Parquet
    # decode of every later epoch. The corpus decodes to about 5x its on-disk size, so this
    # needs a machine with the memory to spare; unset re-reads each shard every epoch.
    in_memory: bool = False
    epochs: int = 60
    batch_size: int = 256
    # Keep Polars-backed streaming in the trainer process by default. Forking a DataLoader
    # after Polars has initialized its native thread pool can deadlock on Linux, and measured
    # prepared-shard throughput did not improve with process workers. Polars still parallelizes
    # decode internally; explicit workers remain available for controlled experiments.
    num_workers: int = 0
    # Intra-op threads used by the model process. Four was fastest for the `small` preset on
    # the local CPU benchmark; loader workers run separately and PyTorch pins each of them to 1.
    model_threads: int = 4
    lr: float = 1e-3
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # ms2, irt, raw_rt
    mod_align_weight: float = 1.0
    # Validation spectral-angle early stop; 0 disables it. Patience counts validation checks.
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-3
    # Cut `lr` by `lr_decay_factor` after this many validation checks with no improvement;
    # 0 disables it. Must be below `early_stop_patience`, which is checked, or the run would
    # stop before the rate was ever cut. `early_stop_min_delta` defines improvement for both.
    lr_decay_patience: int = 0
    lr_decay_factor: float = 0.5
    lr_decay_min: float = 0.0
    # Lightning's native wall-clock interval runs validation after the first completed batch
    # that crosses this duration. Long streaming epochs therefore get useful feedback without
    # tying validation cadence to corpus size.
    validation_interval_minutes: float = 60.0

    def __post_init__(self) -> None:
        if self.validation_interval_minutes <= 0:
            raise ValueError("[train] validation_interval_minutes must be positive")
        # Rejected here rather than at the trainer, where it would surface hours into a run.
        if 0 < self.early_stop_patience <= self.lr_decay_patience:
            raise ValueError(
                f"[train] lr_decay_patience {self.lr_decay_patience} must be below "
                f"early_stop_patience {self.early_stop_patience}, or the run stops before the "
                "learning rate is ever cut"
            )
        if not 0.0 < self.lr_decay_factor < 1.0:
            raise ValueError("[train] lr_decay_factor must be between 0 and 1")


@dataclass
class AugmentationCfg:
    """Chemistry-preserving input augmentation shared by both training stages."""

    # Probability per peptide; selected peptides receive exactly one residue substitution.
    residue_substitution_probability: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.residue_substitution_probability <= 1.0:
            raise ValueError(
                "[augmentation] residue_substitution_probability must be between 0 and 1"
            )


# What the real-data stage trains on, and therefore the population the RT affine is estimated
# from. ONE constant for both, because they must not drift: the affine is set once and is
# permanent for the run, so a mismatched population is a silent, unrecoverable change of scale.
#
# Train only. Both other splits are genuinely held out: val is what the run is evaluated on,
# and test is untouched by this pipeline end to end, it is not trained on and it is not
# normalised from. This is a deliberate departure from the pre-streaming `fit_realspeclib`,
# which trained on `split != "val"` and so consumed test as well.
def _train_cfg(raw: dict) -> TrainCfg:
    d = dict(raw)
    if "sources" in d or any(key in d for key in ("record", "meta", "zip", "shards")):
        raise ValueError(
            "[train.sources] was removed; run the prepared ETL first and set "
            '[train] prepared_prefix = "..."'
        )
    prepared_prefix = d.pop("prepared_prefix", None)
    if "loss_weights" in d:
        d["loss_weights"] = tuple(d["loss_weights"])
    cfg = TrainCfg(prepared_prefix=prepared_prefix, **d)
    if cfg.enabled and not cfg.prepared_prefix:
        raise ValueError("[train] requires prepared_prefix; run the prepared ETL before training")
    return cfg


@dataclass
class TrackingCfg:
    """Optional Weights & Biases experiment tracking for both training stages."""

    enabled: bool = False
    project: str = "msspeculator"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    mode: str = "online"  # online | offline
    min_log_interval_seconds: float = 10.0
    # Wall-clock throttling alone can collapse a fast stage into two points. Retain a sparse
    # but bounded trace even when hundreds of optimizer steps fit inside the time interval.
    max_log_interval_steps: int = 100

    def __post_init__(self) -> None:
        if self.min_log_interval_seconds < 0:
            raise ValueError("[tracking] min_log_interval_seconds must be non-negative")
        if self.max_log_interval_steps < 1:
            raise ValueError("[tracking] max_log_interval_steps must be positive")


@dataclass
class DiagnosticsCfg:
    """Low-frequency longitudinal plots for representations, iRT, and reference spectra.

    Disabled by default because plotting is an optional dependency. The butterflies draw against
    the vendored experimental panel, so enabling this needs no teacher installed.
    """

    enabled: bool = False
    butterflies: int = 3
    every_n_epochs: int = 1  # 0 disables epoch renders
    interval_minutes: float = 60.0  # 0 disables wall-clock renders
    render_initial: bool = True

    def __post_init__(self) -> None:
        if self.butterflies < 1:
            raise ValueError("[diagnostics] butterflies must be positive")
        if self.every_n_epochs < 0:
            raise ValueError("[diagnostics] every_n_epochs must be non-negative")
        if self.interval_minutes < 0:
            raise ValueError("[diagnostics] interval_minutes must be non-negative")


@dataclass
class RunConfig:
    out: str = "runs/exp"
    # Mirror durable artifacts as they are produced while retaining the local output directory.
    remote_output_prefix: str | None = None
    preset: str = "small"
    activation: str | None = None  # override preset activation for controlled retraining sweeps
    # Override the preset's dropout, including on a loaded checkpoint (dropout holds no weights,
    # so it is a property of the run rather than of the trained model). Mask generation is about
    # a quarter of the measured training step, so 0.0 is a real speedup where the corpus is large
    # enough to regularize on its own.
    dropout: float | None = None
    device: str = "auto"
    seed: int = 0
    model_in: str | None = None  # optional checkpoint to initialize pretrain/train
    pretrain: PretrainCfg = field(default_factory=PretrainCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    diagnostics: DiagnosticsCfg = field(default_factory=DiagnosticsCfg)
    augmentation: AugmentationCfg = field(default_factory=AugmentationCfg)

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
                    "dropout",
                    "device",
                    "seed",
                    "model_in",
                )
                if k in raw
            },
            pretrain=PretrainCfg(sources=sources, **pre),
            train=_train_cfg(raw.get("train", {})),
            tracking=TrackingCfg(**raw.get("tracking", {})),
            diagnostics=DiagnosticsCfg(**raw.get("diagnostics", {})),
            augmentation=AugmentationCfg(**raw.get("augmentation", {})),
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
        fixed_mods=tuple(s.fixed_mods),
        # TOML spells the variable rules as an inline table, `{ "STY[UNIMOD:21]" = 0.001 }`.
        # It arrives as a dict. Accepted either way so a config and a constructed source agree.
        variable_mods=(
            tuple(s.variable_mods.items())
            if isinstance(s.variable_mods, dict)
            else tuple(s.variable_mods)
        ),
    )


def _stream_mixes(cfg: PretrainCfg, log=None) -> list[StreamMix]:
    """Map each pretrain source to a StreamMix (enzyme 'unspecific' -> immunopeptidome windows)."""
    return [
        StreamMix(
            name=s.enzyme,
            kind="unspecific" if s.enzyme == "unspecific" else "tryptic",
            fasta=str(resolve_fasta(s.fasta, log=log)),
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


def _artifact_mirror(
    prefix: str, log=print, on_mirrored=None, relative_root: str | Path | None = None
):
    """Build a synchronous, fail-loud artifact mirror rooted at an fsspec URI."""
    fs, root = fsspec.core.url_to_fs(prefix)
    root = root.rstrip("/")

    def mirror(path: str | Path) -> str:
        source = Path(path)
        relative = source.name
        if relative_root is not None:
            try:
                relative = source.resolve().relative_to(Path(relative_root).resolve()).as_posix()
            except ValueError:
                pass
        target = f"{root}/{relative}" if root else relative
        parent = target.rpartition("/")[0]
        if parent:
            fs.makedirs(parent, exist_ok=True)
        with source.open("rb") as src, fs.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        uri = f"{prefix.rstrip('/')}/{relative}"
        log(f"[artifact] mirrored {relative} -> {uri}")
        if on_mirrored is not None:
            on_mirrored(source, uri)
        return uri

    return mirror


def _wandb_metric_namespaces(metrics: dict, stage: str) -> dict:
    """Map Lightning's internal names to W&B panel-oriented top-level namespaces.

    W&B groups charts by the first slash-delimited component. A blanket ``train/`` prefix put
    every real-data loss, diagnostic, and per-dataset validation series in one enormous panel.
    Keep internal callback keys stable and translate only at the telemetry boundary.
    """
    train_names = {
        "train_ms2": "ms2_cosine_loss",
        "train_spectral_angle": "spectral_angle",
        "train_irt": "irt_loss",
        "train_rawrt": "rawrt_loss",
        "train_total": "total_loss",
        "train_mod_align": "mod_alignment_loss",
        "train_residue_augmented_fraction": "residue_augmented_fraction",
        "train_irt_labeled_fraction": "irt_labeled_fraction",
    }
    result = {}
    for raw_key, value in metrics.items():
        key = str(raw_key)
        if stage == "train" and key.startswith("val/"):
            _, dataset, metric = key.split("/", 2)
            family = {
                "spectral_angle": "val_sa",
                "irt_mae": "val_irt_mae",
                "rawrt_mae": "val_rawrt_mae",
                "n": "val_n",
            }.get(metric, f"val_{metric}")
            namespaced = f"{family}/{dataset}"
        elif key.startswith(f"diagnostics/{stage}/"):
            namespaced = f"{stage}_diagnostics/{key.removeprefix(f'diagnostics/{stage}/')}"
        elif key in train_names:
            namespaced = f"{stage}_metrics/{train_names[key]}"
        elif key.startswith("lr-") or key == "epoch":
            namespaced = f"{stage}_metrics/{key}"
        else:
            namespaced = f"{stage}_metrics/{key}"
        result[namespaced] = value
    return result


class _RemoteLogThrottle:
    """Wall-clock rate limiter for remote telemetry, holding at most one pending payload.

    Separate from the logger that owns it so the buffering rule can be tested without a W&B
    session. It was a nested class when a bug in exactly this rule silently dropped every
    diagnostics render for a whole run.

    A pending payload is REPLACED, not merged, once a later step arrives, merging across steps
    would attribute one step's metrics to another. That makes forcing the only way for an
    infrequent payload to survive: without it, anything logged between two training batches is
    overwritten within milliseconds. So payloads whose keys start with ``boundary_prefixes``
    (validation, diagnostics) are emitted immediately. The throttle is here to thin per-batch
    samples; a render that happens once an epoch is an event, not a sample.
    """

    def __init__(
        self,
        emit: Callable[[dict, int | None], None],
        *,
        min_interval_seconds: float,
        max_interval_steps: int,
        boundary_prefixes: tuple[str, ...],
    ) -> None:
        self._emit = emit
        self._min_interval_seconds = min_interval_seconds
        self._max_interval_steps = max_interval_steps
        self._boundary_prefixes = boundary_prefixes
        self._last_emit_at: float | None = None
        self._last_emit_step: int | None = None
        self._pending: tuple[dict, int | None] | None = None

    def _is_boundary(self, metrics: dict) -> bool:
        return any(str(key).startswith(self._boundary_prefixes) for key in metrics)

    def flush(self, now: float) -> None:
        if self._pending is None:
            return
        metrics, step = self._pending
        self._pending = None
        self._last_emit_at = now
        self._last_emit_step = step
        self._emit(metrics, step)

    def offer(self, metrics: dict, step: int | None, now: float) -> None:
        boundary = self._is_boundary(metrics)
        if self._pending is not None:
            pending_metrics, pending_step = self._pending
            if pending_step == step:
                # Lightning emits LearningRateMonitor and module metrics in separate calls at
                # the same global step. Merge before rate limiting or whichever callback runs
                # first permanently starves the other metric family.
                pending_metrics.update(metrics)
                if boundary:
                    self.flush(now)
                return
        due = (
            self._last_emit_at is None
            or self._min_interval_seconds <= 0
            or now - self._last_emit_at >= self._min_interval_seconds
            or (
                step is not None
                and self._last_emit_step is not None
                # The completed record available to flush is the pending prior step, so wait
                # until the incoming step is strictly beyond the configured gap.
                and step - self._last_emit_step > self._max_interval_steps
            )
        )
        # A new step proves the pending step is complete. Flush it if the interval is due;
        # otherwise replace it with the newer sample. The current step remains pending so later
        # logger calls at that same step can be merged into one W&B record.
        if due:
            self.flush(now)
        self._pending = (dict(metrics), step)
        if boundary:
            self.flush(now)


def _final_training_metadata(module) -> dict:
    """Describe the validation evidence attached to the final inference checkpoint."""
    trainer = module.trainer
    values = {
        key: float(value.detach().cpu())
        for key, value in trainer.callback_metrics.items()
        if key.startswith("val/")
        and key.endswith("/spectral_angle")
        and torch.is_tensor(value)
        and value.numel() == 1
    }
    return {
        "stage": "train",
        "checkpoint_kind": "model",
        "global_step": int(trainer.global_step),
        "epoch": int(trainer.current_epoch) + 1,
        "validation": {
            "metric": "mean_per_dataset_spectral_angle",
            "values": dict(sorted(values.items())),
            "mean": sum(values.values()) / len(values) if values else None,
            # None when the run finished without ever validating; see the callback's `_save`.
            "validated_at_step": module.last_validation_step,
        },
        # Travels with the checkpoint because it is a claim about the checkpoint: whoever picks
        # this artifact up can see whether its export still predicted what it predicts.
        "export_ms2_max_abs_diff": module.export_ms2_max_abs_diff,
    }


def _wandb_loggers(cfg: RunConfig, out: Path):
    """Create one W&B run with panel-oriented metric namespaces."""
    if not cfg.tracking.enabled:
        return None, None, None
    if cfg.tracking.mode not in {"online", "offline"}:
        raise ValueError("[tracking] mode must be 'online' or 'offline'")
    try:
        import wandb
        from lightning.pytorch.loggers import WandbLogger
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is unavailable; install the 'tracking' extra"
        ) from exc

    try:
        experiment = wandb.init(
            project=cfg.tracking.project,
            entity=cfg.tracking.entity,
            name=cfg.tracking.name or f"{cfg.preset}-seed{cfg.seed}",
            group=cfg.tracking.group,
            tags=cfg.tracking.tags,
            notes=cfg.tracking.notes,
            dir=str(out),
            mode=cfg.tracking.mode,
            job_type="training",
            config=asdict(cfg),
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is unavailable; install the 'tracking' extra"
        ) from exc
    if experiment is None:
        raise RuntimeError("wandb.init() did not return a run")

    class ThrottledWandbLogger(WandbLogger):
        """Limit remote train telemetry by wall time while retaining important boundaries."""

        def __init__(self, *args, stage: str, **kwargs):
            super().__init__(*args, **kwargs)
            self._stage = stage
            self._throttle = _RemoteLogThrottle(
                lambda metrics, step: WandbLogger.log_metrics(self, metrics, step),
                min_interval_seconds=cfg.tracking.min_log_interval_seconds,
                max_interval_steps=cfg.tracking.max_log_interval_steps,
                # Validation and diagnostics are per-epoch events rather than per-batch samples,
                # so they bypass the rate limit instead of being overwritten by the next batch.
                boundary_prefixes=("val_", f"{stage}_diagnostics/"),
            )

        def log_metrics(self, metrics: dict, step: int | None = None) -> None:
            self._throttle.offer(
                _wandb_metric_namespaces(metrics, self._stage), step, time.monotonic()
            )

        def finalize(self, status: str) -> None:
            self._throttle.flush(time.monotonic())
            super().finalize(status)

    root = WandbLogger(experiment=experiment, log_model=False)
    pretrain = ThrottledWandbLogger(experiment=experiment, stage="pretrain")
    train = ThrottledWandbLogger(experiment=experiment, stage="train")
    return root, pretrain, train


def _runbook_for_datasets(
    existing: ChromRunbook | None, datasets: Iterable[str], context_dim: int
) -> ChromRunbook:
    """Reuse a checkpoint's runbook, giving any new dataset a row without moving trained ones.

    Row assignment is the book's own job (:meth:`ChromRunbook.ensure`), so this only has to
    resolve which book to use. The manifest's own numbering is deliberately not consulted: it is
    assigned by sorted position and therefore moves whenever the corpus gains a source.
    """
    book = existing if existing is not None else ChromRunbook(n_datasets=0, context_dim=context_dim)
    if book.context_dim != context_dim:
        raise ValueError(
            f"checkpoint runbook context_dim {book.context_dim} != model's {context_dim}"
        )
    book.ensure(sorted(datasets))
    return book


def _diagnostic_renderer(cfg: RunConfig, out: Path):
    from ..training_diagnostics import DiagnosticAcquisition, TrainingDiagnosticRenderer

    return TrainingDiagnosticRenderer(
        out / "diagnostics",
        acquisition=DiagnosticAcquisition(
            instrument=cfg.pretrain.instrument,
            detector=cfg.pretrain.detector,
            fragmentation=cfg.pretrain.fragmentation,
            nce=(cfg.pretrain.nce_min + cfg.pretrain.nce_max) / 2.0,
        ),
        butterflies=cfg.diagnostics.butterflies,
        nce_range=(cfg.pretrain.nce_min, cfg.pretrain.nce_max),
        # The teacher yardstick and replicate ceiling are published beside the corpus being
        # trained on, so the panel compares the student against the same data it is fitting.
        reference_prefix=cfg.train.prepared_prefix,
    )


def _diagnostic_callback(cfg, renderer, stage, mirror, trainer_logger):
    from ..training_diagnostics import TrainingDiagnosticCallback

    return TrainingDiagnosticCallback(
        renderer,
        stage,
        every_n_epochs=cfg.diagnostics.every_n_epochs,
        interval_minutes=cfg.diagnostics.interval_minutes,
        render_initial=cfg.diagnostics.render_initial,
        artifact_mirror=mirror,
        wandb_logger=trainer_logger,
    )


def _run_pretrain(
    cfg: RunConfig, model, encoder, acc, out: Path, mirror, tracking, trainer_logger, log
):
    p = cfg.pretrain
    assert encoder is not None  # guaranteed by need_encoder in run_pipeline
    mixes = _stream_mixes(p, log)
    kw = {} if p.teacher == "fake" else {"device": p.device, "instrument": p.instrument}
    teacher = get_teacher(p.teacher, **kw)
    renderer = _diagnostic_renderer(cfg, out) if cfg.diagnostics.enabled else None
    spc = StreamPretrainCfg(
        mixes=mixes,
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
        residue_substitution_probability=(cfg.augmentation.residue_substitution_probability),
    )
    log(
        f"[pretrain] stream: {[m.name for m in spc.mixes]}, NCE {spc.nce_range}, "
        f"{spc.passes} pass(es), chunk {spc.chunk_size}, "
        f"OneCycle max_lr={spc.onecycle_max_lr} over {spc.onecycle_total_steps} step(s), "
        f"residue augmentation={spc.residue_substitution_probability:.2%} of peptides"
    )
    module = fit_stream_pretrain(
        model,
        encoder,
        teacher,
        spc,
        accelerator=acc,
        log=log,
        checkpoint_every=p.checkpoint_every_steps,
        checkpoint_path=out / "pretrain-latest.ckpt",
        artifact_mirror=mirror,
        logger=trainer_logger or False,
        callbacks=(
            [_diagnostic_callback(cfg, renderer, "pretrain", mirror, trainer_logger)]
            if renderer is not None
            else None
        ),
    )
    return module, renderer


def run_pipeline(cfg: RunConfig, log=print) -> dict:
    """Execute the enabled stages in order. Returns a summary dict of per-stage metrics."""
    _configure_runtime_warnings()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = _accelerator(cfg.device)
    summary: dict = {}
    final_training_metadata = None
    tracking, pretrain_logger, train_logger = _wandb_loggers(cfg, out)

    def record_artifact(path: Path, uri: str) -> None:
        try:
            name = path.resolve().relative_to(out.resolve()).as_posix()
        except ValueError:
            name = path.name
        summary.setdefault("artifacts", {})[name] = uri
        if tracking is not None:
            tracking.experiment.summary[f"artifacts/{name}"] = uri

    mirror = (
        _artifact_mirror(cfg.remote_output_prefix, log, record_artifact, relative_root=out)
        if cfg.remote_output_prefix
        else None
    )
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
    if cfg.dropout is not None:
        model.set_dropout(cfg.dropout)
        log(f"dropout set to {cfg.dropout:g} for this run")
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
    log(f"student '{cfg.preset}', {model.num_parameters():,} params (device={cfg.device})")

    if cfg.pretrain.enabled:
        mod, diagnostic_renderer = _run_pretrain(
            cfg, model, encoder, acc, out, mirror, tracking, pretrain_logger, log
        )
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
        # killing a single-shard train stage on a laptop. The student weights are unaffected,
        # `model` is the shared backbone and is held separately.
        del mod
        gc.collect()
        _release_accelerator_cache(acc)

    else:
        diagnostic_renderer = None

    runbook = None
    dataset_index = None
    if cfg.train.enabled:
        assert encoder is not None, "need_encoder covers cfg.train.enabled"
        if cfg.diagnostics.enabled and diagnostic_renderer is None:
            diagnostic_renderer = _diagnostic_renderer(cfg, out)
        if not cfg.train.prepared_prefix:
            raise ValueError("[train] requires prepared_prefix; run the prepared ETL first")
        if cfg.train.model_threads < 1:
            raise ValueError("[train] model_threads must be positive")
        torch.set_num_threads(cfg.train.model_threads)
        # Seed before the runbook is built: fit_realspeclib_datasets deliberately does NOT seed
        # globally, so a second call there would reset the stream after the context modules had
        # already drawn from it.
        L.seed_everything(cfg.seed, verbose=False)
        prepared_manifest = PreparedManifest.load(cfg.train.prepared_prefix)
        local_cache = Path(cfg.train.local_cache) if cfg.train.local_cache else None
        train_ds = PreparedStreamingDataset(
            prepared_manifest,
            encoder,
            frozenset({"train"}),
            seed=cfg.seed,
            log=log,
            local_cache=local_cache,
            in_memory=cfg.train.in_memory,
        )
        val_ds = PreparedStreamingDataset(
            prepared_manifest,
            encoder,
            frozenset({"val"}),
            seed=cfg.seed,
            log=log,
            local_cache=local_cache,
            in_memory=cfg.train.in_memory,
        )
        log(
            f"[train] prepared prefix: {cfg.train.prepared_prefix}; "
            f"{len(prepared_manifest.chunks)} chunk(s), "
            f"{len(prepared_manifest.datasets)} dataset(s)"
            + (f"; caching shards in {local_cache}" if local_cache else "")
            + ("; holding decoded shards in RAM" if cfg.train.in_memory else "")
        )
        # Whether the affine was set here or inherited is the difference between a cold start
        # and a continued curriculum, for a value that is permanent once set, so say which.
        if establish_rt_norm(model, [prepared_manifest.irt_stats]):
            log(
                f"[train] RT affine set: mean {float(model.rt_mean):.4g}, "
                f"std {float(model.rt_std):.4g}"
            )
        else:
            log("[train] RT affine inherited from an earlier stage; not recalibrated")
        # The book assigns its own rows, so a corpus that gained a source since this checkpoint
        # was written keeps every trained dataset where it was and only the new one moves.
        runbook = _runbook_for_datasets(
            loaded_context.runbook if loaded_context else None,
            prepared_manifest.datasets,
            model.cfg.context_dim,
        )
        dataset_index = runbook.names
        log(
            f"[train] streaming prepared chunks directly; "
            f"train rows={len(train_ds):,}, val rows={len(val_ds):,}, "
            f"loader workers={cfg.train.num_workers}, model threads={torch.get_num_threads()}, "
            f"validation every {cfg.train.validation_interval_minutes:g} min, "
            f"residue augmentation="
            f"{cfg.augmentation.residue_substitution_probability:.2%} of peptides"
        )
        diagnostic_callbacks = (
            [_diagnostic_callback(cfg, diagnostic_renderer, "train", mirror, train_logger)]
            if diagnostic_renderer is not None
            else []
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
            num_workers=cfg.train.num_workers,
            lr=cfg.train.lr,
            loss_weights=cfg.train.loss_weights,
            seed=cfg.seed,
            accelerator=acc,
            mod_align_weight=cfg.train.mod_align_weight,
            early_stop_patience=cfg.train.early_stop_patience,
            early_stop_min_delta=cfg.train.early_stop_min_delta,
            lr_decay_patience=cfg.train.lr_decay_patience,
            lr_decay_factor=cfg.train.lr_decay_factor,
            lr_decay_min=cfg.train.lr_decay_min,
            residue_substitution_probability=(cfg.augmentation.residue_substitution_probability),
            val_check_interval=timedelta(minutes=cfg.train.validation_interval_minutes),
            check_val_every_n_epoch=None,
            # A sanity pass delays the first training batch and is redundant with the first
            # wall-clock validation check.
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            progress_metrics_path=out / "train_metrics.jsonl",
            checkpoint_dir=out,
            artifact_mirror=mirror,
            logger=train_logger or False,
            callbacks=diagnostic_callbacks,
        )
        summary["train"] = {k: float(v) for k, v in module.trainer.callback_metrics.items()}
        summary["dataset_index"] = dataset_index
        final_training_metadata = _final_training_metadata(module)
        log(f"[train] {summary['train']}")

    if encoder is not None:
        summary["energy_curve"] = _energy_curve(encoder, cfg.pretrain.nce_min, cfg.pretrain.nce_max)

    ckpt = out / "model.ckpt"
    # Persist the context too, or the artifact can only make base (context-free) predictions.
    save_checkpoint(
        model,
        ckpt,
        encoder=encoder,
        runbook=runbook,
        dataset_index=dataset_index,
        training_metadata=final_training_metadata,
    )
    if mirror is not None:
        mirror(ckpt)
    log(f"saved {ckpt}")

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    if mirror is not None:
        mirror(summary_path)
    if tracking is not None:
        tracking.experiment.summary.update(summary)
        tracking.experiment.finish(exit_code=0)
    return summary


def _energy_curve(encoder, ce_min: float, ce_max: float, n: int = 5) -> dict[str, float]:
    """ms_context magnitude across the energy range, a quick read on what the encoder learned.

    Keys are strings because this dict is handed to W&B's run summary, whose encoder builds key
    paths by concatenation and raises on a non-string key. `json.dumps` coerces float keys
    silently, so `summary.json` looked correct while every tracked run died on the summary update
    after training had finished.
    """
    import torch

    ces = [ce_min + (ce_max - ce_min) * i / (n - 1) for i in range(n)]
    zeros = torch.zeros(n, dtype=torch.long)
    energy = torch.tensor(ces, dtype=torch.float32)
    with torch.no_grad():
        norms = encoder(zeros, zeros, zeros, energy=energy).norm(dim=1)
    return {f"{c:.1f}": round(float(v), 4) for c, v in zip(ces, norms)}


__all__ = [
    "RunConfig",
    "DigestSource",
    "PretrainCfg",
    "TrainCfg",
    "TrackingCfg",
    "DiagnosticsCfg",
    "run_pipeline",
]
