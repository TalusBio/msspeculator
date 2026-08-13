"""``pepdistill`` command-line interface.

    run      config.toml           -> trained model (+ optional bench)
    predict  model + FASTA          -> library.parquet
    prepare  config.toml             -> deterministic shard assets

``run`` drives the whole Lightning pipeline from one :class:`RunConfig` (pretrain -> train ->
export -> bench, each stage toggleable); see :mod:`pepdistill.distill.pipeline`. ``predict``
is standalone inference: generate a spectral library from an already-trained model.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer

from .data.config import DigestConfig, SplitConfig
from .data.digest import digest_fasta
from .data.precursors import enumerate_precursors, frame_to_precursors

# torch, pandas and the training pipeline are imported inside the commands that use them, as the
# other commands here already do. At module scope they made `pepdistill prepare` load the whole
# training stack: an ETL worker paid seconds of import and a GB of memory for a path that never
# touches a tensor. Torch is also an optional dependency (the `torch` extra), so importing it here
# would make every command require an install the preparation commands do not need.
from .util import resolve_device


def _require_torch(command: str) -> None:
    """Fail with the command that installs torch, rather than a missing-module traceback.

    Torch lives behind an extra so preparation workers can skip the CUDA wheels, which means a
    valid install can be missing it. Checked by spec so the message arrives before a partial
    import leaves a stack trace pointing at whichever module happened to import torch first.
    """
    if importlib.util.find_spec("torch") is None:
        raise SystemExit(
            f"`pepdistill {command}` needs PyTorch, which is not installed. It lives behind the "
            "`torch` extra so preparation workers can skip the CUDA wheels:\n"
            "    uv sync --extra torch"
        )


app = typer.Typer(add_completion=False, help="Distill AlphaPeptDeep into fast spectral libraries.")


def _parse_range(value: str | None, total: int) -> tuple[int, int]:
    if value is None:
        return 0, total
    parts = value.split(":")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise typer.BadParameter("range must be a half-open integer range such as 0:100")
    start, stop = (int(part) for part in parts)
    if not 0 <= start <= stop <= total:
        raise typer.BadParameter(f"range {value!r} outside catalog of {total} shard task(s)")
    return start, stop


def _parse_partition(value: str) -> tuple[int, int]:
    parts = value.split("/")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise typer.BadParameter("partition must be zero-based INDEX/TOTAL, such as 1/10")
    return int(parts[0]), int(parts[1])


@app.command()
def prepare(
    config: Path = typer.Argument(..., exists=True, readable=True, help="Prepare config (TOML)."),
    range_spec: Optional[str] = typer.Option(
        None, "--range", help="Half-open global shard range START:STOP (default: all)."
    ),
    partition_spec: Optional[str] = typer.Option(
        None,
        "--partition",
        help="Byte-balanced zero-based array partition INDEX/TOTAL, such as 1/10.",
    ),
    force: bool = typer.Option(False, "--force", help="Rebuild completed shard assets."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Discover and report tasks without processing."
    ),
) -> None:
    """Discover the global shard catalog and prepare one optional range of shards."""
    from .etl.config import PrepareConfig
    from .etl.prospect import balanced_partition_range, ensure_catalog, prepare_range

    cfg = PrepareConfig.load(config)
    catalog = ensure_catalog(cfg, force=force)
    if range_spec is not None and partition_spec is not None:
        raise typer.BadParameter("--range and --partition are mutually exclusive")
    if partition_spec is not None:
        partition_index, partitions = _parse_partition(partition_spec)
        try:
            start, stop, estimated_bytes = balanced_partition_range(
                catalog, partition_index, partitions
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(
            f"partition {partition_index}/{partitions}: estimated raw input "
            f"{estimated_bytes / 1e9:.2f} GB"
        )
    else:
        start, stop = _parse_range(range_spec, len(catalog["tasks"]))
    typer.echo(f"catalog: {len(catalog['tasks']):,} shard task(s); selected [{start}:{stop})")
    if dry_run:
        for task in catalog["tasks"][start:stop]:
            typer.echo(
                f"{task['ordinal']:06d} {task['source_id']} shard={int(task['shard_index']):06d} "
                f"dataset={task['dataset']}"
            )
        return
    prepare_range(cfg, start=start, stop=stop, force=force, log=typer.echo)


@app.command(name="prepare-finalize")
def prepare_finalize(
    config: Path = typer.Argument(..., exists=True, readable=True, help="Prepare config (TOML)."),
) -> None:
    """Validate every shard asset and write the worker-independent training manifest."""
    from .etl.config import PrepareConfig
    from .etl.prospect import finalize_catalog

    manifest = finalize_catalog(PrepareConfig.load(config), log=typer.echo)
    typer.echo(
        f"manifest ready -> {manifest['catalog_uri'].removesuffix('/catalog.json')}/manifest.json"
    )


@app.command(name="prepare-status")
def prepare_status(
    config: Path = typer.Argument(..., exists=True, readable=True, help="Prepare config (TOML)."),
    count_only: bool = typer.Option(
        False, "--count-only", help="Only discover the catalog size; do not check shard outputs."
    ),
) -> None:
    """Report how many catalog shard assets are complete."""
    from .etl.config import PrepareConfig
    from .etl.prospect import catalog_status

    status = catalog_status(PrepareConfig.load(config), count_only=count_only)
    typer.echo(
        f"prepared shards: {status['complete']:,}/{status['total']:,} complete "
        f"({status['missing']:,} missing)"
    )


@app.command(name="curation-report")
def curation_report(
    prepared: Path = typer.Argument(..., exists=True, readable=True, help="Prepared shard."),
    metadata: Path = typer.Argument(
        ..., exists=True, readable=True, help="Matching PROSPECT metadata Parquet."
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output JSON report."),
    annotations_out: Optional[Path] = typer.Option(
        None,
        "--annotations-out",
        help="Optional per-spectrum curation annotations Parquet.",
    ),
    half_max_fraction: float = typer.Option(
        0.5, min=0.0, max=1.0, help="Observed apex-intensity fraction to retain."
    ),
    min_in_window_psms: int = typer.Option(
        4,
        "--min-in-window-psms",
        min=1,
        help="Minimum PSMs in the shared apex window required for a peptidoform.",
    ),
    max_psms_per_context: int = typer.Option(
        2,
        "--max-psms-per-context",
        min=1,
        help="Maximum retained PSMs per charge/acquisition context.",
    ),
    width_anchor_min_psms: int = typer.Option(
        8,
        "--width-anchor-min-psms",
        min=2,
        help="Minimum half-height observations for a run-width anchor.",
    ),
    energy_bucket_width: float = typer.Option(
        1.0,
        "--energy-bucket-width",
        min=0.001,
        help="Collision-energy width used to identify equivalent contexts.",
    ),
    min_run_width_minutes: float = typer.Option(
        0.05,
        "--min-run-width-minutes",
        min=0.0,
        help="Narrowest plausible per-run acceptance window (apex +/- width/2).",
    ),
    max_run_width_minutes: float = typer.Option(
        0.25,
        "--max-run-width-minutes",
        min=0.0,
        help="Widest plausible per-run acceptance window (apex +/- width/2).",
    ),
) -> None:
    """Analyze shared apex-window filtering and context deduplication on one shard."""
    from .etl.curation import analyze_prepared_curation

    analysis = analyze_prepared_curation(
        prepared,
        metadata,
        half_max_fraction=half_max_fraction,
        min_in_window_psms=min_in_window_psms,
        max_psms_per_context=max_psms_per_context,
        width_anchor_min_psms=width_anchor_min_psms,
        energy_bucket_width=energy_bucket_width,
        min_run_width_minutes=min_run_width_minutes,
        max_run_width_minutes=max_run_width_minutes,
    )
    analysis.write(out, annotations_out)
    selection = analysis.report["selection"]
    ceiling = analysis.report["achievable_ceiling"]

    def mean(key: str) -> float:
        return ceiling[key]["mean"] or float("nan")

    typer.echo(
        f"selected {selection['selected_rows']:,}/{analysis.report['input']['rows']:,} PSMs "
        f"({selection['selected_fraction_of_rows']:.1%}); achievable ceiling "
        f"all={mean('all'):.4f}, apex-window={mean('within_apex_window'):.4f}, "
        f"selected={mean('selected'):.4f} -> {out}"
    )


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, readable=True, help="Run config (TOML)."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Override config `out`."),
    device: Optional[str] = typer.Option(None, help="Override device: auto | cpu | mps | cuda."),
    model_in: Optional[str] = typer.Option(
        None,
        "--model-in",
        help="Initialize from a local checkpoint path or object-store URI.",
    ),
    no_pretrain: bool = typer.Option(False, help="Disable the teacher-distill pretrain stage."),
    no_train: bool = typer.Option(False, help="Disable the real-speclib train stage."),
) -> None:
    """Run the training pipeline described by a TOML config."""
    _require_torch("run")
    from .distill.pipeline import RunConfig, run_pipeline

    cfg = RunConfig.from_toml(config)
    if out is not None:
        cfg = replace(cfg, out=str(out))
    if device is not None:
        cfg = replace(cfg, device=device)
    if model_in is not None:
        cfg = replace(cfg, model_in=model_in)
    if no_pretrain:
        cfg = replace(cfg, pretrain=replace(cfg.pretrain, enabled=False))
    if no_train:
        cfg = replace(cfg, train=replace(cfg.train, enabled=False))

    run_pipeline(cfg, log=typer.echo)
    typer.echo(f"done -> {cfg.out}/summary.json")


@app.command()
def predict(
    model: Path = typer.Option(..., exists=True, readable=True, help="Checkpoint (.ckpt)."),
    out: Path = typer.Option(..., "--out", "-o", help="Output library parquet."),
    fasta: Optional[Path] = typer.Option(None, exists=True, help="Digest this FASTA to predict."),
    precursors: Optional[Path] = typer.Option(None, exists=True, help="Or use a precursor table."),
    min_intensity: float = 0.01,
    batch_size: int = 4096,
    device: str = typer.Option("auto", help="auto | cpu | mps | cuda."),
    nce: Optional[float] = typer.Option(
        None, help="Collision energy for context-aware MS2 (needs a ckpt with a saved encoder)."
    ),
    ms_context: Optional[str] = typer.Option(
        None,
        "--ms-context",
        help=(
            "Full acquisition context 'INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY', "
            "e.g. Lumos::FTMS::HCD::30."
        ),
    ),
    enzyme: str = "trypsin",
    missed: int = 2,
    min_len: int = 7,
    max_len: int = 30,
    min_charge: int = 2,
    max_charge: int = 4,
    max_var_mods: int = 1,
) -> None:
    """Predict a spectral library from a trained student (vectorized, length-bucketed)."""
    _require_torch("predict")
    import pandas as pd
    import torch

    from .models.registry import load_checkpoint
    from .predict.fast import TorchRunner, predict_library_fast
    from .predict.library import write_library

    if (fasta is None) == (precursors is None):
        raise typer.BadParameter("provide exactly one of --fasta or --precursors")

    if fasta is not None:
        dcfg = DigestConfig(
            enzyme=enzyme,
            missed_cleavages=missed,
            min_length=min_len,
            max_length=max_len,
            min_charge=min_charge,
            max_charge=max_charge,
            max_variable_mods=max_var_mods,
        )
        precs = enumerate_precursors(digest_fasta(fasta, dcfg), dcfg, SplitConfig())
    else:
        precs = frame_to_precursors(pd.read_parquet(precursors))

    # Resolve MS context: --ms-context "INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY" wins;
    # else --nce is a shorthand for unknown categoricals + energy only.
    instrument, detector, fragmentation = "", "", ""
    if ms_context is not None:
        parts = ms_context.split("::")
        if len(parts) != 4:
            raise typer.BadParameter(
                "--ms-context must be 'INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY'"
            )
        instrument, detector, fragmentation, nce = parts[0], parts[1], parts[2], float(parts[3])

    ms_ctx_vec = None
    if nce is not None:
        from .models.registry import load_context

        ctx = load_context(model)
        if ctx is None or ctx.encoder is None:
            raise typer.BadParameter(
                f"{model} has no saved acquisition encoder; can't condition MS2"
            )
        enc = ctx.encoder
        ms_ctx_vec = (
            enc(
                torch.tensor([enc.instrument_id(instrument)]),
                torch.tensor([enc.detector_id(detector)]),
                torch.tensor([enc.fragmentation_id(fragmentation)]),
                torch.tensor([float(nce)]),
            )
            .detach()
            .numpy()[0]
        )
        typer.echo(
            f"context-aware: {instrument or '-'}::{detector or '-'}::"
            f"{fragmentation or '-'}::{nce} "
            f"-> ms_context |v|={float((ms_ctx_vec**2).sum() ** 0.5):.3f}"
        )
    runner = TorchRunner(load_checkpoint(model), resolve_device(device), ms_context=ms_ctx_vec)

    t0 = time.perf_counter()
    lib = predict_library_fast(runner, precs, batch_size=batch_size, min_intensity=min_intensity)
    dt = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    write_library(lib, out)
    rate = len(precs) / dt if dt > 0 else float("inf")
    typer.echo(
        f"{len(precs)} precursors -> {len(lib)} fragment rows in {dt:.2f}s "
        f"({rate:,.0f} precursors/s) -> {out}"
    )


@app.command(name="export-rust")
def export_rust(
    model: Path = typer.Option(..., exists=True, readable=True, help="Checkpoint (.ckpt)."),
    out: Path = typer.Option(..., "--out", "-o", help="Output .safetensors artifact."),
) -> None:
    """Export a checkpoint to a self-contained .safetensors for the Rust predict CLI."""
    _require_torch("export-rust")
    from .export import export_safetensors

    out.parent.mkdir(parents=True, exist_ok=True)
    export_safetensors(model, out)
    typer.echo(f"exported -> {out}")


@app.command()
def diagnose(
    model: Path = typer.Option(..., exists=True, readable=True, help="Checkpoint (.ckpt)."),
    out: Path = typer.Option(..., "--out", "-o", help="Diagnostic output directory."),
    teacher: str = typer.Option("alphapeptdeep", help="Reference teacher name."),
    butterflies: int = typer.Option(3, min=1, help="Number of reference spectra."),
    device: str = typer.Option("cpu", help="auto | cpu | mps | cuda"),
) -> None:
    """Render the same fixed diagnostic panel used during training for one checkpoint."""
    _require_torch("diagnose")
    from .models.context import MSContextEncoder
    from .models.registry import load_checkpoint, load_context
    from .teacher import get_teacher
    from .training_diagnostics import TrainingDiagnosticRenderer

    resolved = resolve_device(device)
    student = load_checkpoint(model, map_location=str(resolved)).to(resolved)
    context = load_context(model, map_location=str(resolved))
    encoder = context.encoder if context is not None else None
    if encoder is None:
        encoder = MSContextEncoder(context_dim=student.cfg.context_dim)
    encoder = encoder.to(resolved)
    teacher_kwargs = {} if teacher == "fake" else {"device": "cpu", "instrument": "Lumos"}
    renderer = TrainingDiagnosticRenderer(
        out,
        get_teacher(teacher, **teacher_kwargs),
        butterflies=butterflies,
    )
    result = renderer.render(student, encoder, "checkpoint")
    for name, path in result.paths.items():
        typer.echo(f"{name} -> {path}")
    typer.echo(
        "metrics: " + ", ".join(f"{name}={value:.4f}" for name, value in result.metrics.items())
    )


if __name__ == "__main__":
    app()
