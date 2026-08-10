"""``pepdistill`` command-line interface.

    run      config.toml           -> trained model (+ optional onnx / bench)
    predict  model + FASTA          -> library.parquet
    prepare  config.toml             -> deterministic shard assets

``run`` drives the whole Lightning pipeline from one :class:`RunConfig` (pretrain -> train ->
export -> bench, each stage toggleable); see :mod:`pepdistill.distill.pipeline`. ``predict``
is standalone inference: generate a spectral library from an already-trained model.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import typer

from .data.config import DigestConfig, SplitConfig
from .data.digest import digest_fasta
from .data.precursors import enumerate_precursors, frame_to_precursors
from .distill.pipeline import RunConfig, run_pipeline
from .models.registry import load_checkpoint
from .predict.fast import TorchRunner, predict_library_fast
from .predict.library import write_library
from .util import resolve_device

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
    dry_run: bool = typer.Option(False, "--dry-run", help="Discover and report tasks without processing."),
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
    typer.echo(f"manifest ready -> {manifest['catalog_uri'].removesuffix('/catalog.json')}/manifest.json")


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
    model: Path = typer.Option(
        ..., exists=True, readable=True, help="Checkpoint (.ckpt) or .onnx."
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output library parquet."),
    fasta: Optional[Path] = typer.Option(None, exists=True, help="Digest this FASTA to predict."),
    precursors: Optional[Path] = typer.Option(None, exists=True, help="Or use a precursor table."),
    min_intensity: float = 0.01,
    batch_size: int = 4096,
    device: str = typer.Option("auto", help="auto | cpu | mps | cuda (torch runtime)"),
    runtime: str = typer.Option("torch", help="torch | onnx"),
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

    if runtime == "onnx" or str(model).endswith(".onnx"):
        from .predict.onnx import OnnxRunner  # optional [onnx] extra — import only if used

        if nce is not None:
            raise typer.BadParameter(
                "context-aware MS2 (--nce/--ms-context) needs the torch runtime"
            )
        runner = OnnxRunner(model)
    else:
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
    from .export import export_safetensors

    out.parent.mkdir(parents=True, exist_ok=True)
    export_safetensors(model, out)
    typer.echo(f"exported -> {out}")


if __name__ == "__main__":
    app()
