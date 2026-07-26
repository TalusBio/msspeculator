"""``pepdistill`` command-line interface — two commands.

    run      config.toml           -> trained model (+ optional onnx / bench)
    predict  model + FASTA          -> library.parquet

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


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, readable=True, help="Run config (TOML)."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Override config `out`."),
    device: Optional[str] = typer.Option(None, help="Override device: auto | cpu | mps | cuda."),
    no_pretrain: bool = typer.Option(False, help="Disable the teacher-distill pretrain stage."),
    no_train: bool = typer.Option(False, help="Disable the real-speclib train stage."),
) -> None:
    """Run the training pipeline described by a TOML config."""
    cfg = RunConfig.from_toml(config)
    if out is not None:
        cfg = replace(cfg, out=str(out))
    if device is not None:
        cfg = replace(cfg, device=device)
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
        help="Full acquisition context 'ANALYZER::FRAGMENTATION::CE', e.g. FTMS::HCD::30.",
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

    # Resolve acquisition context: --ms-context "ANALYZER::FRAG::CE" wins; else --nce (CE only).
    analyzer, fragmentation = "", ""
    if ms_context is not None:
        parts = ms_context.split("::")
        if len(parts) != 3:
            raise typer.BadParameter("--ms-context must be 'ANALYZER::FRAGMENTATION::CE'")
        analyzer, fragmentation, nce = parts[0], parts[1], float(parts[2])

    if runtime == "onnx" or str(model).endswith(".onnx"):
        from .predict.onnx import OnnxRunner  # optional [onnx] extra — import only if used

        if nce is not None:
            raise typer.BadParameter(
                "context-aware MS2 (--nce/--ms-context) needs the torch runtime"
            )
        runner = OnnxRunner(model)
    else:
        ctx_acq = None
        if nce is not None:
            from .models.registry import load_context

            ctx = load_context(model)
            if ctx is None or ctx.encoder is None:
                raise typer.BadParameter(
                    f"{model} has no saved acquisition encoder; can't condition MS2"
                )
            enc = ctx.encoder
            ce = torch.tensor([float(nce)])
            ctx_acq = enc.encode_batch(ce, analyzer, fragmentation, "cpu").detach().numpy()[0]
            typer.echo(
                f"context-aware: {analyzer or '-'}::{fragmentation or '-'}::{nce} "
                f"-> ctx_acq |v|={float((ctx_acq**2).sum() ** 0.5):.3f}"
            )
        runner = TorchRunner(load_checkpoint(model), resolve_device(device), ctx_acq=ctx_acq)

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


if __name__ == "__main__":
    app()
