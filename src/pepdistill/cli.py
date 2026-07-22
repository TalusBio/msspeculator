"""``pepdistill`` command-line interface.

Staged pipeline, each stage reading/writing on-disk artifacts:

    digest    FASTA               -> precursors.parquet
    label     precursors.parquet  -> labels/ (teacher soft labels)
    distill   precursors + labels -> model.ckpt
    predict   model + FASTA       -> library.parquet
    pipeline  FASTA               -> model.ckpt (+ optional library) in one shot
    benchmark model                -> throughput report

The ``run_*`` functions hold the real logic (plain args, real defaults) so both the
typer commands and ``pipeline`` can call them without typer's sentinel defaults leaking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Distill AlphaPeptDeep into fast spectral libraries.")


# ----------------------------------------------------------------------------- core logic
def run_digest(
    fasta: Path,
    out: Path,
    enzyme: str = "trypsin",
    missed: int = 2,
    min_len: int = 7,
    max_len: int = 30,
    min_charge: int = 2,
    max_charge: int = 4,
    max_var_mods: int = 1,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    salt: str = "pepdistill-v1",
) -> Path:
    from .data.config import DigestConfig, SplitConfig
    from .data.digest import digest_fasta
    from .data.precursors import enumerate_precursors, precursors_to_frame

    dcfg = DigestConfig(
        enzyme=enzyme,
        missed_cleavages=missed,
        min_length=min_len,
        max_length=max_len,
        min_charge=min_charge,
        max_charge=max_charge,
        max_variable_mods=max_var_mods,
    )
    scfg = SplitConfig(train=train, val=val, test=test, salt=salt)

    peptides = digest_fasta(fasta, dcfg)
    precursors = enumerate_precursors(peptides, dcfg, scfg)
    frame = precursors_to_frame(precursors)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    counts = frame["split"].value_counts().to_dict()
    typer.echo(
        f"{len(peptides)} peptides -> {len(frame)} precursors "
        f"(train/val/test = {counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('test', 0)}) "
        f"-> {out}"
    )
    return out


def run_label(
    precursors: Path,
    out: Path,
    teacher: str = "fake",
    device: str = "cpu",
    nce: float = 30.0,
    instrument: str = "Lumos",
) -> Path:
    import pandas as pd

    from .data.precursors import frame_to_precursors
    from .teacher import get_teacher, labels_to_frames

    precs = frame_to_precursors(pd.read_parquet(precursors))
    kwargs = {} if teacher == "fake" else {"device": device, "nce": nce, "instrument": instrument}
    tchr = get_teacher(teacher, **kwargs)

    t0 = time.perf_counter()
    labels = tchr.predict(precs)
    dt = time.perf_counter() - t0

    prec_df, frag_df = labels_to_frames(precs, labels)
    out.mkdir(parents=True, exist_ok=True)
    prec_df.to_parquet(out / "prec_labels.parquet", index=False)
    frag_df.to_parquet(out / "frag_labels.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({"teacher": tchr.name, "n": len(precs)}, indent=2))
    typer.echo(f"labeled {len(precs)} precursors with '{tchr.name}' in {dt:.2f}s -> {out}")
    return out


def _load_dataset_split(precursors: Path, labels: Path):
    import pandas as pd

    from .data.precursors import frame_to_precursors
    from .distill.dataset import DistillDataset
    from .teacher.base import labels_from_frames

    precs = frame_to_precursors(pd.read_parquet(precursors))
    prec_df = pd.read_parquet(labels / "prec_labels.parquet")
    frag_df = pd.read_parquet(labels / "frag_labels.parquet")
    labs = labels_from_frames(prec_df, frag_df)
    if len(precs) != len(labs):
        raise typer.BadParameter("precursor/label count mismatch; re-run `label`.")

    buckets: dict[str, tuple[list, list]] = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for prec, lab in zip(precs, labs):
        buckets[prec.split][0].append(prec)
        buckets[prec.split][1].append(lab)
    return {k: DistillDataset(ps, ls) for k, (ps, ls) in buckets.items()}


def run_distill(
    precursors: Path,
    labels: Path,
    out: Path,
    preset: str = "small",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> Path:
    from .distill.trainer import TrainConfig, train
    from .models.registry import build_student, save_checkpoint

    datasets = _load_dataset_split(precursors, labels)
    model = build_student(preset)
    typer.echo(f"student '{preset}' — {model.num_parameters():,} params")

    tcfg = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr, device=device, seed=seed)
    val = datasets["val"] if len(datasets["val"]) else None
    history = train(model, datasets["train"], val, tcfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out)
    out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))

    last = history[-1] if history else {}
    typer.echo(
        f"trained {epochs} epochs -> {out}\n"
        f"  val spectral_angle={last.get('val_spectral_angle', float('nan')):.4f} "
        f"rt_mae={last.get('val_rt_mae', float('nan')):.3f} "
        f"ccs_mae={last.get('val_ccs_mae', float('nan')):.3f}"
    )
    return out


def _make_runner(model_path: Path, runtime: str, device: str):
    """Build a ModelRunner from a checkpoint (torch) or an .onnx file."""
    from .predict.fast import TorchRunner
    from .util import resolve_device

    if runtime == "onnx" or str(model_path).endswith(".onnx"):
        from .predict.onnx import OnnxRunner

        return OnnxRunner(model_path)
    from .models.registry import load_checkpoint

    return TorchRunner(load_checkpoint(model_path), resolve_device(device))


def run_predict(
    model: Path,
    out: Path,
    fasta: Optional[Path] = None,
    precursors: Optional[Path] = None,
    min_intensity: float = 0.01,
    batch_size: int = 4096,
    device: str = "auto",
    runtime: str = "torch",
    enzyme: str = "trypsin",
    missed: int = 2,
    min_len: int = 7,
    max_len: int = 30,
    min_charge: int = 2,
    max_charge: int = 4,
    max_var_mods: int = 1,
) -> Path:
    import pandas as pd

    from .data.config import DigestConfig, SplitConfig
    from .data.digest import digest_fasta
    from .data.precursors import enumerate_precursors, frame_to_precursors
    from .predict.fast import predict_library_fast
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

    runner = _make_runner(model, runtime, device)
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
    return out


def run_distill_stream(
    out: Path,
    fasta: Optional[Path] = None,
    teacher: str = "fake",
    preset: str = "flash",
    total_batches: int = 5000,
    warmup_batches: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "auto",
    seed: int = 0,
    val_n: int = 2000,
    eval_every: int = 250,
    nce: float = 30.0,
    instrument: str = "Lumos",
) -> Path:
    """Online distillation: teacher labels batches live; random warmup then FASTA."""
    import numpy as np

    from .data.config import DigestConfig
    from .distill.streaming import build_val_set, curriculum_batches, estimate_norm
    from .distill.trainer import TrainConfig, train_streaming
    from .models.registry import build_student, save_checkpoint
    from .teacher import get_teacher
    from .util import resolve_device

    dev = resolve_device(device)
    dcfg = DigestConfig()
    kwargs = {} if teacher == "fake" else {"device": "cpu", "nce": nce, "instrument": instrument}
    tchr = get_teacher(teacher, **kwargs)

    model = build_student(preset)
    typer.echo(f"student '{preset}' — {model.num_parameters():,} params, device={dev}")

    rng = np.random.default_rng(seed)
    model.set_norm(*estimate_norm(tchr, dcfg, rng))
    val_ds = build_val_set(tchr, dcfg, rng, val_n, fasta)

    batches = curriculum_batches(
        tchr, dcfg, rng, batch_size, total_batches, warmup_batches, fasta
    )
    tcfg = TrainConfig(batch_size=batch_size, lr=lr, device=dev, seed=seed)
    history = train_streaming(
        model, batches, total_batches, tcfg, val_ds, eval_every, log=typer.echo
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out)
    out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    last = history[-1] if history else {}
    typer.echo(
        f"streamed {total_batches} batches -> {out}\n"
        f"  val spectral_angle={last.get('val_spectral_angle', float('nan')):.4f} "
        f"rt_mae={last.get('val_rt_mae', float('nan')):.3f} "
        f"ccs_mae={last.get('val_ccs_mae', float('nan')):.3f}"
    )
    return out


# ----------------------------------------------------------------------------- commands
@app.command()
def digest(
    fasta: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(..., "--out", "-o", help="Output precursors parquet."),
    enzyme: str = "trypsin",
    missed: int = 2,
    min_len: int = 7,
    max_len: int = 30,
    min_charge: int = 2,
    max_charge: int = 4,
    max_var_mods: int = 1,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    salt: str = "pepdistill-v1",
) -> None:
    """Digest a FASTA into a precursor table with a deterministic train/val/test split."""
    run_digest(
        fasta,
        out,
        enzyme,
        missed,
        min_len,
        max_len,
        min_charge,
        max_charge,
        max_var_mods,
        train,
        val,
        test,
        salt,
    )


@app.command()
def label(
    precursors: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(..., "--out", "-o", help="Output labels directory."),
    teacher: str = typer.Option("fake", help="Teacher: fake | alphapeptdeep"),
    device: str = "cpu",
    nce: float = 30.0,
    instrument: str = "Lumos",
) -> None:
    """Generate teacher soft labels for a precursor table."""
    run_label(precursors, out, teacher, device, nce, instrument)


@app.command()
def distill(
    precursors: Path = typer.Option(..., exists=True, readable=True),
    labels: Path = typer.Option(..., exists=True, file_okay=False),
    out: Path = typer.Option(..., "--out", "-o", help="Output checkpoint (.ckpt)."),
    preset: str = typer.Option("small", help="Student preset: tiny | small | base"),
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> None:
    """Train a student model against cached teacher labels."""
    run_distill(precursors, labels, out, preset, epochs, batch_size, lr, device, seed)


@app.command()
def predict(
    model: Path = typer.Option(..., exists=True, readable=True, help="Checkpoint (.ckpt) or .onnx."),
    out: Path = typer.Option(..., "--out", "-o", help="Output library parquet."),
    fasta: Optional[Path] = typer.Option(None, exists=True, help="Digest this FASTA to predict."),
    precursors: Optional[Path] = typer.Option(None, exists=True, help="Or use a precursor table."),
    min_intensity: float = 0.01,
    batch_size: int = 4096,
    device: str = typer.Option("auto", help="auto | cpu | mps | cuda (torch runtime)"),
    runtime: str = typer.Option("torch", help="torch | onnx"),
    enzyme: str = "trypsin",
    missed: int = 2,
    min_len: int = 7,
    max_len: int = 30,
    min_charge: int = 2,
    max_charge: int = 4,
    max_var_mods: int = 1,
) -> None:
    """Predict a spectral library from a trained student (vectorized, length-bucketed)."""
    run_predict(
        model,
        out,
        fasta=fasta,
        precursors=precursors,
        min_intensity=min_intensity,
        batch_size=batch_size,
        device=device,
        runtime=runtime,
        enzyme=enzyme,
        missed=missed,
        min_len=min_len,
        max_len=max_len,
        min_charge=min_charge,
        max_charge=max_charge,
        max_var_mods=max_var_mods,
    )


@app.command(name="distill-stream")
def distill_stream(
    out: Path = typer.Option(..., "--out", "-o", help="Output checkpoint (.ckpt)."),
    fasta: Optional[Path] = typer.Option(None, exists=True, help="FASTA for post-warmup phase."),
    teacher: str = typer.Option("fake", help="Teacher: fake | alphapeptdeep"),
    preset: str = typer.Option("flash", help="Student preset: flash | tiny | small | base"),
    total_batches: int = 5000,
    warmup_batches: int = typer.Option(1000, help="Random-peptide warmup before FASTA."),
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = typer.Option("auto", help="auto | cpu | mps | cuda"),
    seed: int = 0,
    val_n: int = 2000,
    eval_every: int = 250,
    nce: float = 30.0,
    instrument: str = "Lumos",
) -> None:
    """Online distillation with a live in-loop teacher; random warmup then FASTA."""
    run_distill_stream(
        out,
        fasta=fasta,
        teacher=teacher,
        preset=preset,
        total_batches=total_batches,
        warmup_batches=warmup_batches,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
        val_n=val_n,
        eval_every=eval_every,
        nce=nce,
        instrument=instrument,
    )


@app.command()
def export(
    model: Path = typer.Option(..., exists=True, readable=True, help="Checkpoint (.ckpt)."),
    out: Path = typer.Option(..., "--out", "-o", help="Output .onnx file."),
    opset: int = 17,
) -> None:
    """Export a trained student to ONNX (dynamic batch/length; needs the onnx extra)."""
    from .models.registry import load_checkpoint
    from .predict.onnx import export_onnx

    m = load_checkpoint(model)
    if m.cfg.backbone == "transformer":
        typer.echo(
            "note: transformer attention exports with dynamic batch but the legacy tracer "
            "bakes in the length; predict() buckets by length so use torch runtime for "
            "transformers, or use a cnn preset for fully-dynamic ONNX."
        )
    export_onnx(m, out, opset=opset)
    typer.echo(f"exported ONNX -> {out}")


@app.command()
def pipeline(
    fasta: Path = typer.Argument(..., exists=True, readable=True),
    workdir: Path = typer.Option(..., "--workdir", "-w", help="Directory for all artifacts."),
    teacher: str = "fake",
    preset: str = "small",
    epochs: int = 20,
    device: str = "cpu",
    library: bool = typer.Option(True, help="Also predict a library from the FASTA at the end."),
) -> None:
    """Run digest -> label -> distill (-> predict) end to end into a working directory."""
    workdir.mkdir(parents=True, exist_ok=True)
    precs = workdir / "precursors.parquet"
    labels_dir = workdir / "labels"
    ckpt = workdir / "model.ckpt"

    run_digest(fasta, precs)
    run_label(precs, labels_dir, teacher=teacher, device=device)
    run_distill(precs, labels_dir, ckpt, preset=preset, epochs=epochs, device=device)
    if library:
        run_predict(ckpt, workdir / "library.parquet", precursors=precs, device=device)


@app.command()
def benchmark(
    model: Path = typer.Option(..., exists=True, readable=True, help="Checkpoint (.ckpt) or .onnx."),
    fasta: Path = typer.Option(..., exists=True),
    device: str = typer.Option("auto", help="auto | cpu | mps | cuda (torch runtime)"),
    runtime: str = typer.Option("torch", help="torch | onnx"),
    min_intensity: float = 0.01,
    repeats: int = 3,
) -> None:
    """Measure student library-generation throughput on a FASTA digest."""
    from .data.config import DigestConfig, SplitConfig
    from .data.digest import digest_fasta
    from .data.precursors import enumerate_precursors
    from .predict.fast import predict_library_fast

    dcfg = DigestConfig()
    precs = enumerate_precursors(digest_fasta(fasta, dcfg), dcfg, SplitConfig())
    runner = _make_runner(model, runtime, device)
    predict_library_fast(runner, precs[: min(2000, len(precs))], min_intensity=min_intensity)  # warmup

    best = float("inf")
    rows = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        lib = predict_library_fast(runner, precs, min_intensity=min_intensity)
        best = min(best, time.perf_counter() - t0)
        rows = len(lib)

    typer.echo(
        f"runtime={runtime} device={device}  precursors={len(precs)} rows={rows}\n"
        f"best of {repeats}: {best:.3f}s -> {len(precs) / best:,.0f} precursors/s"
    )


if __name__ == "__main__":
    app()
