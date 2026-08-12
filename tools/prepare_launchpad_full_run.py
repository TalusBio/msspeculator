"""Build the small source bundle consumed by the prepared ETL/training jobs.

The PROSPECT data itself is deliberately not copied here. Workers discover shards from the
vendored Zenodo catalog/index and use S3 only as a read-through archive cache, keeping the
Launchpad worker's 32 GB scratch volume independent of the full corpus size.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / ".launchpad" / "full-run-stage"
S3_PREFIX = "s3://terraform-workstations-bucket/jspaezp/20241022_prospect"
PREPARED_PREFIX = "s3://terraform-workstations-bucket/jspaezp/pepdistill-prepared/v1"
TRAIN_OUTPUT_PREFIX = "s3://terraform-workstations-bucket/jspaezp/pepdistill-training/full-v1"
TRAIN_RUNS = (
    # run name, model preset, real-training batch size, real-training learning rate
    ("flash", "flash", 256, 3e-4),
    ("small-2h", "small-2h", 256, 3e-4),
    ("small", "small", 256, 3e-4),
    ("base-4h", "base-4h", 256, 3e-4),
    ("base", "base", 256, 3e-4),
    ("flash-lr1e4", "flash", 256, 1e-4),
    ("flash-b512", "flash", 512, 3e-4),
)
PRETRAIN_FASTA = ROOT / "fasta_ignore" / "ecoli_k12.fasta"
UNIPROT_FASTA_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=proteome%3AUP000000625"
)


def ensure_pretrain_fasta() -> Path:
    """Return the cached E. coli K-12 FASTA, downloading it once when absent."""
    if PRETRAIN_FASTA.exists() and PRETRAIN_FASTA.stat().st_size > 0:
        return PRETRAIN_FASTA
    PRETRAIN_FASTA.parent.mkdir(parents=True, exist_ok=True)
    temporary = PRETRAIN_FASTA.with_suffix(".fasta.part")
    try:
        print(f"downloading pretrain FASTA from {UNIPROT_FASTA_URL}")
        with urllib.request.urlopen(UNIPROT_FASTA_URL, timeout=120) as response:
            with temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
        if temporary.stat().st_size == 0:
            raise RuntimeError("UniProt returned an empty FASTA")
        os.replace(temporary, PRETRAIN_FASTA)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return PRETRAIN_FASTA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out: Path = args.out.resolve()

    if out.exists():
        shutil.rmtree(out)
    (out / "src").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "src", out / "src")
    shutil.copytree(ROOT / "rust", out / "rust", ignore=shutil.ignore_patterns("target"))
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(ROOT / name, out / name)

    runs = out / "runs"
    runs.mkdir()
    shutil.copy2(ROOT / "runs" / "prepare-full.toml", runs / "prepare-full.toml")
    for run_name, model_preset, batch_size, learning_rate in TRAIN_RUNS:
        (runs / f"prepared-cloud-{run_name}.toml").write_text(
            f'''out = "cloud-output-{run_name}"
remote_output_prefix = "{TRAIN_OUTPUT_PREFIX}/{run_name}"
preset = "{model_preset}"
activation = "gelu_tanh"
device = "cpu"
seed = 0

[tracking]
enabled = true
project = "pepdistill"
name = "{run_name}"
group = "full-v1"
tags = ["full-nontest", "{model_preset}", "{run_name}"]

[diagnostics]
enabled = true
butterflies = 3
every_n_epochs = 1
interval_minutes = 60.0
render_initial = true

[pretrain]
enabled = true
teacher = "alphapeptdeep"
passes = 2
chunk_size = 10000
patience = 5
checkpoint_every_steps = 500

[[pretrain.sources]]
fasta = "pretrain.fasta"
enzyme = "trypsin"
missed = 2
min_len = 7
max_len = 30
min_charge = 2
max_charge = 4
max_var_mods = 1

[train]
enabled = true
prepared_prefix = "{PREPARED_PREFIX}"
epochs = 60
batch_size = {batch_size}
num_workers = 0
model_threads = 4
lr = {learning_rate}
early_stop_patience = 5
early_stop_min_delta = 0.001
'''
        )
    fasta = ensure_pretrain_fasta()
    shutil.copy2(fasta, out / "pretrain.fasta")

    print(f"prepared {out}")
    print(
        "staged files: source, Rust extension, lockfile, prepare catalog, "
        f"{len(TRAIN_RUNS)} train configs, E. coli pretrain FASTA"
    )
    print(f"data source (read-only): {S3_PREFIX}")


if __name__ == "__main__":
    main()
