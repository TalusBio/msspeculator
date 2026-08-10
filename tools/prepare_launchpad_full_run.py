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
TRAIN_PRESETS = ("flash", "small-2h", "small", "base-4h", "base")
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
    for preset in TRAIN_PRESETS:
        (runs / f"prepared-cloud-{preset}.toml").write_text(
            f'''out = "cloud-output-{preset}"
remote_output_prefix = "{TRAIN_OUTPUT_PREFIX}/{preset}"
preset = "{preset}"
activation = "gelu_tanh"
device = "cpu"
seed = 0

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
batch_size = 256
lr = 0.0003
early_stop_patience = 5
early_stop_min_delta = 0.001
'''
        )
    fasta = ensure_pretrain_fasta()
    shutil.copy2(fasta, out / "pretrain.fasta")

    print(f"prepared {out}")
    print(
        "staged files: source, Rust extension, lockfile, prepare catalog, "
        f"{len(TRAIN_PRESETS)} train configs, E. coli pretrain FASTA"
    )
    print(f"data source (read-only): {S3_PREFIX}")


if __name__ == "__main__":
    main()
