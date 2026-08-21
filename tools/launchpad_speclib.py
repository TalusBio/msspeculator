"""Launchpad entrypoint for sharded DIA-NN spectral-library generation.

A whole-proteome library with variable modifications is far larger than a worker's disk: human
tryptic 7-30 with variable CysPAT/Ox/Phospho projects to ~28.5M precursors and ~140 GB at one
placement per peptide, against a 32 GB scratch volume shared by every array child on the host.

So the unit of work is a *logical shard*, not a job. Each child walks its own strided subset of
`--shards` shards and, for each one, generates it, gzips it, uploads it, and deletes the local
copy before starting the next. Peak disk is therefore one shard regardless of how large the
library is or how many children share a host -- raise `--shards` to lower it.

Shards are disjoint by peptide (`pepdistill-cli library --partition`), so the finished library is
the concatenation of its shards with the repeated header rows dropped. Sharding the FASTA instead
would emit a peptide shared by two proteins in both of their shards.

A shard whose object already exists is skipped, which makes the job restartable: cloud jobs here
vanish from `launchpad list` rather than reaching a terminal state, so completion is judged by the
artifacts, and a partial run is resumed by launching the same command again.

    launchpad run tools/launchpad_speclib.py --stage "$STAGE_URI" --array-size 40 \
      --env PEPDISTILL_SPECLIB_ARRAY_SIZE=40 \
      --env PEPDISTILL_SPECLIB_MODEL=s3://bucket/pepdistill-training/.../model.safetensors \
      --env PEPDISTILL_SPECLIB_OUT=s3://bucket/pepdistill-speclib/human-cyspat-v1
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
#
# [tool.launchpad]
# vcpus = 8
# memory = 30000
# job_name = "pepdistill-speclib"
# ///

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Human reviewed canonical proteome. The `stream` endpoint is slow and resets on large queries,
#: so this is retried rather than assumed to succeed on the first call.
UNIPROT_STREAM = (
    "https://rest.uniprot.org/uniprotkb/stream?format=fasta"
    "&query=%28proteome%3A{accession}%29+AND+%28reviewed%3Atrue%29"
)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def fetch(source: str, target: Path) -> Path:
    """Resolve `source` to a local file: an S3 URI, a `uniprot:UP...` proteome, or a path."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if source.startswith("s3://"):
        run(["aws", "s3", "cp", source, str(partial)])
    elif source.startswith("uniprot:"):
        url = UNIPROT_STREAM.format(accession=source.removeprefix("uniprot:"))
        # curl rather than urllib: this transfer resets often enough that the retry matters, and
        # reproducing --retry-all-errors by hand would be the same code with more places to be
        # wrong.
        run(
            [
                "curl",
                "-sSL",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--max-time",
                "3600",
                "-o",
                str(partial),
                url,
            ]
        )
    else:
        path = Path(source)
        if not path.is_file():
            raise SystemExit(f"not a file, an s3:// URI, or a uniprot: reference: {source}")
        return path
    if partial.stat().st_size == 0:
        partial.unlink()
        raise SystemExit(f"{source} resolved to an empty file")
    os.replace(partial, target)
    return target


def build_cli() -> Path:
    """Build the release `pepdistill-cli`, the production inference path.

    Built on the worker because the local machine is a different platform. A missing toolchain is
    an error rather than a fall back to the torch predictor: that path writes parquet, not the
    DIA-NN TSV this job exists to produce, so quietly substituting it would hand back the wrong
    artifact after an hour of compute.
    """
    if shutil.which("cargo") is None:
        raise SystemExit(
            "cargo is not on PATH; this job builds pepdistill-cli from the staged rust/ tree. "
            "Install a Rust toolchain in the worker image, or pre-build the binary for "
            "linux/x86_64 and stage it."
        )
    run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            "rust/Cargo.toml",
            "-p",
            "pepdistill-cli",
        ]
    )
    binary = Path("rust/target/release/pepdistill-cli")
    if not binary.is_file():
        raise SystemExit(f"cargo reported success but {binary} is missing")
    return binary


def s3_exists(uri: str) -> bool:
    return (
        subprocess.run(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                uri.split("/", 3)[2],
                "--key",
                uri.split("/", 3)[3],
            ],
            capture_output=True,
        ).returncode
        == 0
    )


def shard_indices(total: int) -> list[int]:
    """The logical shards this array child owns.

    Strided rather than contiguous so an uneven shard cost spreads across children instead of
    landing on one; `PEPDISTILL_SPECLIB_SHARD` overrides for retrying a single shard by hand.
    """
    single = os.environ.get("PEPDISTILL_SPECLIB_SHARD")
    if single is not None:
        return [int(single)]
    index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    size = os.environ.get("PEPDISTILL_SPECLIB_ARRAY_SIZE")
    if index is None or size is None:
        return list(range(total))
    index, size = int(index), int(size)
    if not 0 <= index < size:
        raise SystemExit(f"array index {index} outside 0..{size - 1}")
    return list(range(index, total, size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    env = os.environ.get
    parser.add_argument(
        "--model",
        default=env("PEPDISTILL_SPECLIB_MODEL"),
        help="exported .safetensors artifact (s3:// or local)",
    )
    parser.add_argument(
        "--out-prefix",
        default=env("PEPDISTILL_SPECLIB_OUT"),
        help="s3:// prefix the shards are written under",
    )
    parser.add_argument("--fasta", default=env("PEPDISTILL_SPECLIB_FASTA", "uniprot:UP000005640"))
    parser.add_argument(
        "--shards",
        type=int,
        default=int(env("PEPDISTILL_SPECLIB_SHARDS", "800")),
        help="logical shards; peak worker disk is one shard, so raise to lower it",
    )
    parser.add_argument("--missed-cleavages", type=int, default=2)
    parser.add_argument("--min-length", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--max-charge", type=int, default=4)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    # CysPAT is the alkylating agent here, not an incidental PTM, so there is no fixed
    # carbamidomethyl rule: a thiol is CysPAT-labelled or bare. The CLI refuses a fixed rule that
    # overlaps a variable one, so leaving the default in place would fail the job outright.
    parser.add_argument(
        "--variable-mod",
        action="append",
        default=None,
        help="repeatable; defaults to CysPAT/Ox/Phospho",
    )
    parser.add_argument(
        "--max-variable-mods",
        type=int,
        default=2,
        help="2 lets a two-cysteine peptide be fully labelled, which 1 cannot",
    )
    parser.add_argument(
        "--ms-context",
        default="Lumos::FTMS::HCD::30",
        help="acquisition factors, or the name of a fitted setup",
    )
    args = parser.parse_args()

    if not args.model or not args.out_prefix:
        raise SystemExit("--model and --out-prefix are required (or set their env vars)")
    if not Path("pyproject.toml").exists():
        raise SystemExit("stage is incomplete; run tools/prepare_launchpad_full_run.py first")
    variable_mods = args.variable_mod or [
        "C[UNIMOD:2057]",  # 6C-CysPAT
        "M[UNIMOD:35]",  # Oxidation
        "STY[UNIMOD:21]",  # Phospho
    ]

    work = Path("speclib-work")
    work.mkdir(exist_ok=True)
    model = fetch(args.model, work / "model.safetensors")
    fasta = fetch(args.fasta, work / "proteome.fasta")
    binary = build_cli()
    prefix = args.out_prefix.rstrip("/")
    mine = shard_indices(args.shards)
    print(f"this child owns {len(mine)} of {args.shards} shards: {mine[:6]}...", flush=True)

    for shard in mine:
        name = f"shard-{shard:05d}-of-{args.shards:05d}"
        destination = f"{prefix}/{name}.tsv.gz"
        if s3_exists(destination):
            print(f"{name}: already published, skipping", flush=True)
            continue
        raw = work / f"{name}.tsv"
        result = run(
            [
                str(binary),
                "library",
                "--model",
                str(model),
                "--fasta",
                str(fasta),
                "--out",
                str(raw),
                "--partition",
                f"{shard}/{args.shards}",
                "--no-fixed-mods",
                "--max-variable-mods",
                str(args.max_variable_mods),
                "--missed-cleavages",
                str(args.missed_cleavages),
                "--min-length",
                str(args.min_length),
                "--max-length",
                str(args.max_length),
                "--min-charge",
                str(args.min_charge),
                "--max-charge",
                str(args.max_charge),
                "--min-intensity",
                str(args.min_intensity),
                "--ms-context",
                args.ms_context,
                *[flag for mod in variable_mods for flag in ("--variable-mod", mod)],
            ],
            capture_output=True,
            text=True,
        )
        print(result.stderr.strip(), flush=True)

        compressed = raw.with_suffix(".tsv.gz")
        with raw.open("rb") as source, gzip.open(compressed, "wb", compresslevel=6) as sink:
            shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
        rows = sum(1 for _ in gzip.open(compressed, "rt")) - 1  # the header is not a transition
        raw_bytes = raw.stat().st_size
        raw.unlink()  # before the upload, so peak disk is one shard rather than one and a half

        run(["aws", "s3", "cp", str(compressed), destination])
        # A sidecar per shard, because a cloud job here disappears from `list` instead of
        # reporting success: these are how a merge decides the library is complete.
        report = work / f"{name}.json"
        report.write_text(
            json.dumps(
                {
                    "shard": shard,
                    "shards": args.shards,
                    "transitions": rows,
                    "bytes_raw": raw_bytes,
                    "bytes_gzip": compressed.stat().st_size,
                    "settings": {
                        "fasta": args.fasta,
                        "model": args.model,
                        "missed_cleavages": args.missed_cleavages,
                        "length": [args.min_length, args.max_length],
                        "charge": [args.min_charge, args.max_charge],
                        "variable_mods": variable_mods,
                        "max_variable_mods": args.max_variable_mods,
                        "fixed_mods": [],
                        "min_intensity": args.min_intensity,
                        "ms_context": args.ms_context,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        run(["aws", "s3", "cp", str(report), f"{prefix}/{name}.json"])
        compressed.unlink()
        report.unlink()
        print(f"{name}: {rows:,} transitions published", flush=True)

    print(f"child finished {len(mine)} shard(s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
