"""Launchpad entrypoint for sharded DIA-NN spectral-library generation.

Drives a **staged** `pepdistill-cli` binary; it does not build one. The launchpad container
pre-installs nothing, so a Rust toolchain cannot be assumed, and cross-compiling once locally
(`cross build --release --target x86_64-unknown-linux-musl`) is faster and reproducible besides.
musl means a static binary that does not care what the image's glibc is.

Work is measured in *logical shards* rather than jobs. Each shard is generated straight to
`.tsv.gz`, uploaded, and deleted before the next begins, so peak disk is one shard however large
the library is. With `--max-fragments 15` a whole human tryptic library at two variable
placements is ~13 GB gzipped, so sharding is now about restartability more than about disk.

Shards are disjoint by peptide (`library --partition`), so the library is the concatenation of its
shards with the repeated header rows dropped. Sharding the FASTA instead would emit a peptide
shared by two proteins in both of their shards.

A shard whose object already exists is skipped, which makes the job restartable: jobs here vanish
from `launchpad list` rather than reaching a terminal state, so completion is judged from the
artifacts, and a partial run resumes by relaunching the same command.

    launchpad run tools/launchpad_speclib.py --stage speclib-stage --vcpus 16 --memory 32000 \
      --env PEPDISTILL_SPECLIB_OUT=s3://bucket/pepdistill-speclib/human-cyspat-v1
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
#
# [tool.launchpad]
# vcpus = 16
# memory = 32000
# job_name = "pepdistill-speclib"
# ///

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

#: Lazily constructed so `--help` works without credentials.
_S3 = None


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def fetch(source: str, target: Path) -> Path:
    """Resolve `source` to a local file: a staged path, or an S3 object to download."""
    if not source.startswith("s3://"):
        path = Path(source)
        if not path.is_file():
            raise SystemExit(f"not a staged file or an s3:// URI: {source}")
        return path
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    bucket, key = split_uri(source)
    s3().download_file(bucket, key, str(partial))
    if partial.stat().st_size == 0:
        partial.unlink()
        raise SystemExit(f"{source} resolved to an empty file")
    os.replace(partial, target)
    return target


def s3() -> "boto3.client":  # noqa: F821 - resolved by the PEP 723 dependency
    global _S3
    if _S3 is None:
        import boto3

        _S3 = boto3.client("s3")
    return _S3


def split_uri(uri: str) -> tuple[str, str]:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return bucket, key


def s3_exists(uri: str) -> bool:
    """Whether the object is already published.

    boto3 rather than the `aws` CLI: the launchpad container pre-installs nothing, so the CLI is
    absent, while a PEP 723 dependency is installed by uv before the script runs.
    """
    from botocore.exceptions import ClientError

    bucket, key = split_uri(uri)
    try:
        s3().head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    return True


def s3_upload(path: Path, uri: str) -> None:
    bucket, key = split_uri(uri)
    print(f"+ upload {path} -> {uri}", flush=True)
    s3().upload_file(str(path), bucket, key)


def shard_indices(total: int) -> list[int]:
    """The logical shards this invocation owns.

    A single job owns all of them. `AWS_BATCH_JOB_ARRAY_INDEX` splits them across an array when
    one is used, strided so uneven shard cost spreads instead of landing on one child, and
    `PEPDISTILL_SPECLIB_SHARD` narrows to a single shard for a smoke test or a manual retry.
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
        "--binary",
        default=env("PEPDISTILL_SPECLIB_BINARY", "./pepdistill-cli"),
        help="staged linux pepdistill-cli",
    )
    parser.add_argument(
        "--model",
        default=env("PEPDISTILL_SPECLIB_MODEL", "./model.safetensors"),
        help="exported .safetensors artifact (staged path or s3://)",
    )
    parser.add_argument(
        "--out-prefix",
        default=env("PEPDISTILL_SPECLIB_OUT"),
        help="s3:// prefix the shards are written under",
    )
    # A staged FASTA, not a release stream: the same bytes on a re-run, and no chance of a
    # truncated download whose digest still looks valid in config.json.
    parser.add_argument(
        "--fasta",
        default=env("PEPDISTILL_SPECLIB_FASTA", "./proteome.fasta"),
        help="staged FASTA path, or an s3:// object",
    )
    parser.add_argument("--shards", type=int, default=int(env("PEPDISTILL_SPECLIB_SHARDS", "200")))
    parser.add_argument("--missed-cleavages", type=int, default=2)
    parser.add_argument("--min-length", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--max-charge", type=int, default=4)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument(
        "--max-fragments",
        type=int,
        default=15,
        help="strongest N transitions per precursor; 15 is the DIA convention",
    )
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

    if not args.out_prefix:
        raise SystemExit("--out-prefix is required (or set PEPDISTILL_SPECLIB_OUT)")
    variable_mods = args.variable_mod or [
        "C[UNIMOD:2057]",  # 6C-CysPAT
        "M[UNIMOD:35]",  # Oxidation
        "STY[UNIMOD:21]",  # Phospho
    ]

    work = Path("speclib-work")
    work.mkdir(exist_ok=True)
    # Absolute: `Path("./pepdistill-cli")` stringifies to a bare name, and subprocess resolves a
    # name with no directory component through PATH rather than the working directory.
    binary = fetch(args.binary, work / "pepdistill-cli").resolve()
    binary.chmod(0o755)
    model = fetch(args.model, work / "model.safetensors").resolve()
    fasta = fetch(args.fasta, work / "proteome.fasta").resolve()
    prefix = args.out_prefix.rstrip("/")
    mine = shard_indices(args.shards)
    print(f"{len(mine)} of {args.shards} shards to generate", flush=True)

    for shard in mine:
        name = f"shard-{shard:05d}-of-{args.shards:05d}"
        destination = f"{prefix}/{name}.tsv.gz"
        if s3_exists(destination):
            print(f"{name}: already published, skipping", flush=True)
            continue
        # Straight to .gz: the CLI compresses in its writer thread, so the uncompressed library
        # never exists on disk. It also writes `<out>.config.json` -- the resolved settings plus
        # input digests -- which is what makes a published shard reproducible.
        local = work / f"{name}.tsv.gz"
        config = Path(f"{local}.config.json")
        run(
            [
                str(binary),
                "library",
                "--model",
                str(model),
                "--fasta",
                str(fasta),
                "--out",
                str(local),
                "--partition",
                f"{shard}/{args.shards}",
                "--no-fixed-mods",
                "--max-variable-mods",
                str(args.max_variable_mods),
                "--max-fragments",
                str(args.max_fragments),
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
            ]
        )
        s3_upload(local, destination)
        s3_upload(config, f"{destination}.config.json")
        size = local.stat().st_size
        local.unlink()
        config.unlink()
        print(f"{name}: published {size / 1e6:.1f} MB", flush=True)

    print(f"finished {len(mine)} shard(s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
