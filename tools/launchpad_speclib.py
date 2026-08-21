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
# dependencies = []
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

#: Reviewed canonical human proteome. The `stream` endpoint is slow and resets on large queries,
#: so this is retried rather than assumed to succeed on the first call.
UNIPROT_STREAM = (
    "https://rest.uniprot.org/uniprotkb/stream?format=fasta"
    "&query=%28proteome%3A{accession}%29+AND+%28reviewed%3Atrue%29"
)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def fetch(source: str, target: Path) -> Path:
    """Resolve `source` to a local file: an S3 URI, a `uniprot:UP...` proteome, or a path."""
    if target.exists() and target.stat().st_size > 0:
        return target
    if not source.startswith(("s3://", "uniprot:")):
        path = Path(source)
        if not path.is_file():
            raise SystemExit(f"not a file, an s3:// URI, or a uniprot: reference: {source}")
        return path
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if source.startswith("s3://"):
        run(["aws", "s3", "cp", source, str(partial)])
    else:
        # curl rather than urllib: this transfer resets often enough that the retry matters, and
        # hand-rolling --retry-all-errors would be the same logic with more places to be wrong.
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
                UNIPROT_STREAM.format(accession=source.removeprefix("uniprot:")),
            ]
        )
    if partial.stat().st_size == 0:
        partial.unlink()
        raise SystemExit(f"{source} resolved to an empty file")
    os.replace(partial, target)
    return target


def s3_exists(uri: str) -> bool:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return (
        subprocess.run(
            ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
            capture_output=True,
        ).returncode
        == 0
    )


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
    parser.add_argument("--fasta", default=env("PEPDISTILL_SPECLIB_FASTA", "uniprot:UP000005640"))
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
    binary = fetch(args.binary, work / "pepdistill-cli")
    binary.chmod(0o755)
    model = fetch(args.model, work / "model.safetensors")
    fasta = fetch(args.fasta, work / "proteome.fasta")
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
        run(["aws", "s3", "cp", str(local), destination])
        run(["aws", "s3", "cp", str(config), f"{destination}.config.json"])
        size = local.stat().st_size
        local.unlink()
        config.unlink()
        print(f"{name}: published {size / 1e6:.1f} MB", flush=True)

    print(f"finished {len(mine)} shard(s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
