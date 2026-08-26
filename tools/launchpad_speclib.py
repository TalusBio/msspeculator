"""Launchpad entrypoint for spectral-library generation.

Drives a **staged** `pepdistill-cli` binary; it does not build one. The launchpad container
pre-installs nothing; no Rust toolchain, no `aws`, no `curl`; so the binary is cross-compiled
once locally (`cross build --release --target x86_64-unknown-linux-musl`) and staged alongside the
FASTA. musl means a static binary that does not care what the image's glibc is, and the binary
carries its own weights, so only the FASTA has to be staged with it.

One library, one object, mzSpecLib by default so the published object carries its own provenance.
`--max-fragments 15` and gzip output put a whole human tryptic CysPAT/Ox/Phospho library at ~10 GB,
so it fits the worker volume and there is nothing to shard: the CLI compresses in its writer
thread, so the uncompressed form never exists on disk.

    launchpad run tools/launchpad_speclib.py --stage "$STAGE" \
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


def s3():
    global _S3
    if _S3 is None:
        import boto3

        _S3 = boto3.client("s3")
    return _S3


def split_uri(uri: str) -> tuple[str, str]:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return bucket, key


def s3_upload(path: Path, uri: str) -> None:
    bucket, key = split_uri(uri)
    print(f"+ upload {path} -> {uri}", flush=True)
    s3().upload_file(str(path), bucket, key)


def staged(source: str, target: Path) -> Path:
    """Resolve `source` to a local file: a staged path, or an S3 object to download.

    Absolute on return: `Path("./pepdistill-cli")` stringifies to a bare name, and subprocess
    resolves a name with no directory component through PATH rather than the working directory.
    """
    if not source.startswith("s3://"):
        path = Path(source)
        if not path.is_file():
            raise SystemExit(f"not a staged file or an s3:// URI: {source}")
        return path.resolve()
    if not (target.exists() and target.stat().st_size > 0):
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        bucket, key = split_uri(source)
        s3().download_file(bucket, key, str(partial))
        if partial.stat().st_size == 0:
            partial.unlink()
            raise SystemExit(f"{source} resolved to an empty file")
        os.replace(partial, target)
    return target.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    env = os.environ.get
    parser.add_argument("--binary", default=env("PEPDISTILL_SPECLIB_BINARY", "./pepdistill-cli"))
    # The binary carries its weights, so there is nothing to stage unless a different model is
    # wanted. A `builtin:` spec is passed through untouched; anything else is staged or downloaded.
    parser.add_argument("--model", default=env("PEPDISTILL_SPECLIB_MODEL", "builtin:small-v0"))
    parser.add_argument("--fasta", default=env("PEPDISTILL_SPECLIB_FASTA", "./proteome.fasta"))
    parser.add_argument(
        "--out-prefix",
        default=env("PEPDISTILL_SPECLIB_OUT"),
        help="s3:// prefix the library and its config are written under",
    )
    parser.add_argument("--name", default=env("PEPDISTILL_SPECLIB_NAME", "library"))
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
    parser.add_argument(
        "--diann-tsv",
        action="store_true",
        help="publish DIA-NN TSV instead of mzSpecLib, for a consumer that reads only the TSV",
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
    binary = staged(args.binary, work / "pepdistill-cli")
    binary.chmod(0o755)
    model = (
        args.model
        if args.model.startswith("builtin:")
        else str(staged(args.model, work / "model.safetensors"))
    )
    fasta = staged(args.fasta, work / "proteome.fasta")

    # mzSpecLib, gzipped: this script publishes libraries other people consume, and mzSpecLib is
    # the format that carries its own provenance. `--diann-tsv` for a consumer that needs the TSV.
    local = work / (f"{args.name}.tsv.gz" if args.diann_tsv else f"{args.name}.mzspeclib.txt.gz")
    command = [
        str(binary),
        "library",
        "--model",
        str(model),
        "--fasta",
        str(fasta),
        "--out",
        str(local),
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
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)

    prefix = args.out_prefix.rstrip("/")
    s3_upload(local, f"{prefix}/{local.name}")
    # The CLI writes `<out>.config.json` beside the library: resolved settings plus blake2b
    # digests of the model and FASTA, which is what makes a published library reproducible. An
    # mzSpecLib library carries the same record inside itself; the sidecar stays for the TSV and
    # for anyone reading the prefix without opening a multi-gigabyte file.
    s3_upload(Path(f"{local}.config.json"), f"{prefix}/{local.name}.config.json")
    print(f"published {local.stat().st_size / 1e9:.2f} GB to {prefix}/", flush=True)


if __name__ == "__main__":
    sys.exit(main())
