"""Launchpad entrypoint for range-addressed prepared-asset generation.

The preparation catalog is global and deterministic.  Launchpad can run this script once
for the full catalog or assign disjoint ``--range START:STOP`` intervals to workers; neither
choice changes the resulting assets.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
#
# [tool.launchpad]
# vcpus = 8
# memory = 30000
# job_name = "pepdistill-etl"
# ///

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("runs/prepare-full.toml"),
        help="prepared ETL config staged with the job",
    )
    parser.add_argument(
        "--range",
        dest="range_spec",
        help="optional half-open global shard range START:STOP",
    )
    parser.add_argument("--force", action="store_true", help="rebuild completed shard assets")
    parser.add_argument("--dry-run", action="store_true", help="discover and print selected shards")
    args = parser.parse_args()
    if not Path("pyproject.toml").exists():
        raise SystemExit("stage is incomplete; run tools/prepare_launchpad_full_run.py first")
    # `etl` only. Torch is not a core dependency and the default `dev` group does not reference it,
    # so this resolves without the ~3 GB of CUDA wheels that a worker never uses; downloading
    # those is what timed out and killed 12 of 40 children of one array job. Deliberately no
    # group-exclusion flag: the cloud image's uv predates `--no-default-groups` and rejects it.
    command = [
        "uv",
        "run",
        "--project",
        ".",
        "--extra",
        "etl",
        "pepdistill",
        "prepare",
        str(args.config),
    ]
    range_spec = args.range_spec or os.environ.get("PEPDISTILL_PREPARE_RANGE")
    if range_spec:
        command.extend(["--range", range_spec])
    else:
        array_index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
        array_size = os.environ.get("PEPDISTILL_PREPARE_ARRAY_SIZE")
        if array_index is not None and array_size is not None:
            index = int(array_index)
            workers = int(array_size)
            if not 0 <= index < workers:
                raise SystemExit(f"array index {index} outside 0..{workers - 1}")
            command.extend(["--partition", f"{index}/{workers}"])
            print(f"array child {index}/{workers}: selecting a byte-balanced catalog range")
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
