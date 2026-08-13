"""Launchpad entrypoint that finalizes a completed prepared-asset catalog."""

# /// script
# requires-python = ">=3.11"
# dependencies = []
#
# [tool.launchpad]
# vcpus = 8
# memory = 16000
# job_name = "pepdistill-etl-finalize"
# ///

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("runs/prepare-full.toml"))
    args = parser.parse_args()
    if not Path("pyproject.toml").exists():
        raise SystemExit("stage is incomplete; run tools/prepare_launchpad_full_run.py first")
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            ".",
            # See launchpad_prospect_etl: `etl` only, and no group-exclusion flag.
            "--extra",
            "etl",
            "pepdistill",
            "prepare-finalize",
            str(args.config),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
