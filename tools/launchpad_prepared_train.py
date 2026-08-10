"""Launchpad entrypoint for training from the prepared S3 prefix."""

# /// script
# requires-python = ">=3.11"
# dependencies = []
#
# [tool.launchpad]
# vcpus = 31
# memory = 30000
# job_name = "pepdistill-prepared-train"
# ///

from __future__ import annotations

import subprocess
from pathlib import Path


CONFIG = Path("runs/prepared-cloud.toml")


def main() -> None:
    if not CONFIG.exists():
        raise SystemExit("stage is incomplete; run tools/prepare_launchpad_full_run.py first")
    subprocess.run(
        ["uv", "run", "--project", ".", "--extra", "teacher", "--extra", "etl",
         "pepdistill", "run", str(CONFIG), "--device", "cpu"],
        check=True,
    )


if __name__ == "__main__":
    main()
