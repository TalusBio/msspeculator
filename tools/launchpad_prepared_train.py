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

import os
import subprocess
from pathlib import Path


CONFIG = Path("runs/prepared-cloud.toml")


def main() -> None:
    if not CONFIG.exists():
        raise SystemExit("stage is incomplete; run tools/prepare_launchpad_full_run.py first")
    env = os.environ.copy()
    env["PEPDISTILL_S3_PREFIX"] = "s3://terraform-workstations-bucket/jspaezp/pepdistill-prepared/v1"
    env["PEPDISTILL_SOURCE_S3_PREFIX"] = env["PEPDISTILL_S3_PREFIX"]
    subprocess.run(
        ["uv", "run", "--project", ".", "--extra", "teacher", "--extra", "prospect",
         "pepdistill", "run", str(CONFIG), "--device", "cpu"],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
