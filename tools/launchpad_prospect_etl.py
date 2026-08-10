"""Launchpad entrypoint for the first Polars preparation pilot."""

# /// script
# requires-python = ">=3.11"
# dependencies = []
#
# [tool.launchpad]
# vcpus = 8
# memory = 30000
# job_name = "pepdistill-etl-isoform-1"
# ///

from __future__ import annotations

import subprocess
from pathlib import Path


SOURCE_PREFIX = "s3://terraform-workstations-bucket/jspaezp/20241022_prospect"
OUT_PREFIX = "s3://terraform-workstations-bucket/jspaezp/pepdistill-prepared/v1"


def main() -> None:
    command = [
        "uv", "run", "--project", ".", "--extra", "etl", "-m", "pepdistill.etl.prospect",
        "--source-prefix", SOURCE_PREFIX,
        "--meta", "TUM_isoform_meta_data.parquet",
        "--archive-stem", "TUM_isoform_1",
        "--out-prefix", OUT_PREFIX,
        "--dataset", "prospect_tum_isoform",
        "--instrument", "Lumos",
    ]
    if not Path("pyproject.toml").exists():
        raise SystemExit("stage is incomplete; run tools/prepare_launchpad_full_run.py first")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
