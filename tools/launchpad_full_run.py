"""Launchpad entrypoint for the full non-test PROSPECT run.

The project and config are supplied with ``--stage .launchpad/full-run-stage``.  The large
PROSPECT mirror is *not* staged: the project cache resolves its pre-extracted parquet shards
directly from S3 and only falls back to Zenodo for a genuinely missing object.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
#
# [tool.launchpad]
# vcpus = 31
# memory = 30000
# job_name = "pepdistill-full-nontest"
# ///

from __future__ import annotations

import os
import subprocess
from pathlib import Path


S3_PREFIX = "s3://terraform-workstations-bucket/jspaezp/20241022_prospect"
CONFIG = Path("runs/full-nontest-cloud.toml")


def main() -> None:
    required = [Path("pyproject.toml"), Path("uv.lock"), Path("src"), Path("rust"), CONFIG]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "cloud stage is incomplete; run tools/prepare_launchpad_full_run.py first. "
            f"Missing: {', '.join(missing)}"
        )

    env = os.environ.copy()
    # The first variable is the canonical write-through cache.  The second tells the cache
    # that this prefix also has the pre-existing raw PROSPECT layout at its root.
    env["PEPDISTILL_S3_PREFIX"] = S3_PREFIX
    env["PEPDISTILL_SOURCE_S3_PREFIX"] = S3_PREFIX
    env.setdefault("PEPDISTILL_CACHE", "/tmp/pepdistill-cache")

    command = [
        "uv",
        "run",
        "--project",
        ".",
        "--extra",
        "teacher",
        "--extra",
        "prospect",
        "pepdistill",
        "run",
        str(CONFIG),
        "--device",
        "cpu",
    ]
    print("[launchpad] running full non-test training against the S3 mirror", flush=True)
    print(f"[launchpad] data={S3_PREFIX}", flush=True)
    # Launchpad uploads files from the working directory even when the command fails.  Keep
    # the complete inner traceback there as well as streaming it to CloudWatch; the outer
    # ``CalledProcessError`` otherwise leaves only an unhelpful wrapper traceback in the
    # result tail.
    log_path = Path("cloud-output/launchpad-inner.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


if __name__ == "__main__":
    main()
