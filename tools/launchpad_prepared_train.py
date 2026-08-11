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


DEFAULT_PRESETS = ("flash", "small-2h", "small", "base-4h", "base")
WANDB_KEY_FILE = Path("WANDB_SECRET")


def _training_environment() -> dict[str, str]:
    """Load the staged short-lived W&B key, then remove both staged copies."""
    env = os.environ.copy()
    if not WANDB_KEY_FILE.is_file():
        raise SystemExit(
            f"tracking is enabled; stage the short-lived key as {WANDB_KEY_FILE}"
        )
    key = WANDB_KEY_FILE.read_text().strip()
    if not key:
        raise SystemExit(f"staged W&B key {WANDB_KEY_FILE} is empty")
    env["WANDB_API_KEY"] = key
    WANDB_KEY_FILE.unlink()

    stage_uri = env.get("LP_STAGE_URI", "").rstrip("/")
    if stage_uri:
        subprocess.run(
            ["aws", "s3", "rm", f"{stage_uri}/{WANDB_KEY_FILE.name}"],
            check=True,
        )
    return env


def _selected_preset() -> str:
    explicit = os.environ.get("PEPDISTILL_TRAIN_PRESET")
    if explicit:
        return explicit
    presets = tuple(
        value.strip()
        for value in os.environ.get("PEPDISTILL_TRAIN_PRESETS", ",".join(DEFAULT_PRESETS)).split(",")
        if value.strip()
    )
    index_raw = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if index_raw is None:
        return "small"
    index = int(index_raw)
    if not 0 <= index < len(presets):
        raise SystemExit(f"array index {index} outside {len(presets)} configured preset(s)")
    return presets[index]


def main() -> None:
    preset = _selected_preset()
    config = Path(f"runs/prepared-cloud-{preset}.toml")
    if not config.exists():
        raise SystemExit("stage is incomplete; run tools/prepare_launchpad_full_run.py first")
    command = [
        "uv", "run", "--project", ".", "--extra", "teacher", "--extra", "etl",
        "--extra", "tracking",
        "pepdistill", "run", str(config), "--device", "cpu",
    ]
    if checkpoint := os.environ.get("PEPDISTILL_MODEL_IN"):
        command.extend(["--model-in", checkpoint])
    if os.environ.get("PEPDISTILL_NO_PRETRAIN", "").lower() in {"1", "true", "yes"}:
        command.append("--no-pretrain")
    print(f"[train-launch] preset={preset} config={config}", flush=True)
    subprocess.run(command, check=True, env=_training_environment())


if __name__ == "__main__":
    main()
