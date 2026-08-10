"""Launchpad entrypoint for training from the prepared S3 prefix."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.34"]
#
# [tool.launchpad]
# vcpus = 31
# memory = 30000
# job_name = "pepdistill-prepared-train"
# ///

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


DEFAULT_PRESETS = ("flash", "small-2h", "small", "base-4h", "base")


def _training_environment() -> dict[str, str]:
    """Resolve the W&B key from Secrets Manager without exposing it in the job definition."""
    env = os.environ.copy()
    if env.get("WANDB_API_KEY"):
        return env
    secret_id = env.get("PEPDISTILL_WANDB_SECRET_ID")
    if not secret_id:
        raise SystemExit(
            "tracking is enabled; set PEPDISTILL_WANDB_SECRET_ID to an AWS Secrets Manager "
            "secret name/ARN (do not pass WANDB_API_KEY through launchpad --env)"
        )
    import boto3

    response = boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)
    value = response.get("SecretString")
    if not value:
        raise RuntimeError(f"Secrets Manager secret {secret_id!r} has no SecretString")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        key = value
    else:
        key = (
            decoded.get("WANDB_API_KEY") or decoded.get("api_key")
            if isinstance(decoded, dict)
            else None
        )
    if not key or not isinstance(key, str):
        raise RuntimeError(
            f"Secrets Manager secret {secret_id!r} must be a raw key or JSON with WANDB_API_KEY"
        )
    env["WANDB_API_KEY"] = key.strip()
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
