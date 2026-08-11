"""Cloud wrapper selection stays deterministic across standalone and array jobs."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "tools" / "launchpad_prepared_train.py"
    spec = importlib.util.spec_from_file_location("launchpad_prepared_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_training_defaults_to_small(monkeypatch):
    monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX", raising=False)
    monkeypatch.delenv("PEPDISTILL_TRAIN_PRESET", raising=False)
    assert _module()._selected_preset() == "small"


def test_array_training_selects_preset_by_index(monkeypatch):
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "3")
    monkeypatch.setenv("PEPDISTILL_TRAIN_PRESETS", "flash,small-2h,small,base-4h,base")
    assert _module()._selected_preset() == "base-4h"


def test_training_environment_passes_wandb_key_without_logging_it(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "short-lived-key")
    env = _module()._training_environment()
    assert env["WANDB_API_KEY"] == "short-lived-key"


def test_training_environment_requires_wandb_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="short-lived service-account key"):
        _module()._training_environment()
