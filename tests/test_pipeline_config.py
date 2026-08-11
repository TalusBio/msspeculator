"""Prepared-prefix train configuration contract."""

import pytest

from pepdistill.distill.pipeline import RunConfig

BASE = """
out = "runs/x"
preset = "small"

[pretrain]
enabled = false

[train]
enabled = true
epochs = 5
prepared_prefix = "s3://bucket/prepared/v1"
"""


def _write(tmp_path, text):
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def test_prepared_prefix_parses(tmp_path):
    cfg = RunConfig.from_toml(_write(tmp_path, BASE))
    assert cfg.train.prepared_prefix == "s3://bucket/prepared/v1"
    assert cfg.train.num_workers == 0
    assert cfg.train.model_threads == 4


def test_prepared_loader_workers_parse(tmp_path):
    cfg = RunConfig.from_toml(
        _write(tmp_path, BASE + "\nnum_workers = 2\nmodel_threads = 3\n")
    )
    assert cfg.train.num_workers == 2
    assert cfg.train.model_threads == 3


def test_enabled_train_requires_prepared_prefix(tmp_path):
    text = BASE.replace('prepared_prefix = "s3://bucket/prepared/v1"\n', "")
    with pytest.raises(ValueError, match="requires prepared_prefix"):
        RunConfig.from_toml(_write(tmp_path, text))


def test_removed_raw_sources_fail_loudly(tmp_path):
    text = BASE.replace('prepared_prefix = "s3://bucket/prepared/v1"', "") + """
[[train.sources]]
record = "prospect"
meta = "pool_meta.parquet"
zip = "pool.zip"
shards = "all"
"""
    with pytest.raises(ValueError, match=r"\[train.sources\] was removed"):
        RunConfig.from_toml(_write(tmp_path, text))


def test_disabled_train_may_omit_prepared_prefix(tmp_path):
    text = BASE.replace("enabled = true", "enabled = false").replace(
        'prepared_prefix = "s3://bucket/prepared/v1"\n', ""
    )
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert not cfg.train.enabled


def test_activation_override_parses(tmp_path):
    text = BASE.replace('activation = "', 'activation = "', 0)
    text = 'activation = "leaky_relu"\n' + text
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.activation == "leaky_relu"


def test_remote_output_and_pretrain_checkpoint_interval_parse(tmp_path):
    text = BASE.replace(
        'out = "runs/x"',
        'out = "runs/x"\nremote_output_prefix = "s3://bucket/training/small"',
    ).replace("enabled = false", "enabled = false\ncheckpoint_every_steps = 123", 1)
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.remote_output_prefix == "s3://bucket/training/small"
    assert cfg.pretrain.checkpoint_every_steps == 123


def test_wandb_tracking_config_parses(tmp_path):
    text = BASE + '''
[tracking]
enabled = true
project = "spectra"
group = "full-v1"
tags = ["base", "non-test"]
mode = "offline"
min_log_interval_seconds = 12.5
'''
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.tracking.enabled
    assert cfg.tracking.project == "spectra"
    assert cfg.tracking.group == "full-v1"
    assert cfg.tracking.tags == ["base", "non-test"]
    assert cfg.tracking.mode == "offline"
    assert cfg.tracking.min_log_interval_seconds == 12.5
