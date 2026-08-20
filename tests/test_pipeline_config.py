"""Prepared-prefix train configuration contract."""

import pytest

from pepdistill.distill.pipeline import RunConfig
from pepdistill.models import build_student

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
    assert cfg.train.validation_interval_minutes == 60.0
    assert cfg.augmentation.residue_substitution_probability == 0.0


def test_prepared_loader_workers_parse(tmp_path):
    cfg = RunConfig.from_toml(_write(tmp_path, BASE + "\nnum_workers = 2\nmodel_threads = 3\n"))
    assert cfg.train.num_workers == 2
    assert cfg.train.model_threads == 3


def test_enabled_train_requires_prepared_prefix(tmp_path):
    text = BASE.replace('prepared_prefix = "s3://bucket/prepared/v1"\n', "")
    with pytest.raises(ValueError, match="requires prepared_prefix"):
        RunConfig.from_toml(_write(tmp_path, text))


def test_removed_raw_sources_fail_loudly(tmp_path):
    text = (
        BASE.replace('prepared_prefix = "s3://bucket/prepared/v1"', "")
        + """
[[train.sources]]
record = "prospect"
meta = "pool_meta.parquet"
zip = "pool.zip"
shards = "all"
"""
    )
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


def test_dropout_override_parses_and_reaches_attention(tmp_path):
    cfg = RunConfig.from_toml(_write(tmp_path, "dropout = 0.0\n" + BASE))
    assert cfg.dropout == 0.0

    model = build_student("small")
    layer = model.backbone.net.layers[0]
    assert (layer.dropout1.p, layer.self_attn.dropout) != (0.0, 0.0)
    model.set_dropout(cfg.dropout)
    # Attention keeps its rate as a float rather than a child module, so it is the case a
    # modules()-only walk silently misses.
    assert (layer.dropout1.p, layer.self_attn.dropout) == (0.0, 0.0)
    assert model.cfg.dropout == 0.0
    with pytest.raises(ValueError, match="dropout must be in"):
        model.set_dropout(1.0)


def test_remote_output_and_pretrain_checkpoint_interval_parse(tmp_path):
    text = BASE.replace(
        'out = "runs/x"',
        'out = "runs/x"\nremote_output_prefix = "s3://bucket/training/small"',
    ).replace("enabled = false", "enabled = false\ncheckpoint_every_steps = 123", 1)
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.remote_output_prefix == "s3://bucket/training/small"
    assert cfg.pretrain.checkpoint_every_steps == 123


def test_wandb_tracking_config_parses(tmp_path):
    text = (
        BASE
        + """
[tracking]
enabled = true
project = "spectra"
group = "full-v1"
tags = ["base", "non-test"]
mode = "offline"
min_log_interval_seconds = 12.5
max_log_interval_steps = 25
"""
    )
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.tracking.enabled
    assert cfg.tracking.project == "spectra"
    assert cfg.tracking.group == "full-v1"
    assert cfg.tracking.tags == ["base", "non-test"]
    assert cfg.tracking.mode == "offline"
    assert cfg.tracking.min_log_interval_seconds == 12.5
    assert cfg.tracking.max_log_interval_steps == 25


def test_training_diagnostics_config_parses(tmp_path):
    cfg = RunConfig.from_toml(
        _write(
            tmp_path,
            BASE
            + """
[diagnostics]
enabled = true
teacher = "fake"
butterflies = 5
every_n_epochs = 2
interval_minutes = 30
render_initial = false
""",
        )
    )
    assert cfg.diagnostics.enabled
    assert cfg.diagnostics.teacher == "fake"
    assert cfg.diagnostics.butterflies == 5
    assert cfg.diagnostics.every_n_epochs == 2
    assert cfg.diagnostics.interval_minutes == 30
    assert not cfg.diagnostics.render_initial


def test_training_diagnostics_config_rejects_invalid_frequency(tmp_path):
    with pytest.raises(ValueError, match="interval_minutes"):
        RunConfig.from_toml(_write(tmp_path, BASE + "\n[diagnostics]\ninterval_minutes = -1\n"))


def test_augmentation_and_wall_clock_validation_config_parse(tmp_path):
    text = BASE.replace("epochs = 5", "epochs = 5\nvalidation_interval_minutes = 30")
    cfg = RunConfig.from_toml(
        _write(
            tmp_path,
            text
            + """
[augmentation]
residue_substitution_probability = 0.01
""",
        )
    )
    assert cfg.augmentation.residue_substitution_probability == pytest.approx(0.01)
    assert cfg.train.validation_interval_minutes == 30.0


def test_invalid_augmentation_probability_and_validation_interval_fail(tmp_path):
    with pytest.raises(ValueError, match="residue_substitution_probability"):
        RunConfig.from_toml(
            _write(
                tmp_path,
                BASE + "\n[augmentation]\nresidue_substitution_probability = 1.1\n",
            )
        )
    with pytest.raises(ValueError, match="validation_interval_minutes"):
        RunConfig.from_toml(
            _write(
                tmp_path, BASE.replace("epochs = 5", "epochs = 5\nvalidation_interval_minutes = 0")
            )
        )


def test_lr_decay_knobs_parse(tmp_path):
    text = BASE + "\nearly_stop_patience = 10\nlr_decay_patience = 3\nlr_decay_factor = 0.25\n"
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert (cfg.train.lr_decay_patience, cfg.train.lr_decay_factor) == (3, 0.25)


def test_a_decay_that_could_never_fire_is_refused(tmp_path):
    """Caught at config load, not hours into a run: with the decay no more impatient than the
    stop, the run ends at the same plateau that was supposed to trigger the smaller rate."""
    text = BASE + "\nearly_stop_patience = 3\nlr_decay_patience = 3\n"
    with pytest.raises(ValueError, match="must be below early_stop_patience"):
        RunConfig.from_toml(_write(tmp_path, text))


def test_lr_decay_is_off_by_default(tmp_path):
    cfg = RunConfig.from_toml(_write(tmp_path, BASE))
    assert cfg.train.lr_decay_patience == 0
