"""Multi-source train config: parsing, validation, dataset_index assignment."""

import pytest

from pepdistill.distill.pipeline import (
    RunConfig,
    TrainSource,
    resolve_dataset_index,
)

BASE = """
out = "runs/x"
preset = "small"

[pretrain]
enabled = false

[train]
enabled = true
epochs = 5
"""

TWO_SOURCES = BASE + """
[[train.sources]]
record = "prospect"
meta = "TUM_third_pool_meta_data.parquet"
zip = "TUM_third_pool.zip"
shards = [0]
dataset = "third_pool"
instrument = "Lumos"

[[train.sources]]
record = "test_ptm"
meta = "Kmod_Butyryl_meta_data.parquet"
zip = "Kmod_Butyryl.zip"
shards = [0]
dataset = "Kmod_Butyryl"
val_only = true
"""


def _write(tmp_path, text):
    p = tmp_path / "run.toml"
    p.write_text(text)
    return p


def test_sources_parse_into_train_source_objects(tmp_path):
    cfg = RunConfig.from_toml(_write(tmp_path, TWO_SOURCES))
    assert [s.record for s in cfg.train.sources] == ["prospect", "test_ptm"]
    assert cfg.train.sources[1].val_only is True
    assert cfg.train.sources[0].instrument == "Lumos"


def test_activation_override_parses(tmp_path):
    text = BASE.replace("enabled = true", "enabled = false").replace(
        '\n[pretrain]', '\nactivation = "leaky_relu"\n\n[pretrain]'
    )
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.activation == "leaky_relu"


def test_model_in_parses_as_training_initializer(tmp_path):
    text = BASE.replace('out = "runs/x"', 'model_in = "pretrain.ckpt"\nout = "runs/x"')
    text = text.replace("[train]\nenabled = true", "[train]\nenabled = false")
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.model_in == "pretrain.ckpt"


def test_cache_s3_prefix_parses(tmp_path):
    text = BASE.replace('out = "runs/x"', 'cache_s3_prefix = "s3://bucket/cache"\nout = "runs/x"')
    text = text.replace("[train]\nenabled = true", "[train]\nenabled = false")
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.cache_s3_prefix == "s3://bucket/cache"


def test_flat_pool_keys_raise(tmp_path):
    text = BASE + '\nrecord = "prospect"\nzip = "TUM_third_pool.zip"\n'
    with pytest.raises(ValueError, match=r"\[\[train.sources\]\]"):
        RunConfig.from_toml(_write(tmp_path, text))


def test_missing_dataset_name_with_two_sources_raises(tmp_path):
    text = TWO_SOURCES.replace('dataset = "Kmod_Butyryl"\n', "")
    with pytest.raises(ValueError, match="dataset"):
        RunConfig.from_toml(_write(tmp_path, text))


def test_single_source_may_omit_dataset(tmp_path):
    text = BASE + """
[[train.sources]]
record = "prospect"
meta = "TUM_third_pool_meta_data.parquet"
zip = "TUM_third_pool.zip"
shards = [0]
"""
    cfg = RunConfig.from_toml(_write(tmp_path, text))
    assert cfg.train.sources[0].dataset is None


def _src(name, val_only=False):
    return TrainSource(record="r", meta="m", zip="z", shards=[0],
                       dataset=name, val_only=val_only)


def test_dataset_index_follows_declaration_order_from_row_one():
    idx = resolve_dataset_index([_src("alpha"), _src("beta")])
    assert idx == {"alpha": 1, "beta": 2}


def test_sources_sharing_a_name_share_a_row():
    idx = resolve_dataset_index([_src("alpha"), _src("beta"), _src("alpha")])
    assert idx == {"alpha": 1, "beta": 2}


def test_existing_names_keep_their_rows_and_new_ones_append():
    idx = resolve_dataset_index([_src("beta"), _src("gamma")], existing={"alpha": 1, "beta": 2})
    assert idx == {"alpha": 1, "beta": 2, "gamma": 3}


def test_row_zero_is_never_assigned():
    idx = resolve_dataset_index([_src("alpha")])
    assert 0 not in idx.values()


def test_existing_index_claiming_the_neutral_row_raises():
    with pytest.raises(ValueError, match="row 0 is reserved"):
        resolve_dataset_index([_src("beta")], existing={"alpha": 0})


def test_existing_index_with_duplicate_rows_raises():
    with pytest.raises(ValueError, match="duplicate"):
        resolve_dataset_index([_src("gamma")], existing={"alpha": 1, "beta": 1})


def test_enabled_train_with_no_sources_raises_at_parse_time(tmp_path):
    """Names the actual cause. This used to reach _build_train_stage and be reported as
    'every [[train.sources]] is val_only', which is a different (and wrong) diagnosis."""
    with pytest.raises(ValueError, match="declares no"):
        RunConfig.from_toml(_write(tmp_path, BASE))


def test_all_sources_val_only_raises_at_parse_time(tmp_path):
    """Decidable from the config alone, so it must not wait until every shard has been
    downloaded and extracted — on a real pool that is multiple GB spent on a config error."""
    text = TWO_SOURCES.replace('dataset = "third_pool"', 'dataset = "third_pool"\nval_only = true')
    with pytest.raises(ValueError, match="every \\[\\[train.sources\\]\\] is val_only"):
        RunConfig.from_toml(_write(tmp_path, text))


def test_disabled_train_may_declare_no_sources(tmp_path):
    """The checks are about a stage that will actually run; [train] enabled = false is the
    ordinary pretrain-only config and must stay parseable."""
    cfg = RunConfig.from_toml(_write(tmp_path, BASE.replace("enabled = true", "enabled = false")))
    assert cfg.train.sources == []
