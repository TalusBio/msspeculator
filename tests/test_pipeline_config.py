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
