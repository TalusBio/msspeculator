"""Streaming reader: row-group boundaries, split filtering, determinism, val dedup."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pepdistill.chem import Peptide
from pepdistill.data.meta_index import MetaIndex, SpectrumMeta
from pepdistill.distill.real_stream import (
    ShardSpec,
    StreamingRealDataset,
    collect_val_examples,
)
from pepdistill.models.context import MSContextEncoder


def _index(n_spectra, split="train", andromeda=None):
    """Distinct charge per scan by default, so each spectrum is its own dedup key."""
    idx = MetaIndex()
    for scan in range(n_spectra):
        idx.by_key[("RUN_A", scan)] = SpectrumMeta(
            Peptide("PEPTIDEK", ()), 2 + scan, 50.0 + scan, 5.0 + scan, split,
            "FTMS", "HCD", 28.0,
            100.0 if andromeda is None else andromeda[scan],
        )
    return idx


def _write_shard(path, n_spectra, row_group_size, intensity=1.0):
    """One shard, 4 usable b-ion fragments per spectrum, laid out so spectra straddle groups."""
    rows = []
    for scan in range(n_spectra):
        for ordinal in (1, 2, 3, 4):
            rows.append(("b", ordinal, 1, intensity, "", scan, "RUN_A"))
    df = pd.DataFrame(
        rows,
        columns=["ion_type", "no", "charge", "intensity", "neutral_loss",
                 "scan_number", "raw_file"],
    )
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                   row_group_size=row_group_size)
    return path


def _shard(tmp_path, n_spectra=10, row_group_size=3):
    p = str(tmp_path / "shard.parquet")
    _write_shard(p, n_spectra, row_group_size)
    return ShardSpec(path=p, raw_files=("RUN_A",), dataset_id=1, instrument="Lumos")


def test_spectrum_straddling_row_groups_is_reconstructed_whole(tmp_path):
    # 4 fragment rows per spectrum, row groups of 3 -> every spectrum crosses a boundary.
    shard = _shard(tmp_path, n_spectra=10, row_group_size=3)
    ds = StreamingRealDataset([shard], _index(10), MSContextEncoder(context_dim=8),
                              frozenset({"train"}), shuffle_buffer=0)
    examples = list(ds.iter_examples(epoch=0))
    assert len(examples) == 10
    for e in examples:
        # b1..b4 all present -> exactly 4 non-zero cells in column 0.
        assert int((e.label.ms2[:, 0] > 0).sum()) == 4


def test_val_only_split_filter_yields_nothing_for_a_train_stream(tmp_path):
    shard = _shard(tmp_path, n_spectra=6)
    ds = StreamingRealDataset([shard], _index(6, split="val"),
                              MSContextEncoder(context_dim=8), frozenset({"train"}),
                              shuffle_buffer=0)
    assert list(ds.iter_examples(epoch=0)) == []


def test_split_filter_selects_only_matching_spectra(tmp_path):
    idx = _index(6, split="train")
    for scan in (0, 1):
        sm = idx.by_key[("RUN_A", scan)]
        idx.by_key[("RUN_A", scan)] = SpectrumMeta(
            sm.peptide, sm.charge, sm.irt, sm.raw_rt, "val",
            sm.mass_analyzer, sm.fragmentation, sm.energy, sm.andromeda,
        )
    shard = _shard(tmp_path, n_spectra=6)
    train = StreamingRealDataset([shard], idx, MSContextEncoder(context_dim=8),
                                 frozenset({"train"}), shuffle_buffer=0)
    assert len(list(train.iter_examples(epoch=0))) == 4


def test_epochs_are_deterministic_and_shuffle_differs_between_them(tmp_path):
    shard = _shard(tmp_path, n_spectra=40, row_group_size=7)
    kw = dict(index=_index(40), encoder=MSContextEncoder(context_dim=8),
              splits=frozenset({"train"}), shuffle_buffer=16, seed=3)
    a = [e.raw_rt for e in StreamingRealDataset([shard], **kw).iter_examples(epoch=0)]
    b = [e.raw_rt for e in StreamingRealDataset([shard], **kw).iter_examples(epoch=0)]
    c = [e.raw_rt for e in StreamingRealDataset([shard], **kw).iter_examples(epoch=1)]
    assert a == b                      # same epoch, same order
    assert sorted(a) == sorted(c)      # same content
    assert a != c                      # different epoch, different order


def test_batches_yields_real_batches_of_the_requested_size(tmp_path):
    shard = _shard(tmp_path, n_spectra=10)
    ds = StreamingRealDataset([shard], _index(10), MSContextEncoder(context_dim=8),
                              frozenset({"train"}), shuffle_buffer=0)
    gen = torch.Generator().manual_seed(0)
    batches = list(ds.batches(batch_size=4, shuffle=False, generator=gen))
    assert [b.raw_rt.shape[0] for b in batches] == [4, 4, 2]
    assert batches[0].dataset_id.tolist() == [1, 1, 1, 1]


def test_shard_with_zero_usable_examples_raises(tmp_path):
    p = str(tmp_path / "empty.parquet")
    # Only y-ions with a neutral loss: the fragment filter removes every row.
    df = pd.DataFrame({
        "ion_type": ["y"], "no": [1], "charge": [1], "intensity": [1.0],
        "neutral_loss": ["H2O"], "scan_number": [0], "raw_file": ["RUN_A"],
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    shard = ShardSpec(path=p, raw_files=("RUN_A",), dataset_id=1, instrument="Lumos")
    ds = StreamingRealDataset([shard], _index(1), MSContextEncoder(context_dim=8),
                              frozenset({"train"}), shuffle_buffer=0)
    with pytest.raises(ValueError, match="zero usable examples"):
        list(ds.iter_examples(epoch=0))


def test_collect_val_keeps_only_the_highest_andromeda_per_key(tmp_path):
    """Three observations of ONE key in one shard; only the best-scoring scan is decoded."""
    p = str(tmp_path / "val.parquet")
    rows = []
    for scan, inten in ((0, 0.1), (1, 0.5), (2, 0.9)):
        for ordinal in (1, 2, 3, 4):
            rows.append(("b", ordinal, 1, inten, "", scan, "RUN_A"))
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(rows, columns=["ion_type", "no", "charge", "intensity",
                                        "neutral_loss", "scan_number", "raw_file"]),
            preserve_index=False,
        ),
        p,
    )
    # Same peptide AND charge for all three -> one dedup key. Scan 1 has the best andromeda,
    # and deliberately NOT the highest intensity, so the two rules are distinguishable.
    idx = MetaIndex()
    for scan, andromeda in ((0, 10.0), (1, 99.0), (2, 50.0)):
        idx.by_key[("RUN_A", scan)] = SpectrumMeta(
            Peptide("PEPTIDEK", ()), 2, 50.0, 5.0, "val", "FTMS", "HCD", 28.0, andromeda
        )
    shard = ShardSpec(path=p, raw_files=("RUN_A",), dataset_id=1, instrument="Lumos")
    got = collect_val_examples([shard], idx, MSContextEncoder(context_dim=8), {1: "poolA"})
    assert len(got) == 1
    assert got[0].label.ms2.max() == pytest.approx(0.5)  # scan 1, not the brightest
