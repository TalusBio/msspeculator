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


def _write_one_scan_shard(path, raw_file, intensity):
    rows = [("b", ordinal, 1, intensity, "", 0, raw_file) for ordinal in (1, 2, 3, 4)]
    df = pd.DataFrame(
        rows,
        columns=["ion_type", "no", "charge", "intensity", "neutral_loss",
                 "scan_number", "raw_file"],
    )
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return path


def test_collect_val_dedups_across_shards_of_the_same_dataset(tmp_path):
    """Two shards (two raw files) of ONE dataset share a val dedup key. The winner must be
    chosen across BOTH shards, not once per shard -- else the key emits twice."""
    p_a = _write_one_scan_shard(str(tmp_path / "a.parquet"), "RUN_A", 0.2)
    p_b = _write_one_scan_shard(str(tmp_path / "b.parquet"), "RUN_B", 0.7)

    idx = MetaIndex()
    idx.by_key[("RUN_A", 0)] = SpectrumMeta(
        Peptide("PEPTIDEK", ()), 2, 50.0, 5.0, "val", "FTMS", "HCD", 28.0, 10.0
    )
    idx.by_key[("RUN_B", 0)] = SpectrumMeta(
        Peptide("PEPTIDEK", ()), 2, 50.0, 5.0, "val", "FTMS", "HCD", 28.0, 99.0
    )
    shard_a = ShardSpec(path=p_a, raw_files=("RUN_A",), dataset_id=1, instrument="Lumos")
    shard_b = ShardSpec(path=p_b, raw_files=("RUN_B",), dataset_id=1, instrument="Lumos")
    got = collect_val_examples([shard_a, shard_b], idx, MSContextEncoder(context_dim=8),
                                {1: "poolA"})
    assert len(got) == 1
    assert got[0].label.ms2.max() == pytest.approx(0.7)  # RUN_B: the higher-andromeda scan


def test_collect_val_raises_for_an_unmapped_dataset_id(tmp_path):
    shard = _shard(tmp_path, n_spectra=4)
    with pytest.raises(KeyError, match="dataset_id"):
        collect_val_examples([shard], _index(4, split="val"), MSContextEncoder(context_dim=8), {})


def test_shard_whose_raw_files_match_no_meta_row_raises(tmp_path):
    """raw_files that don't correspond to anything in the index at all -- the mistake of
    deriving raw_files from the shard filename instead of shard_raw_files(path) -- must fail
    loudly, distinct from a shard that's merely empty for the requested split."""
    p = str(tmp_path / "shard.parquet")
    _write_shard(p, n_spectra=4, row_group_size=3)
    shard = ShardSpec(path=p, raw_files=("WRONG_RUN",), dataset_id=1, instrument="Lumos")
    ds = StreamingRealDataset([shard], _index(4), MSContextEncoder(context_dim=8),
                              frozenset({"train"}), shuffle_buffer=0)
    with pytest.raises(ValueError, match="match no meta rows"):
        list(ds.iter_examples(epoch=0))


def test_shard_where_filtered_rows_never_scatter_to_a_spectrum_raises(tmp_path):
    """Rows survive the fragment+split filter, but the peptide is too short to have any valid
    fragment site, so decode_fragments returns zero examples from a non-empty frame -- the
    ``not real.precursors`` branch, distinct from the ``not kept`` branch."""
    p = str(tmp_path / "short.parquet")
    df = pd.DataFrame({
        "ion_type": ["b"], "no": [1], "charge": [1], "intensity": [1.0],
        "neutral_loss": [""], "scan_number": [0], "raw_file": ["RUN_A"],
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    idx = MetaIndex()
    idx.by_key[("RUN_A", 0)] = SpectrumMeta(
        Peptide("P", ()), 2, 50.0, 5.0, "train", "FTMS", "HCD", 28.0, 100.0
    )
    shard = ShardSpec(path=p, raw_files=("RUN_A",), dataset_id=1, instrument="Lumos")
    ds = StreamingRealDataset([shard], idx, MSContextEncoder(context_dim=8),
                              frozenset({"train"}), shuffle_buffer=0)
    with pytest.raises(ValueError, match="zero usable examples"):
        list(ds.iter_examples(epoch=0))


def test_batches_shuffle_false_is_sequential_not_just_epoch_pinned(tmp_path):
    """shuffle=False must mean sequential order, not merely a pinned epoch: with the default
    shuffle_buffer (50_000, far bigger than 10 examples) an unshuffled pass used to still
    shuffle shard order and the buffer."""
    shard = _shard(tmp_path, n_spectra=10)
    ds = StreamingRealDataset([shard], _index(10), MSContextEncoder(context_dim=8),
                              frozenset({"train"}))  # default shuffle_buffer=50_000
    gen = torch.Generator().manual_seed(0)
    batches = list(ds.batches(batch_size=10, shuffle=False, generator=gen))
    got_rt = batches[0].raw_rt.tolist()
    assert got_rt == sorted(got_rt)  # scan order 0..9 -> raw_rt is already increasing


def test_per_spectrum_acquisition_factors_are_not_collapsed_to_one_value(tmp_path):
    """Two scans in ONE raw file with different analyzer/fragmentation/energy must each surface
    their OWN factors -- the defect Task 2 exists to fix. An implementation that resolved
    acquisition factors once per raw file (e.g. from the first spectrum) would pass every other
    test in this file but fail this one."""
    p = str(tmp_path / "mixed.parquet")
    rows = []
    for scan in (0, 1):
        for ordinal in (1, 2, 3, 4):
            rows.append(("b", ordinal, 1, 1.0, "", scan, "RUN_A"))
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(rows, columns=["ion_type", "no", "charge", "intensity",
                                        "neutral_loss", "scan_number", "raw_file"]),
            preserve_index=False,
        ),
        p,
    )
    idx = MetaIndex()
    idx.by_key[("RUN_A", 0)] = SpectrumMeta(
        Peptide("PEPTIDEK", ()), 2, 50.0, 5.0, "train", "ITMS", "CID", 30.0, 100.0
    )
    idx.by_key[("RUN_A", 1)] = SpectrumMeta(
        Peptide("PEPTIDEK", ()), 2, 51.0, 6.0, "train", "FTMS", "HCD", 28.0, 100.0
    )
    shard = ShardSpec(path=p, raw_files=("RUN_A",), dataset_id=1, instrument="Lumos")
    ds = StreamingRealDataset([shard], idx, MSContextEncoder(context_dim=8),
                              frozenset({"train"}), shuffle_buffer=0)
    examples = sorted(ds.iter_examples(epoch=0), key=lambda e: e.raw_rt)
    assert len(examples) == 2
    e0, e1 = examples
    assert e0.detector_id != e1.detector_id
    assert e0.fragmentation_id != e1.fragmentation_id
    assert e0.energy != e1.energy
