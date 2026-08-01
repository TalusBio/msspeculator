"""Streaming reader: row-group boundaries, split filtering, determinism, val dedup."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pepdistill.chem import Peptide
from pepdistill.data.cache import FileCache
from pepdistill.data.meta_index import MetaIndex, SpectrumMeta, build_meta_index
from pepdistill.data.prospect import ProspectSource
from pepdistill.distill.real_stream import (
    ShardSpec,
    StreamingRealDataset,
    collect_val_examples,
)
from pepdistill.models.context import MSContextEncoder

_ALL_SPLITS = frozenset({"train", "val", "test"})


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


def test_collect_val_logs_a_winner_that_decodes_to_nothing_instead_of_aborting(tmp_path):
    """A val winner is picked from meta BEFORE its fragments are read, so one that scatters to
    an all-zero MS2 simply disappears -- there is no runner-up fallback. Two things must then
    hold: the run must not abort (the damaged-export guard is a TRAIN-path detector, and the
    val read is narrowed to hand-picked scans that may legitimately yield nothing), and the
    shortfall must be logged rather than shrinking the val set in silence."""
    p = str(tmp_path / "dud.parquet")
    # ordinal 99 on an 8-mer: the row passes the fragment filter but lands on no cell.
    df = pd.DataFrame({
        "ion_type": ["b"], "no": [99], "charge": [1], "intensity": [1.0],
        "neutral_loss": [""], "scan_number": [0], "raw_file": ["RUN_A"],
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    idx = MetaIndex()
    idx.by_key[("RUN_A", 0)] = SpectrumMeta(
        Peptide("PEPTIDEK", ()), 2, 50.0, 5.0, "val", "FTMS", "HCD", 28.0, 100.0
    )
    shard = ShardSpec(path=p, raw_files=("RUN_A",), dataset_id=1, instrument="Lumos")
    lines = []
    got = collect_val_examples([shard], idx, MSContextEncoder(context_dim=8), {1: "poolA"},
                               log=lines.append)
    assert got == []
    assert len(lines) == 1
    assert "0 of 1 winner" in lines[0] and "poolA" in lines[0], lines


def test_collect_val_is_silent_when_every_winner_decodes(tmp_path):
    """The shortfall log must fire on a shortfall, not on every run."""
    shard = _shard(tmp_path, n_spectra=4)
    lines = []
    got = collect_val_examples([shard], _index(4, split="val"),
                               MSContextEncoder(context_dim=8), {1: "poolA"}, log=lines.append)
    assert len(got) == 4  # _index gives each scan its own charge -> four dedup keys
    assert lines == []


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
    enc = MSContextEncoder(context_dim=8)
    ds = StreamingRealDataset([shard], idx, enc, frozenset({"train"}), shuffle_buffer=0)
    # raw_rt is copied straight off the meta row inside the scatter, so it identifies WHICH
    # spectrum an example is independently of the factor binding under test. Asserting the
    # exact triple per identified spectrum (rather than e0 != e1) is what makes this fail when
    # the keys->SpectrumMeta binding is reversed instead of merely collapsed.
    examples = sorted(ds.iter_examples(epoch=0), key=lambda e: e.raw_rt)
    assert len(examples) == 2
    e0, e1 = examples
    assert (e0.raw_rt, e0.detector_id, e0.fragmentation_id, e0.energy) == (
        5.0, enc.detector_id("ITMS"), enc.fragmentation_id("CID"), 30.0
    )
    assert (e1.raw_rt, e1.detector_id, e1.fragmentation_id, e1.energy) == (
        6.0, enc.detector_id("FTMS"), enc.fragmentation_id("HCD"), 28.0
    )
    assert e0.detector_id != e1.detector_id  # the categories must be distinct ids at all


# --- streaming vs reference-decoder equivalence -------------------------------------------

_EQUIV_META = [
    # (raw_file, scan, modified_sequence, charge, irt, raw_rt, nce, analyzer, frag, andromeda)
    ("RUN_A", 11, "PEPTIDEK", 2, 50.0, 5.0, 28.0, "FTMS", "HCD", 90.0),
    ("RUN_A", 12, "[UNIMOD:21]SPEPTIDEK", 3, 51.0, 6.0, 30.0, "ITMS", "CID", 70.0),
    ("RUN_A", 13, "A[UNIMOD:1]CDEFGHIK", 2, 52.0, 7.0, 28.0, "FTMS", "HCD", 80.0),
    ("RUN_B", 21, "LNGVIDEMK", 2, 53.0, 8.0, 27.0, "FTMS", "HCD", 60.0),
    ("RUN_B", 22, "SAMPLERPEPK", 3, 54.0, 9.0, 33.0, "ITMS", "CID", 50.0),
    ("RUN_B", 23, "QWERTYIPK", 2, 55.0, 10.0, 28.0, "FTMS", "HCD", 40.0),
]

_EQUIV_META_COLUMNS = [
    "raw_file", "scan_number", "modified_sequence", "precursor_charge",
    "indexed_retention_time", "retention_time", "aligned_collision_energy",
    "mass_analyzer", "fragmentation", "andromeda_score",
]

_ANN_COLUMNS = [
    "raw_file", "scan_number", "ion_type", "no", "charge", "intensity", "neutral_loss",
]


def _equiv_frames():
    """Meta + long-format annotation covering every branch the two paths must agree on.

    Deliberately included: two raw files interleaved, spectra of different lengths, a
    duplicated (site, col) cell that only max-collapse resolves, an out-of-range ordinal that
    scatters nowhere, and rows the fragment filter must drop (3+ fragment charge, a neutral
    loss, a non-b/y ion type).
    """
    meta = pd.DataFrame(_EQUIV_META, columns=_EQUIV_META_COLUMNS)
    rows = []
    for rf, scan, *_ in _EQUIV_META:
        for ordinal in (1, 2, 3, 4):
            rows.append((rf, scan, "b", ordinal, 1, 0.1 * ordinal, ""))
            rows.append((rf, scan, "y", ordinal, 2, 0.05 * ordinal, ""))
        rows.append((rf, scan, "b", 1, 1, 0.9, ""))       # duplicate cell -> max wins
        rows.append((rf, scan, "b", 99, 1, 1.0, ""))      # ordinal off the end -> no cell
        rows.append((rf, scan, "y", 1, 3, 1.0, ""))       # fragment charge 3 -> filtered
        rows.append((rf, scan, "b", 2, 1, 1.0, "H2O"))    # neutral loss -> filtered
        rows.append((rf, scan, "precursor", 0, 1, 1.0, ""))  # not b/y -> filtered
    # Interleave the two raw files so a row group can span both.
    rows.sort(key=lambda r: r[1] % 10)
    return meta, pd.DataFrame(rows, columns=_ANN_COLUMNS)


def _fingerprint(triples):
    """(modified_sequence, charge, raw_rt) -> MS2 bytes. Order-independent by construction."""
    return {
        (p.peptide.modified_sequence(), p.charge, float(rrt)): lab.ms2.tobytes()
        for p, lab, rrt in triples
    }


def test_streaming_matches_the_reference_decoder_on_the_same_shard(tmp_path, monkeypatch):
    """The same shard through StreamingRealDataset and through ProspectSource.to_labels must
    yield the SAME example set (order-independent).

    to_labels is the reference: it reads the annotation as one frame, so it has no row groups,
    no split filter and no per-shard bookkeeping. The streaming path reassembles the same
    spectra out of row groups that cut through them. Anything that breaks reassembly, the
    split filter, or the scatter shows up as a difference here rather than as a hand-written
    expectation that was updated to match the new behaviour.
    """
    meta, ann = _equiv_frames()
    meta_path = str(tmp_path / "meta.parquet")
    ann_path = str(tmp_path / "shard.parquet")
    meta.to_parquet(meta_path)
    # 5 rows per group against 13 annotation rows per spectrum: every spectrum straddles at
    # least two row groups, and most row groups hold fragments of more than one spectrum.
    pq.write_table(pa.Table.from_pandas(ann, preserve_index=False), ann_path, row_group_size=5)
    assert pq.ParquetFile(ann_path).num_row_groups > 1

    src = ProspectSource("prospect", cache=FileCache([str(tmp_path / "cache")]))
    reference = src.to_labels(meta, pd.read_parquet(ann_path))
    ref_map = _fingerprint(zip(reference.precursors, reference.labels, reference.raw_rt))
    assert len(ref_map) == len(_EQUIV_META)  # the reference itself decoded everything

    monkeypatch.setattr(src, "resolve_file", lambda filename: meta_path)
    index = build_meta_index(src, "meta.parquet", ["RUN_A", "RUN_B"])
    # The fixture spans more than one split, so the split filter is genuinely exercised rather
    # than trivially uniform -- and asking for all three splits is what makes the streamed set
    # comparable to to_labels, which does not filter by split at all.
    assert len({sm.split for sm in index.by_key.values()}) > 1
    shard = ShardSpec(path=ann_path, raw_files=("RUN_A", "RUN_B"), dataset_id=1,
                      instrument="Lumos")
    ds = StreamingRealDataset([shard], index, MSContextEncoder(context_dim=8), _ALL_SPLITS,
                              shuffle_buffer=0)
    streamed = list(ds.iter_examples(epoch=0))
    stream_map = _fingerprint((e.precursor, e.label, e.raw_rt) for e in streamed)

    assert len(streamed) == len(stream_map)  # no key collapsed two examples
    assert stream_map == ref_map
