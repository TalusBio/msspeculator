"""Meta index: projected + raw_file-filtered pool meta, split assignment, iRT stats."""

import math
import os

import pandas as pd
import pytest

from pepdistill.data.cache import FileCache
from pepdistill.data.config import SplitConfig
from pepdistill.data.meta_index import build_meta_index
from pepdistill.data.prospect import RECORDS, ProspectSource
from pepdistill.data.split import assign_split

META = "TUM_third_pool_meta_data.parquet"  # exists in the 'prospect' catalog


def _seed_meta(tmp_path):
    root = tmp_path / "local" / "zenodo" / RECORDS["prospect"]
    os.makedirs(root, exist_ok=True)
    # RUN_A deliberately MIXES analyzer/fragmentation/NCE across its two spectra — that is the
    # real third-pool shape, and the shape a per-raw_file acquisition value gets wrong.
    pd.DataFrame(
        {
            "raw_file": ["RUN_A", "RUN_A", "RUN_B"],
            "scan_number": [1, 2, 3],
            "modified_sequence": ["PEPTIDEK", "M[UNIMOD:35]ELVIN", "OTHERPEPK"],
            "precursor_charge": [2, 3, 2],
            "retention_time": [10.0, 20.0, 30.0],
            "indexed_retention_time": [100.0, 200.0, 300.0],
            "aligned_collision_energy": [28.0, 30.0, 32.0],
            "mass_analyzer": ["FTMS", "ITMS", "ITMS"],
            "fragmentation": ["HCD", "CID", "CID"],
        }
    ).to_parquet(root / META)
    return FileCache([str(tmp_path / "local")], write_through=False)


def _src(tmp_path):
    return ProspectSource("prospect", cache=_seed_meta(tmp_path))


def test_index_holds_only_requested_raw_files(tmp_path):
    idx = build_meta_index(_src(tmp_path), META, ["RUN_A"])
    assert set(idx.by_key) == {("RUN_A", 1), ("RUN_A", 2)}


def test_peptide_and_scalars_decoded(tmp_path):
    idx = build_meta_index(_src(tmp_path), META, ["RUN_A"])
    sm = idx.by_key[("RUN_A", 2)]
    assert sm.peptide.sequence == "MELVIN"
    assert sm.peptide.mods == [(0, "Oxidation@M")]
    assert sm.charge == 3 and sm.irt == 200.0 and sm.raw_rt == 20.0


def test_split_matches_assign_split_on_stripped_sequence(tmp_path):
    cfg = SplitConfig()
    idx = build_meta_index(_src(tmp_path), META, ["RUN_A", "RUN_B"], split_cfg=cfg)
    assert idx.by_key[("RUN_A", 1)].split == assign_split("PEPTIDEK", cfg)
    assert idx.by_key[("RUN_B", 3)].split == assign_split("OTHERPEPK", cfg)


def test_allowed_keys_filters_by_split(tmp_path):
    idx = build_meta_index(_src(tmp_path), META, ["RUN_A"])
    every = idx.allowed_keys(["RUN_A"], frozenset({"train", "val", "test"}))
    assert every == {("RUN_A", 1), ("RUN_A", 2)}
    for s in {idx.by_key[("RUN_A", n)].split for n in (1, 2)}:
        assert idx.allowed_keys(["RUN_A"], frozenset({s})) <= every


def test_acquisition_is_per_spectrum_not_per_raw_file(tmp_path):
    """RUN_A mixes FTMS/HCD with ITMS/CID; each spectrum must keep its own."""
    idx = build_meta_index(_src(tmp_path), META, ["RUN_A"])
    a, b = idx.by_key[("RUN_A", 1)], idx.by_key[("RUN_A", 2)]
    assert (a.mass_analyzer, a.fragmentation, a.energy) == ("FTMS", "HCD", 28.0)
    assert (b.mass_analyzer, b.fragmentation, b.energy) == ("ITMS", "CID", 30.0)


def test_irt_stats_are_sum_and_sumsq_over_selected_splits(tmp_path):
    idx = build_meta_index(_src(tmp_path), META, ["RUN_A", "RUN_B"])
    every = frozenset({"train", "val", "test"})
    n, total, sumsq = idx.irt_stats(every)
    assert n == 3
    assert math.isclose(total, 600.0)
    assert math.isclose(sumsq, 100.0**2 + 200.0**2 + 300.0**2)


def test_unknown_raw_file_raises(tmp_path):
    with pytest.raises(ValueError, match="no meta rows"):
        build_meta_index(_src(tmp_path), META, ["RUN_MISSING"])


def test_val_winner_keys_picks_the_highest_andromeda_per_key(tmp_path):
    """Two val observations of one peptide+charge -> only the better-scoring scan survives."""
    root = tmp_path / "local" / "zenodo" / RECORDS["prospect"]
    os.makedirs(root, exist_ok=True)
    seq = "OTHERPEPK"
    assert assign_split(seq, SplitConfig()) == "val", "fixture needs a val-hashed sequence"
    pd.DataFrame(
        {
            "raw_file": ["RUN_A", "RUN_A", "RUN_A"],
            "scan_number": [1, 2, 3],
            "modified_sequence": [seq, seq, seq],
            "precursor_charge": [2, 2, 3],       # scan 3 is a different key
            "retention_time": [10.0, 11.0, 12.0],
            "indexed_retention_time": [100.0, 110.0, 120.0],
            "aligned_collision_energy": [28.0, 28.0, 28.0],
            "mass_analyzer": ["FTMS"] * 3,
            "fragmentation": ["HCD"] * 3,
            "andromeda_score": [10.0, 99.0, 50.0],
        }
    ).to_parquet(root / META)
    src = ProspectSource("prospect", cache=FileCache([str(tmp_path / "local")],
                                                     write_through=False))
    idx = build_meta_index(src, META, ["RUN_A"])
    assert idx.val_winner_keys(["RUN_A"]) == {("RUN_A", 2), ("RUN_A", 3)}


def test_missing_factor_column_warns_and_still_loads_as_unknown(tmp_path):
    """A meta file entirely missing `mass_analyzer` is not fatal (pools may lack it), but it
    must not be silently indistinguishable from a schema misconfiguration: build_meta_index
    warns, naming the absent column, and the resulting SpectrumMeta still carries the
    documented "" unknown-category placeholder."""
    root = tmp_path / "local" / "zenodo" / RECORDS["prospect"]
    os.makedirs(root, exist_ok=True)
    pd.DataFrame(
        {
            "raw_file": ["RUN_A"],
            "scan_number": [1],
            "modified_sequence": ["PEPTIDEK"],
            "precursor_charge": [2],
            "retention_time": [10.0],
            "indexed_retention_time": [100.0],
            "aligned_collision_energy": [28.0],
            "fragmentation": ["HCD"],
            # mass_analyzer deliberately omitted entirely.
        }
    ).to_parquet(root / META)
    src = ProspectSource(
        "prospect", cache=FileCache([str(tmp_path / "local")], write_through=False)
    )
    with pytest.warns(UserWarning, match="mass_analyzer"):
        idx = build_meta_index(src, META, ["RUN_A"])
    assert idx.by_key[("RUN_A", 1)].mass_analyzer == ""
