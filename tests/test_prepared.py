"""Prepared-prefix ETL and reader contract tests."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

pytest.importorskip("polars")

from pepdistill.data.prepared import PreparedManifest, PreparedStreamingDataset
from pepdistill.distill.context_regime import MSContextEncoder
from pepdistill.etl.prospect import prepare_source


def _source(tmp_path):
    stem = "TUM_isoform_1"
    root = tmp_path / "source"
    shard_dir = root / stem / stem
    shard_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "raw_file": ["run1", "run1"],
            "scan_number": [1, 2],
            "modified_sequence": ["PEPTIDEK", "ACDEK"],
            "precursor_charge": [2, 2],
            "retention_time": [10.0, 11.0],
            "indexed_retention_time": [20.0, 21.0],
            "aligned_collision_energy": [30.0, 31.0],
            "mass_analyzer": ["FTMS", "FTMS"],
            "fragmentation": ["HCD", "HCD"],
            "andromeda_score": [10.0, 20.0],
        }
    ).to_parquet(root / "meta.parquet", index=False)
    rows = []
    for scan in (1, 2):
        rows.extend(
            [
                {"raw_file": "run1", "scan_number": scan, "ion_type": "b", "no": 1,
                 "charge": 1, "intensity": 1.0, "neutral_loss": None},
                {"raw_file": "run1", "scan_number": scan, "ion_type": "y", "no": 1,
                 "charge": 1, "intensity": 0.5, "neutral_loss": None},
            ]
        )
    pd.DataFrame(rows).to_parquet(shard_dir / "run1.parquet", index=False)
    return root, stem


def test_prepare_source_writes_manifest_and_chunked_rows(tmp_path):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    manifest = prepare_source(str(root), "meta.parquet", stem, str(out), "isoform")
    assert manifest["version"] == 1
    assert len(manifest["chunks"]) == 1
    assert manifest["irt_stats"][0] == 1
    assert (out / "manifest.json").exists()
    assert (out / "val_winners.parquet").exists()

    loaded = PreparedManifest.load(str(out))
    assert loaded.datasets == {"isoform": 1}
    assert len(loaded.val_winners) == 1  # one val winner per sequence/charge
    assert loaded.split_datasets == {"isoform": frozenset({"train", "val"})}


def test_prepared_reader_streams_rows_into_real_batches(tmp_path):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    prepare_source(str(root), "meta.parquet", stem, str(out), "isoform")
    manifest = PreparedManifest.load(str(out))
    ds = PreparedStreamingDataset(
        manifest, MSContextEncoder(context_dim=8), frozenset({"train"}), shuffle_buffer=0
    )
    examples = list(ds.iter_examples(0, shuffle=False))
    assert len(examples) == 1
    batch = next(ds.batches(1, shuffle=False, generator=torch.Generator().manual_seed(0)))
    assert batch.base.ms2_target.shape[0] == 1


def test_prepare_source_skips_complete_matching_manifest(tmp_path, monkeypatch):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    first = prepare_source(str(root), "meta.parquet", stem, str(out), "isoform")

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("a complete matching manifest should skip shard decoding")

    monkeypatch.setattr("pepdistill.etl.prospect._rows_for_shard", unexpected_decode)
    second = prepare_source(str(root), "meta.parquet", stem, str(out), "isoform")
    assert second.pop("_skipped") is True
    assert second == first
