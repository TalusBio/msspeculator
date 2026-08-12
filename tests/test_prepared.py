"""Prepared-prefix ETL and reader contract tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip("polars")

import polars as pl

from pepdistill.data.prepared import PreparedChunk, PreparedManifest, PreparedStreamingDataset
from pepdistill.data.prepared_schema import (
    PREPARED_SPECTRA_SCHEMA,
    VALIDATION_WINNER_SCHEMA,
)
from pepdistill.distill.context_regime import MSContextEncoder
from pepdistill.distill.dataset import BatchIterable
from pepdistill.etl.config import PrepareConfig, PrepareGroup, PrepareSource
from pepdistill.etl.prospect import (
    balanced_partition_range,
    catalog_status,
    discover_catalog,
    ensure_catalog,
    finalize_catalog,
    prepare_range,
)


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
                {
                    "raw_file": "run1",
                    "scan_number": scan,
                    "ion_type": "b",
                    "no": 1,
                    "charge": 1,
                    "intensity": 1.0,
                    "neutral_loss": None,
                },
                {
                    "raw_file": "run1",
                    "scan_number": scan,
                    "ion_type": "y",
                    "no": 1,
                    "charge": 1,
                    "intensity": 0.5,
                    "neutral_loss": None,
                },
            ]
        )
    pd.DataFrame(rows).to_parquet(shard_dir / "run1.parquet", index=False)
    return root, stem


def _config(root, stem, out):
    return PrepareConfig(
        source_prefix=str(root),
        output_prefix=str(out),
        sources=(
            PrepareSource(id="isoform", dataset="isoform", meta="meta.parquet", archive=stem),
        ),
    )


def test_prepare_shards_writes_manifest_and_chunked_rows(tmp_path):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    manifest = finalize_catalog(config, log=None)
    assert manifest["version"] == 1
    assert len(manifest["chunks"]) == 1
    assert manifest["irt_stats"][0] == 1
    assert (out / "manifest.json").exists()
    assert (out / "validation" / "val_winners.parquet").exists()
    assert pl.read_parquet_schema(manifest["chunks"][0]["uri"]) == PREPARED_SPECTRA_SCHEMA
    assert (
        pl.read_parquet_schema(out / "validation" / "val_winners.parquet")
        == VALIDATION_WINNER_SCHEMA
    )

    loaded = PreparedManifest.load(str(out))
    assert loaded.datasets == {"isoform": 1}
    assert len(loaded.val_winners) == 1  # one val winner per sequence/charge
    assert loaded.split_datasets == {"isoform": frozenset({"train", "val"})}


def test_prepared_reader_streams_rows_into_real_batches(tmp_path):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    finalize_catalog(config, log=None)
    manifest = PreparedManifest.load(str(out))
    logs: list[str] = []
    ds = PreparedStreamingDataset(
        manifest, MSContextEncoder(context_dim=8), frozenset({"train"}), log=logs.append
    )
    examples = list(ds.iter_examples(0, shuffle=False))
    assert len(examples) == 1
    assert len(logs) == 1
    assert logs[0].startswith("[data] shard 1/1, dataset=isoform, rows=1, open=")
    assert "s, read_decode=" in logs[0]
    batch = next(ds.batches(1, shuffle=False, generator=torch.Generator().manual_seed(0)))
    assert batch.base.ms2_target.shape[0] == 1


def test_prepared_reader_partitions_shards_across_loader_workers(tmp_path):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared-workers"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    finalize_catalog(config, log=None)
    ds = PreparedStreamingDataset(
        PreparedManifest.load(str(out)),
        MSContextEncoder(context_dim=8),
        frozenset({"train", "val"}),
    )
    loader = DataLoader(
        BatchIterable(ds, batch_size=1, shuffle=False, seed=0),
        batch_size=None,
        num_workers=2,
    )
    # The fixture contains two total rows in one shard. A naive IterableDataset would replay
    # that shard in both workers and return four; the worker-aware reader must return it once.
    assert sum(batch.raw_rt.numel() for batch in loader) == 2


def test_prepare_shard_skips_complete_manifest(tmp_path, monkeypatch):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    logs: list[str] = []
    config = _config(root, stem, out)
    first = prepare_range(config, log=logs.append)[0]
    assert logs[0].startswith("[prepare]")

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("a complete matching manifest should skip shard decoding")

    monkeypatch.setattr("pepdistill.etl.prospect._rows_for_shard", unexpected_decode)
    second = prepare_range(config, log=logs.append)[0]
    assert second.pop("_skipped") is True
    assert second == first
    assert "complete manifest exists; skipping" in logs[-1]


def test_shard_catalog_range_and_finalize(tmp_path):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared-shards"
    config = _config(root, stem, out)
    prepared = prepare_range(config, start=0, stop=1, log=None)
    assert len(prepared) == 1
    assert catalog_status(config) == {"complete": 1, "missing": 0, "total": 1}
    manifest = finalize_catalog(config, log=None)
    assert manifest["datasets"] == {"isoform": 1}
    assert manifest["chunks"][0]["uri"].endswith("shards/isoform/000000/data.parquet")
    assert PreparedManifest.load(str(out)).chunks[0].rows == 2


def test_finalize_and_reader_exclude_nonfinite_rt_rows(tmp_path):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared-finite-rt"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    data = out / "shards" / "isoform" / "000000" / "data.parquet"
    frame = pd.read_parquet(data)
    invalid = frame.iloc[[0]].copy()
    invalid["spectrum_id"] += 10_000
    invalid["split"] = "train"
    invalid["irt"] = float("nan")
    pd.concat([frame, invalid], ignore_index=True).to_parquet(data, index=False)

    finalized = finalize_catalog(config, log=None)
    assert finalized["irt_stats"][0] == 1
    assert finalized["split_rows"] == {"train": 1, "val": 1}
    manifest = PreparedManifest.load(str(out))
    ds = PreparedStreamingDataset(manifest, MSContextEncoder(context_dim=8), frozenset({"train"}))
    assert len(list(ds.iter_examples(0, shuffle=False))) == 1


def test_prepared_reader_accepts_int128_spectrum_ids(tmp_path):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared-int128-id"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    data = out / "shards" / "isoform" / "000000" / "data.parquet"
    frame = pl.read_parquet(data).with_columns(
        (pl.col("spectrum_id").cast(pl.Int128) + pl.lit(2**63, dtype=pl.Int128)).alias(
            "spectrum_id"
        )
    )
    frame.write_parquet(data)

    finalize_catalog(config, log=None)
    manifest = PreparedManifest.load(str(out))
    assert manifest.val_winners and max(manifest.val_winners) > 2**63
    val = PreparedStreamingDataset(manifest, MSContextEncoder(context_dim=8), frozenset({"val"}))
    assert len(list(val.iter_examples(0, shuffle=False))) == 1


def test_validation_reader_skips_train_only_shard_with_boolean_winner_mask(tmp_path):
    """An empty per-shard winner list must remain a Boolean predicate, not Polars Null."""
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared-empty-val-shard"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    finalize_catalog(config, log=None)
    manifest = PreparedManifest.load(str(out))

    original = manifest.chunks[0]
    train_only_path = out / "train-only.parquet"
    pl.read_parquet(original.uri).with_columns(pl.lit("train").alias("split")).write_parquet(
        train_only_path
    )
    train_only = PreparedChunk(
        uri=str(train_only_path),
        dataset=original.dataset,
        rows=original.rows,
        source_shard="train-only",
    )
    manifest = PreparedManifest(
        version=manifest.version,
        chunks=manifest.chunks + (train_only,),
        datasets=manifest.datasets,
        val_winners=manifest.val_winners,
        irt_stats=manifest.irt_stats,
        split_rows=manifest.split_rows,
        split_datasets=manifest.split_datasets,
    )
    logs: list[str] = []
    val = PreparedStreamingDataset(
        manifest,
        MSContextEncoder(context_dim=8),
        frozenset({"val"}),
        log=logs.append,
    )

    assert len(list(val.iter_examples(0, shuffle=False))) == 1
    assert len(logs) == 2
    assert "rows=0" in logs[1]


def test_catalog_status_and_finalize_do_not_head_each_data_object(tmp_path, monkeypatch):
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared-batched"
    config = _config(root, stem, out)
    prepare_range(config, log=None)

    def unexpected_head(*args, **kwargs):
        raise AssertionError("catalog validation should use the prefix inventory")

    monkeypatch.setattr("pepdistill.etl.prospect._uri_exists", unexpected_head)
    assert catalog_status(config) == {"complete": 1, "missing": 0, "total": 1}
    assert len(finalize_catalog(config, log=None)["chunks"]) == 1


def test_group_config_discovers_matching_archives(tmp_path):
    config = PrepareConfig(
        source_prefix=str(tmp_path / "cache"),
        output_prefix=str(tmp_path / "prepared"),
        groups=(
            PrepareGroup(
                record="prospect",
                include=("TUM_isoform_*",),
                dataset_prefix="prospect",
            ),
        ),
    )
    catalog = discover_catalog(config)
    assert len(catalog["tasks"]) > 1
    assert {task["source_id"] for task in catalog["tasks"]} == {
        "prospect_TUM_isoform_1",
        "prospect_TUM_isoform_2",
    }
    assert catalog["tasks"][0]["dataset"] == "prospect_tum_isoform"
    assert catalog["version"] == 2
    assert {task["meta_uri"].rsplit("/", 1)[-1] for task in catalog["tasks"]} == {
        "TUM_isoform_meta_data.parquet"
    }
    assert all(
        task["meta_url"].endswith("TUM_isoform_meta_data.parquet/content")
        for task in catalog["tasks"]
    )


def test_group_catalog_v1_is_rebuilt_for_shared_metadata_fix(tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    config = PrepareConfig(
        source_prefix=str(tmp_path / "cache"),
        output_prefix=str(output),
        groups=(PrepareGroup(record="prospect", include=("TUM_isoform_1",)),),
    )
    (output / "catalog.json").write_text(
        json.dumps(
            {
                "version": 1,
                "config_fingerprint": config.fingerprint,
                "tasks": [{"meta_url": "", "source_id": "prospect_TUM_isoform_1"}],
            }
        )
    )

    catalog = ensure_catalog(config)

    assert catalog["version"] == 2
    assert catalog["tasks"][0]["meta_uri"].endswith("TUM_isoform_meta_data.parquet")
    assert catalog["tasks"][0]["meta_url"].endswith("TUM_isoform_meta_data.parquet/content")


def test_balanced_partition_uses_vendored_raw_bytes(monkeypatch):
    catalog = {
        "tasks": [{"record": "r", "archive": "a", "shard_index": index} for index in range(4)]
    }
    monkeypatch.setattr(
        "pepdistill.etl.prospect.load_shard_index",
        lambda: {
            "records": {"r": {"a.zip": [["0", 1, 10], ["1", 1, 10], ["2", 1, 70], ["3", 1, 10]]}}
        },
    )
    assert balanced_partition_range(catalog, 0, 2) == (0, 2, 20)
    assert balanced_partition_range(catalog, 1, 2) == (2, 4, 80)
