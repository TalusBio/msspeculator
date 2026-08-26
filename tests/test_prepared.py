"""Prepared-prefix ETL and reader contract tests."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

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
from pepdistill.etl.config import PrepareConfig, PrepareCuration, PrepareGroup, PrepareSource
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


def _repeated_source(tmp_path):
    """A run where one peptidoform is sampled repeatedly across its elution and two charges."""
    stem = "TUM_isoform_1"
    root = tmp_path / "source"
    shard_dir = root / stem / stem
    shard_dir.mkdir(parents=True)
    # PEPTIDEK apexes at scan 4; scans 1 and 6 sit on the tails below half maximum. Scan 7 is
    # the same peptidoform at charge 3, so it must share the charge-independent window.
    scans = [1, 2, 3, 4, 5, 6, 7, 8]
    pd.DataFrame(
        {
            "raw_file": ["run1"] * len(scans),
            "scan_number": scans,
            "modified_sequence": ["PEPTIDEK"] * 7 + ["ACDEK"],
            "precursor_charge": [2, 2, 2, 2, 2, 2, 3, 2],
            "retention_time": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 10.5, 20.0],
            "indexed_retention_time": [20.0, 20.2, 20.4, 20.6, 20.8, 21.0, 20.5, 40.0],
            "aligned_collision_energy": [30.0] * len(scans),
            "mass_analyzer": ["FTMS"] * len(scans),
            "fragmentation": ["HCD"] * len(scans),
            "andromeda_score": [10.0, 20.0, 50.0, 90.0, 70.0, 30.0, 100.0, 60.0],
            "precursor_intensity": [10.0, 60.0, 90.0, 100.0, 70.0, 20.0, 80.0, 100.0],
        }
    ).to_parquet(root / "meta.parquet", index=False)
    rows = []
    for scan in scans:
        rows.extend(
            [
                {
                    "raw_file": "run1",
                    "scan_number": scan,
                    "ion_type": ion,
                    "no": 1,
                    "charge": 1,
                    "intensity": intensity,
                    "neutral_loss": None,
                }
                for ion, intensity in (("b", 1.0), ("y", 0.5))
            ]
        )
    pd.DataFrame(rows).to_parquet(shard_dir / "run1.parquet", index=False)
    return root, stem


def test_prepare_curation_keeps_top_psms_per_context_and_reports(tmp_path):
    root, stem = _repeated_source(tmp_path)
    out = tmp_path / "prepared-curated"
    config = PrepareConfig(
        source_prefix=str(root),
        output_prefix=str(out),
        curation=PrepareCuration(
            enabled=True,
            min_in_window_psms=3,
            max_psms_per_context=2,
            width_anchor_min_psms=4,
            # The fixture's replicates are 0.2 min apart for readable arithmetic, so widen the
            # plausible-width clamp out of the way; clamping is covered in test_curation.py.
            max_run_width_minutes=60.0,
        ),
        sources=(
            PrepareSource(id="isoform", dataset="isoform", meta="meta.parquet", archive=stem),
        ),
    )
    logs: list[str] = []
    manifest = prepare_range(config, log=logs.append)[0]

    data = out / "shards" / "isoform" / "000000" / "data.parquet"
    assert pl.read_parquet_schema(data) == PREPARED_SPECTRA_SCHEMA
    selected = pl.read_parquet(data).sort("scan_number")
    # Half-maximum support spans scans 2-5 and 7 (>= 50 of the 100 apex), so the shared window
    # is 0.6 min wide centered on scan 4 at 10.6: scans 3, 4, 5 and 7 fall inside it. The
    # charge-2 context keeps its two best-scoring members; charge 3 is a separate context.
    assert selected["scan_number"].to_list() == [4, 5, 7]
    assert selected["charge"].to_list() == [2, 2, 3]
    # The single-PSM peptidoform cannot meet in-window replication and is dropped entirely.
    assert not any("ACDEK" in value for value in selected["proforma"].to_list())
    assert manifest["rows"] == 3

    report = manifest["curation"]
    assert report["policy"]["min_in_window_psms"] == 3
    assert report["policy"]["max_psms_per_context"] == 2
    assert report["input"]["rows"] == 8
    assert report["input"]["missing_precursor_intensity_rows"] == 0
    assert report["chromatography"]["run_widths"]["run1"]["width_minutes"] == pytest.approx(0.6)
    # Four PEPTIDEK PSMs plus the lone ACDEK PSM, which sits inside its own window but is
    # rejected by the replication floor rather than by the window itself.
    assert report["selection"]["apex_window_rows"] == 5
    assert report["selection"]["qualifying_peptidoforms"] == 1
    assert report["selection"]["rejected_peptidoforms"] == 1
    assert report["selection"]["selected_rows"] == 3
    assert report["selection"]["selected_contexts"] == 2
    assert any("curation retained 3/8 spectra" in line for line in logs)


def test_prepare_curation_refuses_sources_without_precursor_intensity(tmp_path):
    """A source lacking the intensity column must stop the task, not curate away every row.

    Precursor intensity is optional in source metadata, so a renamed or dropped column arrives as
    all-NaN and curation finds no measurable elution width anywhere. Writing the resulting 0-row
    shard plus a valid manifest would let a whole source vanish from the corpus while every
    downstream check reports success.
    """
    root, stem = _repeated_source(tmp_path)
    meta_path = root / "meta.parquet"
    pd.read_parquet(meta_path).drop(columns=["precursor_intensity"]).to_parquet(
        meta_path, index=False
    )
    out = tmp_path / "prepared-no-intensity"
    config = PrepareConfig(
        source_prefix=str(root),
        output_prefix=str(out),
        curation=PrepareCuration(enabled=True, min_in_window_psms=3, width_anchor_min_psms=4),
        sources=(
            PrepareSource(id="isoform", dataset="isoform", meta="meta.parquet", archive=stem),
        ),
    )
    with pytest.raises(ValueError, match="no usable precursor_intensity"):
        prepare_range(config, log=None)
    # Nothing was published for the failed task.
    assert not (out / "shards" / "isoform" / "000000" / "manifest.json").exists()


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
        # Match the supported production loader; fork can deadlock after Torch starts threads.
        multiprocessing_context="spawn",
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


def test_a_row_missing_its_irt_is_kept_but_left_out_of_the_affine(tmp_path):
    """A row without iRT still carries MS2 and its own run's retention time, so it belongs in
    the corpus with the iRT term masked (see ``losses.labeled_mse``) rather than dropped. The
    affine is the one place it must not appear: it would have no value to contribute."""
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
    assert finalized["irt_stats"][0] == 1  # the unlabeled row cannot move the RT affine
    # split_rows is the loader's own row count, so it must include the unlabeled row.
    assert finalized["split_rows"] == {"train": 2, "val": 1}
    manifest = PreparedManifest.load(str(out))
    ds = PreparedStreamingDataset(manifest, MSContextEncoder(context_dim=8), frozenset({"train"}))
    examples = list(ds.iter_examples(0, shuffle=False))
    assert len(examples) == 2
    assert sum(1 for example in examples if math.isnan(example.label.rt)) == 1


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


def test_policy_version_bump_restages_published_shards(tmp_path, monkeypatch):
    """A policy that moves in code must invalidate the shards built by the old one.

    The fingerprint covers the config, so before this a labelling change left a corpus reporting
    itself complete and current while holding rows the new code would never emit. Both halves are
    pinned: the bump has to move the fingerprint, and the baseline has to leave it alone, because
    version 1 is encoded as absence so it reproduces fingerprints from before the field existed.
    """
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared-policy"
    config = _config(root, stem, out)

    # Version 1 is encoded as the absence of the field, so it reproduces fingerprints computed
    # before the field existed. Asserted by pinning the version rather than by reading today's,
    # which has since moved on.
    monkeypatch.setattr("pepdistill.etl.config.PREPARE_POLICY_VERSION", 1)
    monkeypatch.setattr("pepdistill.etl.prospect.PREPARE_POLICY_VERSION", 1)
    assert "policy_version" not in config.canonical()
    version_one = config.fingerprint

    first = prepare_range(config, log=None)
    assert [entry.get("_skipped") for entry in first] == [None]
    # Unchanged policy: the shard is complete and stays that way.
    assert [entry.get("_skipped") for entry in prepare_range(config, log=None)] == [True]

    monkeypatch.setattr("pepdistill.etl.config.PREPARE_POLICY_VERSION", 2)
    monkeypatch.setattr("pepdistill.etl.prospect.PREPARE_POLICY_VERSION", 2)
    assert config.canonical()["policy_version"] == 2
    assert config.fingerprint != version_one

    # The moved fingerprint restages the catalog, which restages the shard: rebuilt, not skipped.
    assert discover_catalog(config)["policy_version"] == 2
    assert [entry.get("_skipped") for entry in prepare_range(config, log=None)] == [None]

    manifest = json.loads((out / "shards" / "isoform" / "000000" / "manifest.json").read_text())
    assert manifest["task"]["policy_version"] == 2


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


def test_load_shard_manifests_reads_a_prefix_without_a_config(tmp_path):
    """Auditing a corpus must not require the policy that built it.

    A policy change moves the config fingerprint, so going through PrepareConfig would hide every
    shard of an older corpus; and answering a question about a published prefix must not rewrite
    its catalog as a side effect. So this reads the prefix directly.
    """
    from pepdistill.data.prepared import load_shard_manifests

    root, stem = _source(tmp_path)
    out = tmp_path / "prepared-manifests"
    prepare_range(_config(root, stem, out), log=None)

    logs: list[str] = []
    manifests = load_shard_manifests(str(out), log=logs.append)
    assert len(manifests) == 1
    assert manifests[0]["task"]["dataset"] == "isoform"
    assert manifests[0]["rows"] == 2
    assert "1 shard manifest(s)" in logs[0]

    # An absent prefix is empty, not an error: a corpus may simply not exist yet.
    assert load_shard_manifests(str(tmp_path / "never-prepared")) == []


def test_local_cache_fetches_each_shard_once_and_then_needs_no_source(tmp_path):
    """A warmed cache serves the corpus with the remote gone, and never refetches.

    Exercised over ``file://`` so the remote branch of the reader runs without a network: a plain
    local path bypasses caching entirely, so a test using one would assert nothing.
    """
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    finalize_catalog(config, log=None)

    manifest_path = out / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    for chunk in raw["chunks"]:
        chunk["uri"] = Path(chunk["uri"]).as_uri()
    manifest_path.write_text(json.dumps(raw))

    manifest = PreparedManifest.load(out)
    assert all(chunk.uri.startswith("file://") for chunk in manifest.chunks)

    cache = tmp_path / "cache"
    dataset = PreparedStreamingDataset(
        manifest,
        MSContextEncoder(context_dim=8),
        frozenset({"train"}),
        local_cache=cache,
    )
    warm = list(dataset.iter_examples(epoch=0, shuffle=False))
    assert warm, "cached read yielded no examples"
    assert len(list(cache.rglob("*.parquet"))) == len(manifest.chunks)
    # A partial file left behind would later be accepted as a complete shard.
    assert not list(cache.rglob("*.partial"))

    # With the source removed, only the cache can answer; so a refetch would raise here rather
    # than quietly succeed, which makes this a stronger check than comparing mtimes.
    shutil.rmtree(out)
    again = list(dataset.iter_examples(epoch=0, shuffle=False))
    assert len(again) == len(warm)


def test_in_memory_holds_decoded_shards_across_epochs(tmp_path):
    """A resident corpus replays every epoch from RAM, decoding each shard exactly once."""
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    finalize_catalog(config, log=None)

    manifest = PreparedManifest.load(out)
    dataset = PreparedStreamingDataset(
        manifest, MSContextEncoder(context_dim=8), frozenset({"train"}), in_memory=True
    )
    decoded = 0
    original = dataset._decode_chunk

    def counting(chunk, position):
        nonlocal decoded
        decoded += 1
        return original(chunk, position)

    dataset._decode_chunk = counting  # type: ignore[method-assign]
    first = list(dataset.iter_examples(epoch=0, shuffle=False))
    assert first, "resident read yielded no examples"
    assert decoded == len(manifest.chunks)

    # Removing the source proves the second epoch came from RAM, and the decode count proves it
    # was not re-parsed: either check alone would pass for the wrong reason.
    shutil.rmtree(out)
    second = list(dataset.iter_examples(epoch=1, shuffle=False))
    assert len(second) == len(first)
    assert decoded == len(manifest.chunks)


def test_without_a_local_cache_shards_are_read_from_their_source_every_time(tmp_path):
    """The complement of the test above: no cache means no local copy to fall back on."""
    root, stem = _source(tmp_path)
    out = tmp_path / "prepared"
    config = _config(root, stem, out)
    prepare_range(config, log=None)
    finalize_catalog(config, log=None)

    manifest_path = out / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    for chunk in raw["chunks"]:
        chunk["uri"] = Path(chunk["uri"]).as_uri()
    manifest_path.write_text(json.dumps(raw))

    manifest = PreparedManifest.load(out)
    dataset = PreparedStreamingDataset(
        manifest, MSContextEncoder(context_dim=8), frozenset({"train"})
    )
    assert list(dataset.iter_examples(epoch=0, shuffle=False))
    shutil.rmtree(out)
    with pytest.raises(FileNotFoundError):
        list(dataset.iter_examples(epoch=0, shuffle=False))


def test_vendored_manifests_state_their_own_provenance():
    """Each manifest has to say where it came from and what regenerates it.

    JSON takes no comments, so `_source` / `_generator` keys are the in-band equivalent of the
    `#` headers on the vendored TSVs. They are written by the builders rather than hand-added, so
    a refresh cannot quietly drop them; which is what had happened: the builders emitted a
    one-line note while the checked-in files carried a richer one, because neither builder wrote
    to the path its loader reads.
    """
    from pepdistill.data.prospect_catalog import load_catalog, load_shard_index

    for name, manifest in (("catalog", load_catalog()), ("shards", load_shard_index())):
        assert "pepdistill.data.prospect_catalog:build_" in manifest["_generator"], name
        assert manifest["_source"], name
        assert manifest["records"], name


def test_a_rebuilt_manifest_lands_where_its_loader_reads(tmp_path: Path):
    """The bug this guards: the catalog was written `.gz` but read plain, the shard index the
    reverse, so running either builder left the file being loaded untouched and its provenance
    free to drift.

    Writing to a temp directory and reading back through the same asset definition is what makes
    the two halves impossible to separate; asserting on the vendored files alone would pass even
    if the writer put its output somewhere else entirely.
    """
    from pepdistill.data import prospect_catalog as pc

    for asset in (pc._CATALOG_ASSET, pc._SHARDS_ASSET):
        payload = {"_source": "test", "_generator": "test", "records": {"r": [1, 2]}}
        written = pc._write_asset(asset, payload, str(tmp_path))
        assert written.is_file(), asset
        # The vendored copy sits at the same name the writer just produced.
        assert (Path(pc._VENDOR_DIR) / written.name).is_file(), asset
        # And the suffix the writer chose is the one the loader's rule resolves to.
        name, gzipped = asset
        assert written.name == name + (".gz" if gzipped else "")
