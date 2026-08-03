"""Shard extraction: zip member -> local parquet, offline."""

import io
import os
import zipfile

import pandas as pd
import pyarrow as pa
import pytest

from pepdistill.data.cache import FileCache
from pepdistill.data.prospect import RECORDS, ProspectSource
from pepdistill.data.shard_store import (
    _read_raw_file_column,
    extract_shard,
    extract_shards,
    select_members,
    shard_raw_files,
)

ZIP_NAME = "TUM_third_pool.zip"  # exists in the 'prospect' catalog


def _frag_frame(pairs):
    """pairs: list of (raw_file, scan_number)."""
    return pd.DataFrame(
        {
            "ion_type": ["b"] * len(pairs),
            "no": [1] * len(pairs),
            "charge": [1] * len(pairs),
            "intensity": [1.0] * len(pairs),
            "neutral_loss": [""] * len(pairs),
            "scan_number": [s for _, s in pairs],
            "raw_file": [r for r, _ in pairs],
        }
    )


def _seed_zip(tmp_path):
    """Write a real zip into the local cache tier under the catalog's zip name.

    Member 0 deliberately holds TWO raw files whose names do not match the member stem —
    that is the real third-pool layout, and the shape that a filename-derived raw_file
    would get wrong.
    """
    root = tmp_path / "local" / "zenodo" / RECORDS["prospect"]
    os.makedirs(root, exist_ok=True)
    members = {
        "TUM_third_pool/pool_1_01_01_annotation.parquet": [
            ("01812a_GA3-pool_1_01_01-DDA-1h-R1", 1),
            ("01812a_GA3-pool_1_01_01-DDA-1h-R1", 2),
            ("01812a_GA3-pool_1_01_01-3xHCD-1h-R1", 5),
        ],
        "TUM_third_pool/pool_2_01_01_annotation.parquet": [
            ("01812a_GB3-pool_2_01_01-DDA-1h-R1", 3),
        ],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, pairs in members.items():
            b = io.BytesIO()
            _frag_frame(pairs).to_parquet(b)
            z.writestr(name, b.getvalue())
    (root / ZIP_NAME).write_bytes(buf.getvalue())
    return FileCache([str(tmp_path / "local")], write_through=False)


def test_shard_raw_files_reads_them_from_the_column_not_the_name(tmp_path):
    src = ProspectSource("prospect", cache=_seed_zip(tmp_path))
    path = extract_shard(src, ZIP_NAME, select_members(src, ZIP_NAME, [0])[0])
    assert sorted(shard_raw_files(path)) == [
        "01812a_GA3-pool_1_01_01-3xHCD-1h-R1",
        "01812a_GA3-pool_1_01_01-DDA-1h-R1",
    ]


def test_shard_raw_files_reads_the_column_dictionary_encoded(tmp_path):
    """Pins the ENCODING, not just the returned value.

    A future edit that drops ``read_dictionary=["raw_file"]`` would still pass the
    correctness test above (same unique values), but would reintroduce the ~1.1 GB peak-RSS
    regression on real shards by materializing one string per row instead of reading the
    on-disk dictionary. This asserts the column stays a pyarrow dictionary type.
    """
    src = ProspectSource("prospect", cache=_seed_zip(tmp_path))
    path = extract_shard(src, ZIP_NAME, select_members(src, ZIP_NAME, [0])[0])
    table = _read_raw_file_column(path)
    assert pa.types.is_dictionary(table.schema.field("raw_file").type)


def test_select_members_by_index(tmp_path):
    src = ProspectSource("prospect", cache=_seed_zip(tmp_path))
    assert select_members(src, ZIP_NAME, [1]) == [
        "TUM_third_pool/pool_2_01_01_annotation.parquet"
    ]


def test_select_members_rejects_out_of_range(tmp_path):
    src = ProspectSource("prospect", cache=_seed_zip(tmp_path))
    with pytest.raises(IndexError, match="shard index 5"):
        select_members(src, ZIP_NAME, [5])


def test_extract_writes_readable_parquet(tmp_path):
    src = ProspectSource("prospect", cache=_seed_zip(tmp_path))
    member = select_members(src, ZIP_NAME, [0])[0]
    path = extract_shard(src, ZIP_NAME, member)
    assert os.path.exists(path)
    assert list(pd.read_parquet(path).scan_number) == [1, 2, 5]


def test_extract_is_idempotent_and_cached(tmp_path):
    src = ProspectSource("prospect", cache=_seed_zip(tmp_path))
    member = select_members(src, ZIP_NAME, [0])[0]
    first = extract_shard(src, ZIP_NAME, member)
    mtime = os.path.getmtime(first)
    second = extract_shard(src, ZIP_NAME, member)
    assert first == second and os.path.getmtime(second) == mtime


def _seed_zip_n(tmp_path, n: int):
    """Like ``_seed_zip`` but with ``n`` distinct single-raw-file members, for batch tests
    that care about member *count* (opens-per-batch) rather than the third-pool multi-raw-file
    shape ``_seed_zip`` pins."""
    root = tmp_path / "local" / "zenodo" / RECORDS["prospect"]
    os.makedirs(root, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(n):
            b = io.BytesIO()
            _frag_frame([(f"raw_{i}", i)]).to_parquet(b)
            z.writestr(f"TUM_third_pool/shard_{i}_annotation.parquet", b.getvalue())
    (root / ZIP_NAME).write_bytes(buf.getvalue())
    return FileCache([str(tmp_path / "local")], write_through=False)


def _count_zip_opens(monkeypatch):
    """Monkeypatch ``ProspectSource._open_remote_zip`` to count calls, real behavior kept."""
    calls = {"n": 0}
    orig = ProspectSource._open_remote_zip

    def counting(self, zip_filename):
        calls["n"] += 1
        return orig(self, zip_filename)

    monkeypatch.setattr(ProspectSource, "_open_remote_zip", counting)
    return calls


def test_extract_shards_returns_paths_in_order_and_readable(tmp_path):
    src = ProspectSource("prospect", cache=_seed_zip_n(tmp_path, 3))
    members = select_members(src, ZIP_NAME, [0, 1, 2])
    paths = extract_shards(src, ZIP_NAME, members)
    assert len(paths) == 3
    for i, p in enumerate(paths):
        assert os.path.exists(p)
        assert list(pd.read_parquet(p).scan_number) == [i]


def test_extract_shards_reports_missing_member_progress(tmp_path):
    src = ProspectSource("prospect", cache=_seed_zip_n(tmp_path, 3))
    members = select_members(src, ZIP_NAME, [0, 1, 2])
    events = []
    byte_events = []
    extract_shards(
        src,
        ZIP_NAME,
        members,
        progress=lambda done, total, member: events.append(
            (done, total, member.split("/")[-1])
        ),
        byte_progress=lambda member, done, total: byte_events.append(
            (member.split("/")[-1], done, total)
        ),
    )
    assert [done for done, _, _ in events] == [1, 2, 3]
    assert all(total == 3 for _, total, _ in events)
    assert byte_events and all(done <= total for _, done, total in byte_events)


def test_extract_shards_opens_zip_at_most_once_for_a_cold_batch(tmp_path, monkeypatch):
    """The point of extract_shards: N shards must not cost N zip opens.

    A cold 3-member batch used to open the remote zip 4 times through extract_shard (one
    central-directory read via select_members plus one full re-open per shard) -- enough opens
    in a row, at ~10 shards per real source, to trip Zenodo's rate limiter on its own.
    """
    src = ProspectSource("prospect", cache=_seed_zip_n(tmp_path, 3))
    members = select_members(src, ZIP_NAME, [0, 1, 2])

    calls = _count_zip_opens(monkeypatch)
    paths = extract_shards(src, ZIP_NAME, members)

    assert calls["n"] == 1
    assert len(paths) == 3


def test_extract_shards_zero_opens_when_everything_cached(tmp_path, monkeypatch):
    """A fully-warm batch must make ZERO network requests -- what makes a retry cheap."""
    src = ProspectSource("prospect", cache=_seed_zip_n(tmp_path, 3))
    members = select_members(src, ZIP_NAME, [0, 1, 2])
    extract_shards(src, ZIP_NAME, members)  # warm every member

    calls = _count_zip_opens(monkeypatch)
    paths = extract_shards(src, ZIP_NAME, members)

    assert calls["n"] == 0
    assert len(paths) == 3


def test_extract_shards_partial_cache_extracts_only_missing(tmp_path, monkeypatch):
    src = ProspectSource("prospect", cache=_seed_zip_n(tmp_path, 3))
    members = select_members(src, ZIP_NAME, [0, 1, 2])
    extract_shard(src, ZIP_NAME, members[1])  # pre-warm only the middle member

    calls = _count_zip_opens(monkeypatch)
    paths = extract_shards(src, ZIP_NAME, members)

    assert calls["n"] == 1  # one open covers both still-missing members
    for i, p in enumerate(paths):
        assert list(pd.read_parquet(p).scan_number) == [i]


def test_extract_shards_reports_archive_error_with_missing_members(tmp_path, monkeypatch):
    src = ProspectSource("prospect", cache=_seed_zip_n(tmp_path, 3))
    members = select_members(src, ZIP_NAME, [0, 1, 2])

    def fail_open(_zip_filename):
        raise RuntimeError("Zenodo returned transient HTTP 500")

    monkeypatch.setattr(src, "_open_remote_zip", fail_open)
    with pytest.raises(RuntimeError, match="shard_0_annotation.parquet") as exc_info:
        extract_shards(src, ZIP_NAME, members)
    assert "3 missing shard(s)" in str(exc_info.value)
    assert "Original error: Zenodo returned transient HTTP 500" in str(exc_info.value)


def test_extract_shards_reports_member_error_with_shard_name(tmp_path):
    src = ProspectSource("prospect", cache=_seed_zip_n(tmp_path, 3))
    member = "TUM_third_pool/not_present_annotation.parquet"

    with pytest.raises(RuntimeError, match="not_present_annotation.parquet") as exc_info:
        extract_shards(src, ZIP_NAME, [member])
    assert "shard 1/1" in str(exc_info.value)
