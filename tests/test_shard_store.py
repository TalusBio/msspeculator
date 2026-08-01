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
