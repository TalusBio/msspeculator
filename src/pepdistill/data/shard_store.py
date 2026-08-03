"""Extract annotation-zip members to local parquet.

PROSPECT annotation zips store their members DEFLATE-compressed, so there is no random access
inside the archive: reading one member inflates it from the start. Measured, that is 21s just
to reach the footer of a 90.7 MB shard, which makes per-epoch reads from the zip impossible.

So a selected shard is extracted ONCE to a local parquet file and every later read goes there.
The extracted file is a byte-for-byte copy of the zip member — no transcode and no schema of
our own — so the member name is the whole cache key and there is no staleness mode to manage.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .prospect import ProspectSource


def _read_raw_file_column(path: str) -> pa.Table:
    """Single-column table for ``raw_file``, kept dictionary-encoded.

    WARNING: ``read_dictionary=["raw_file"]`` is not an optimization to drop casually. The
    column is dictionary/RLE-encoded on disk — 226 bytes compressed, 191 bytes uncompressed
    for a whole row group — but plain ``pq.read_table`` expands it into one Python string per
    row on the way out. On a real third-pool shard (8,725,131 rows) that expansion alone costs
    ~1.1 GB of peak RSS, in a pipeline whose entire purpose is bounding peak memory. Reading
    with ``read_dictionary`` keeps the column as indices + a small dictionary, so `.unique()`
    reads the dictionary instead of materializing a string per row.
    """
    return pq.read_table(path, columns=["raw_file"], read_dictionary=["raw_file"])


def shard_raw_files(path: str) -> list[str]:
    """Distinct ``raw_file`` values inside an extracted shard.

    Read from the column, never inferred from the filename. A third-pool shard named
    ``TUM_third_pool_1_01_01_annotation.parquet`` actually holds three raw files
    (``01812a_GA3-…-DDA-1h-R1``, ``…-2xIT_2xHCD-1h-R1``, ``…-3xHCD-1h-R1``), none of them
    equal to the stem; TMT pools use the opposite convention.
    """
    table = _read_raw_file_column(path)
    return table.column("raw_file").unique().to_pylist()


def select_members(src: ProspectSource, zip_filename: str, indices: list[int]) -> list[str]:
    """Resolve shard indices to member names, reading only the zip's central directory."""
    names = [s.name for s in src.annotation_shard_info(zip_filename)]
    out = []
    for i in indices:
        if not 0 <= i < len(names):
            raise IndexError(
                f"shard index {i} out of range for {zip_filename}: it has {len(names)} shards"
            )
        out.append(names[i])
    return out


def _shard_key(src: ProspectSource, zip_filename: str, member: str) -> str:
    return f"shards/{src.record}/{zip_filename.removesuffix('.zip')}/{member.split('/')[-1]}"


def extract_shards(
    src: ProspectSource,
    zip_filename: str,
    members: list[str],
    progress: Callable[[int, int, str], None] | None = None,
    byte_progress: Callable[[str, int, int], None] | None = None,
) -> list[str]:
    """Local parquet paths for ``members``, in the same order, opening the remote zip AT MOST
    ONCE for the whole batch.

    Opening a remote zip re-reads its central directory and, on a cold cache, re-opens the
    whole remote HTTP stream — several range requests each. Paying that once per shard instead
    of once per batch is exactly what tripped Zenodo's rate limiter on a real 19-shard run
    (``ClientResponseError: 429`` out of ``_open_remote_zip``). So: resolve every member's
    cache path FIRST, via :meth:`FileCache.probe`, which never touches the zip — and only open
    it if something actually needs it. A fully-cached batch therefore costs zero network
    requests, which is what makes retrying a run that died mid-way cheap instead of repeating
    the whole cost that killed it.
    """
    cache = src.cache
    keys = [_shard_key(src, zip_filename, m) for m in members]
    paths: list[str | None] = [cache.probe(k) for k in keys]
    missing = [i for i, p in enumerate(paths) if p is None]

    if missing:
        with src._open_remote_zip(zip_filename) as z:
            for done, i in enumerate(missing, start=1):
                member = members[i]

                # copyfileobj, not out.write(z.read(member)): the members are 90.7-387.6 MB,
                # and z.read() inflates the whole thing into RAM before a byte is written. The
                # point of extracting is to bound peak RSS, so the extract must be streaming.
                def write(dest: str, member: str = member) -> None:
                    total_bytes = z.getinfo(member).file_size
                    copied = 0
                    with z.open(member) as member_f, open(dest, "wb") as out:
                        while chunk := member_f.read(8 * 1024 * 1024):
                            out.write(chunk)
                            copied += len(chunk)
                            if byte_progress is not None:
                                byte_progress(member, copied, total_bytes)

                paths[i] = cache.store(keys[i], write)
                if progress is not None:
                    progress(done, len(missing), member)

    return paths  # type: ignore[return-value]  # every entry was filled above


def extract_shard(src: ProspectSource, zip_filename: str, member: str) -> str:
    """Local parquet path for one zip member, extracting it on first use."""
    return extract_shards(src, zip_filename, [member])[0]


__all__ = ["shard_raw_files", "select_members", "extract_shard", "extract_shards"]
