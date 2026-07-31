"""Extract annotation-zip members to local parquet.

PROSPECT annotation zips store their members DEFLATE-compressed, so there is no random access
inside the archive: reading one member inflates it from the start. Measured, that is 21s just
to reach the footer of a 90.7 MB shard, which makes per-epoch reads from the zip impossible.

So a selected shard is extracted ONCE to a local parquet file and every later read goes there.
The extracted file is a byte-for-byte copy of the zip member — no transcode and no schema of
our own — so the member name is the whole cache key and there is no staleness mode to manage.
"""

from __future__ import annotations

import pyarrow.parquet as pq

from .prospect import ProspectSource


def shard_raw_files(path: str) -> list[str]:
    """Distinct ``raw_file`` values inside an extracted shard.

    Read from the column, never inferred from the filename. A third-pool shard named
    ``TUM_third_pool_1_01_01_annotation.parquet`` actually holds three raw files
    (``01812a_GA3-…-DDA-1h-R1``, ``…-2xIT_2xHCD-1h-R1``, ``…-3xHCD-1h-R1``), none of them
    equal to the stem; TMT pools use the opposite convention. The column costs 226 bytes
    compressed in a 90.7 MB shard, so there is nothing to save by guessing.
    """
    table = pq.read_table(path, columns=["raw_file"])
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


def extract_shard(src: ProspectSource, zip_filename: str, member: str) -> str:
    """Local parquet path for one zip member, extracting it on first use."""
    key = f"shards/{src.record}/{zip_filename.removesuffix('.zip')}/{member.split('/')[-1]}"

    def origin(dest: str) -> None:
        with src._open_remote_zip(zip_filename) as z:
            with open(dest, "wb") as out:
                out.write(z.read(member))

    return src.cache.resolve(key, origin)


__all__ = ["shard_raw_files", "select_members", "extract_shard"]
