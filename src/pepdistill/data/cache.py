"""Multi-tier read-through file cache over fsspec.

Motivation: PROSPECT lives on Zenodo. Hitting Zenodo on every run is slow and rude, so we
put an S3 (or any fsspec) mirror in front of it. Tiers are probed fastest-first; the origin
(Zenodo) is the last resort, and a cold fetch populates the writable tiers so the whole team
pays the Zenodo cost once.

Layout convention: ``tiers[0]`` is a LOCAL directory (the materialization target) — parquet
readers want a real filesystem path. Later tiers (e.g. ``s3://bucket/prefix``) are shared
mirrors. ``resolve`` always returns a local path.

An origin is ``Callable[[str], None]`` that writes the file to the local path it's handed —
streaming, so we never hold a multi-GB parquet in memory.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable

import fsspec


class FileCache:
    def __init__(self, tiers: list[str], write_through: bool = True) -> None:
        if not tiers:
            raise ValueError("need at least one cache tier (a local directory)")
        self.tiers = [t.rstrip("/") for t in tiers]
        self.write_through = write_through
        self._local_fs = fsspec.filesystem("file")
        self._local_root = self.tiers[0]

    def _split(self, tier: str, key: str):
        """(filesystem, full-path) for ``key`` under ``tier``."""
        fs, _, paths = fsspec.get_fs_token_paths(f"{tier}/{key}")
        return fs, paths[0]

    def _local_path(self, key: str) -> str:
        return f"{self._local_root}/{key}"

    def probe(self, key: str) -> str | None:
        """Local path for ``key`` if resolvable WITHOUT an origin call, else ``None``.

        Checks local first, then promotes from the first remote tier that has it — the same
        two steps ``resolve`` takes before it would fall back to origin. Split out so a caller
        that is about to make an expensive origin call spanning several keys at once (e.g.
        opening one remote zip to extract several members) can first find out, for free, which
        keys need that call at all and which are already served.
        """
        local = self._local_path(key)
        if self._local_fs.exists(local):
            return local
        self._local_fs.makedirs(os.path.dirname(local) or ".", exist_ok=True)

        # Probe the remaining (remote) tiers; first hit is copied down to local.
        for tier in self.tiers[1:]:
            fs, path = self._split(tier, key)
            try:
                if fs.exists(path):
                    fs.get_file(path, local)
                    return local
            except Exception:  # unreachable/unauthorized tier — skip, try the next
                continue
        return None

    def store(self, key: str, write: Callable[[str], None]) -> str:
        """Atomically materialize ``key`` locally via ``write``, then promote to remote tiers.

        ``write`` gets a temp path to write to; on success it is moved into place with
        ``os.replace`` (atomic on the same filesystem), so a crash or exception mid-write
        cannot leave a truncated file at ``key``'s local path for a later reader to trust. On
        failure the temp file is removed and the exception propagates — no partial file, no
        swallowed error.
        """
        local = self._local_path(key)
        fd, tmp = tempfile.mkstemp(dir=self._local_root)
        os.close(fd)
        try:
            write(tmp)
            os.replace(tmp, local)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

        if self.write_through:
            self._populate_remote(key, local)
        return local

    def resolve(self, key: str, origin: Callable[[str], None]) -> str:
        """Return a local path for ``key``, fetching/promoting through the tiers as needed."""
        cached = self.probe(key)
        if cached is not None:
            return cached
        # Full miss: origin writes straight to a temp file, then atomically moves into local.
        return self.store(key, origin)

    def _populate_remote(self, key: str, local: str) -> None:
        """Best-effort upload of a freshly-fetched file to the writable remote tiers."""
        for tier in self.tiers[1:]:
            fs, path = self._split(tier, key)
            try:
                parent = path.rsplit("/", 1)[0]
                if parent:
                    fs.makedirs(parent, exist_ok=True)
                fs.put_file(local, path)
            except Exception:  # read-only mirror or no write creds — fine, skip
                continue

    def get_bytes(self, key: str, origin: Callable[[str], None]) -> bytes:
        """Convenience for small blobs (e.g. a Zenodo file listing)."""
        path = self.resolve(key, origin)
        return self._local_fs.cat_file(path)


def default_cache_dir() -> str:
    """Local cache root: $PEPDISTILL_CACHE or ~/.cache/pepdistill."""
    return os.environ.get(
        "PEPDISTILL_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "pepdistill")
    )


def http_origin(url: str) -> Callable[[str], None]:
    """Origin that streams an HTTP(S) URL to a local dest path (via fsspec)."""

    def _fetch(dest: str) -> None:
        with fsspec.open(url, "rb") as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)

    return _fetch
