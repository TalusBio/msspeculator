"""FileCache tiering: cold-miss populates, warm hit skips origin, mirror promotes to local."""

import fsspec
import pytest

from pepdistill.data.cache import FileCache


def _origin(payload: bytes):
    """A fake origin that records how many times it was hit."""
    calls = {"n": 0}

    def fetch(dest: str) -> None:
        calls["n"] += 1
        with open(dest, "wb") as f:
            f.write(payload)

    return fetch, calls


def _mirror(tmp_path):
    # Unique in-process memory filesystem path standing in for s3://.
    return f"memory://{tmp_path.name}-mirror"


def test_cold_miss_populates_local_and_mirror(tmp_path):
    fsspec.filesystem("memory").store.clear()
    cache = FileCache([str(tmp_path / "local"), _mirror(tmp_path)], write_through=True)
    fetch, calls = _origin(b"hello")

    path = cache.resolve("zenodo/1/f.bin", fetch)
    assert open(path, "rb").read() == b"hello"
    assert calls["n"] == 1
    # Written through to the mirror tier.
    mem = fsspec.filesystem("memory")
    assert any("f.bin" in p for p in mem.find(_mirror(tmp_path).replace("memory://", "")))


def test_warm_local_hit_skips_origin(tmp_path):
    fsspec.filesystem("memory").store.clear()
    cache = FileCache([str(tmp_path / "local"), _mirror(tmp_path)])
    fetch, calls = _origin(b"data")
    cache.resolve("k/f.bin", fetch)
    cache.resolve("k/f.bin", fetch)
    assert calls["n"] == 1  # second call served from local


def test_mirror_promotes_without_origin(tmp_path):
    fsspec.filesystem("memory").store.clear()
    mirror = _mirror(tmp_path)
    # First cache instance populates the shared mirror.
    c1 = FileCache([str(tmp_path / "local1"), mirror], write_through=True)
    fetch, calls = _origin(b"shared")
    c1.resolve("k/f.bin", fetch)
    assert calls["n"] == 1

    # A fresh local tier + same mirror must serve from the mirror, not the origin.
    c2 = FileCache([str(tmp_path / "local2"), mirror])
    fetch2, calls2 = _origin(b"shared")
    path = c2.resolve("k/f.bin", fetch2)
    assert open(path, "rb").read() == b"shared"
    assert calls2["n"] == 0


def test_no_write_through_leaves_mirror_empty(tmp_path):
    fsspec.filesystem("memory").store.clear()
    mirror = _mirror(tmp_path)
    cache = FileCache([str(tmp_path / "local"), mirror], write_through=False)
    fetch, _ = _origin(b"x")
    cache.resolve("k/f.bin", fetch)
    mem = fsspec.filesystem("memory")
    assert not any("f.bin" in p for p in mem.find(mirror.replace("memory://", "")))


def test_missing_origin_on_miss_raises(tmp_path):
    fsspec.filesystem("memory").store.clear()
    cache = FileCache([str(tmp_path / "local")])

    def boom(dest):
        raise RuntimeError("origin unavailable")

    with pytest.raises(RuntimeError):
        cache.resolve("k/f.bin", boom)
