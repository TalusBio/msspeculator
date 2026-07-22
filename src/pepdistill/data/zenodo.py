"""Zenodo record access through the :class:`FileCache`.

The file listing is itself cached (`_files.json`), so a warm cache never contacts Zenodo —
neither for the listing nor the files. Downloads stream via fsspec's HTTP filesystem.
"""

from __future__ import annotations

import json

from .cache import FileCache, http_origin


class ZenodoRecord:
    API = "https://zenodo.org/api/records"

    def __init__(self, record_id: str, cache: FileCache) -> None:
        self.record_id = str(record_id)
        self.cache = cache

    def _prefix(self) -> str:
        return f"zenodo/{self.record_id}"

    def list_files(self) -> dict[str, str]:
        """Map ``filename -> download url`` (listing cached under ``_files.json``)."""
        key = f"{self._prefix()}/_files.json"
        raw = self.cache.get_bytes(key, http_origin(f"{self.API}/{self.record_id}"))
        meta = json.loads(raw)
        out: dict[str, str] = {}
        for f in meta.get("files", []):
            links = f.get("links", {})
            out[f["key"]] = links.get("content") or links.get("self", "")
        return out

    def file_url(self, filename: str) -> str:
        """Download URL for one record file (no fetch)."""
        files = self.list_files()
        if filename not in files:
            raise KeyError(f"{filename!r} not in record {self.record_id}; have {sorted(files)}")
        return files[filename]

    def resolve_file(self, filename: str) -> str:
        """Local path for one record file, fetched via cache tiers then Zenodo."""
        files = self.list_files()
        if filename not in files:
            raise KeyError(f"{filename!r} not in record {self.record_id}; have {sorted(files)}")
        return self.cache.resolve(
            f"{self._prefix()}/{filename}", http_origin(files[filename])
        )
