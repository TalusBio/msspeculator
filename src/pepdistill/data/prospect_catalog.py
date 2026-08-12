"""Vendored PROSPECT file and shard manifests.

Preparation uses these manifests as the source of truth.  The S3 prefix is only a read-through
cache; regeneration is explicit and networked via :func:`build_catalog` and
:func:`build_shard_index`.
"""

from __future__ import annotations

import json
import gzip
import time
import zipfile
from importlib.resources import files as _pkg_files

import fsspec

RECORDS = {
    "prospect": "6602020",
    "tmt": "8221499",
    "multi_ptm": "11472525",
    "tmt_ptm": "11474099",
    "test_ptm": "11477731",
}
_CATALOG_FILE = "prospect_catalog.json"
_SHARDS_FILE = "prospect_shards.json"


def load_catalog() -> dict:
    return json.loads(_pkg_files("pepdistill.data").joinpath(_CATALOG_FILE).read_text())


def load_shard_index() -> dict:
    asset = _pkg_files("pepdistill.data").joinpath(_SHARDS_FILE + ".gz")
    with asset.open("rb") as stream:
        return json.loads(gzip.decompress(stream.read()))


def build_catalog(out_path: str = "src/pepdistill/data/prospect_catalog.json") -> dict:
    """Refresh the checked-in file catalog from the Zenodo record APIs."""
    import urllib.request

    result: dict = {"_note": "Generated from Zenodo record metadata.", "records": {}}
    for name, record_id in RECORDS.items():
        with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{record_id}", timeout=60
        ) as stream:
            metadata = json.load(stream)
        result["records"][name] = {
            "record_id": record_id,
            "doi": metadata.get("doi", ""),
            "files": {
                entry["key"]: {
                    "size": entry.get("size", 0),
                    "checksum": entry.get("checksum", ""),
                    "url": entry.get("links", {}).get("content", ""),
                }
                for entry in metadata.get("files", [])
            },
        }
    with gzip.open(out_path + ".gz", "wt") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
    return result


def build_shard_index(
    out_path: str = "src/pepdistill/data/prospect_shards.json",
    delay_s: float = 2.0,
) -> dict:
    """Refresh shard names and packed/raw sizes from ZIP central directories.

    This performs range reads through fsspec and does not download complete archives.
    """
    catalog = load_catalog()["records"]
    result: dict = {"_note": "Generated from ZIP central directories.", "records": {}}
    for record, details in catalog.items():
        result["records"][record] = {}
        for filename, entry in sorted(details["files"].items()):
            if not filename.endswith(".zip"):
                continue
            try:
                with (
                    fsspec.open(entry["url"], "rb").open() as stream,
                    zipfile.ZipFile(stream) as archive,
                ):
                    result["records"][record][filename] = [
                        [info.filename, info.compress_size, info.file_size]
                        for info in archive.infolist()
                        if info.filename.endswith(".parquet")
                    ]
                print(f"{record}/{filename}: {len(result['records'][record][filename])} shards")
            except Exception as exc:  # noqa: BLE001 - preserve failures in the manifest
                result["records"][record][filename] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"{record}/{filename}: ERROR {type(exc).__name__}: {exc}")
            time.sleep(delay_s)
    with open(out_path, "w") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
    return result
