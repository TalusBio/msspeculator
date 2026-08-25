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
from pathlib import Path
from importlib.resources import files as _pkg_files

import fsspec

RECORDS = {
    "prospect": "6602020",
    "tmt": "8221499",
    "multi_ptm": "11472525",
    "tmt_ptm": "11474099",
    "test_ptm": "11477731",
}
#: `(filename as vendored, gzipped)`. One definition per asset, read by the loaders and written
#: by the builders below, because they used to disagree: the catalog was written `.gz` and read
#: plain while the shard index was written plain and read `.gz`, so neither "regenerate with X"
#: instruction actually refreshed the file being loaded, and the vendored provenance notes drifted
#: away from what the builders emit without anything failing.
_CATALOG_ASSET = ("prospect_catalog.json", False)
_SHARDS_ASSET = ("prospect_shards.json", True)
_VENDOR_DIR = "src/pepdistill/data"


def _read_asset(asset: tuple[str, bool]) -> dict:
    name, gzipped = asset
    path = _pkg_files("pepdistill.data").joinpath(name + (".gz" if gzipped else ""))
    if not gzipped:
        return json.loads(path.read_text())
    with path.open("rb") as stream:
        return json.loads(gzip.decompress(stream.read()))


def _write_asset(asset: tuple[str, bool], payload: dict, out_dir: str = _VENDOR_DIR) -> Path:
    name, gzipped = asset
    out = Path(out_dir) / (name + (".gz" if gzipped else ""))
    text = json.dumps(payload, indent=1, sort_keys=True)
    if gzipped:
        with gzip.open(out, "wt") as stream:
            stream.write(text)
    else:
        out.write_text(text)
    return out


def _provenance(source: str, generator: str, note: str | None = None) -> dict:
    """Field-per-key rather than one run-on sentence, so a reader can find one fact."""
    fields = {"_source": source, "_generator": generator}
    if note is not None:
        fields["_note"] = note
    return fields


def load_catalog() -> dict:
    return _read_asset(_CATALOG_ASSET)


def load_shard_index() -> dict:
    return _read_asset(_SHARDS_ASSET)


def build_catalog(out_dir: str = _VENDOR_DIR) -> dict:
    """Refresh the checked-in file catalog from the Zenodo record APIs."""
    import urllib.request

    result: dict = {
        **_provenance(
            f"Zenodo record metadata API for records {', '.join(sorted(RECORDS.values()))}",
            "pepdistill.data.prospect_catalog:build_catalog",
        ),
        "records": {},
    }
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
    _write_asset(_CATALOG_ASSET, result, out_dir)
    return result


def build_shard_index(out_dir: str = _VENDOR_DIR, delay_s: float = 2.0) -> dict:
    """Refresh shard names and packed/raw sizes from ZIP central directories.

    This performs range reads through fsspec and does not download complete archives.
    """
    catalog = load_catalog()["records"]
    result: dict = {
        **_provenance(
            "ZIP central directories of the PROSPECT annotation archives, read by range request "
            "through fsspec -- no archive is downloaded. Entries are "
            "[name, packed_bytes, raw_bytes].",
            "pepdistill.data.prospect_catalog:build_shard_index",
            "Zenodo rate-limits; a throttled response surfaces as FileNotFoundError, so a failure "
            "recorded here may mean 'try again', not 'gone'.",
        ),
        "records": {},
    }
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
    _write_asset(_SHARDS_ASSET, result, out_dir)
    return result
